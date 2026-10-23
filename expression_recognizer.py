"""
昇腾 NPU 表情识别模块
基于 FER+ 模型和 Ascend ACL 推理
"""
import os
import numpy as np
import cv2
import aclruntime
from aclruntime import InferenceSession

from camera_util import open_camera


class ExpressionRecognizer:
    """表情识别器 - 基于昇腾 NPU 加速"""

    # FER+ 8 类表情标签
    LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

    def __init__(self, model_path, device_id=0, input_size=64,
                 detect_target=384, face_padding=0.15,
                 detector='yunet', yunet_model=None):
        """
        初始化表情识别器

        Args:
            model_path: OM 模型文件路径
            device_id: NPU 设备 ID (默认 0)
            input_size: 模型输入尺寸 (默认 64x64)
            detect_target: Haar 检测缩放后的长边像素（仅在 YuNet 不可用时使用）
            face_padding: 人脸框扩边比例。YuNet 框本身已略松，默认值应按
                检测器调整；Haar 框紧贴五官，需 0.15 左右扩边
            detector: 'yunet' 优先用 NPU YuNet，失败回退 Haar
            yunet_model: YuNet OM 路径，None 时按 models/yunet_640.om 推导
        """
        self.model_path = model_path
        self.device_id = device_id
        self.input_size = input_size
        self.detect_target = detect_target
        self.face_padding = face_padding

        # 初始化昇腾推理会话（表情）
        options = aclruntime.session_options()
        self.session = InferenceSession(model_path, device_id, options)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 人脸检测：优先 NPU YuNet（10.5ms），不可用时回退 Haar
        self.yunet = None
        self.face_cascade = None
        if yunet_model is None:
            yunet_model = os.path.join(os.path.dirname(os.path.abspath(model_path)),
                                       'yunet_640.om')
        if detector == 'yunet' and os.path.exists(yunet_model):
            try:
                from yunet_detector import YuNetDetector
                self.yunet = YuNetDetector(yunet_model, device_id=device_id)
            except Exception as e:
                print('[WARN] YuNet 初始化失败，回退 Haar: %s' % e)
        if self.yunet is None:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

    def detect_face(self, image):
        """
        检测图像中最大的人脸

        YuNet 走 NPU（640 输入，单次约 10.5ms），Haar 为回退方案
        （在缩小图上检测，720p 下约 36ms）。

        Args:
            image: BGR 格式的 OpenCV 图像

        Returns:
            原图坐标下的人脸区域 (x, y, w, h) 或 None
        """
        if self.yunet is not None:
            box, _score = self.yunet.detect(image)
            return box

        h, w = image.shape[:2]
        # 长边缩到 detect_target 做人脸检测；太小的图直接用原图。
        # 实测 720p 下：target=256 约 13ms、384 约 36ms、512 约 55ms。
        target_w = self.detect_target
        if w > target_w:
            scale = target_w / float(w)
            small = cv2.resize(image, (target_w, int(round(h * scale))))
        else:
            scale = 1.0
            small = image

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # 直方图均衡化，提升弱光下检出率
        gray = cv2.equalizeHist(gray)
        # sf=1.2/mn=3：速度与召回率的折中点（sf=1.3 会漏检逆光人脸，
        # sf=1.1 慢约一倍）；配合 equalizeHist。
        min_side = 30 if scale >= 1.0 else max(12, int(30 * scale))
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=3, minSize=(min_side, min_side)
        )
        if len(faces) == 0:
            return None
        # 返回最大的人脸，并把坐标/尺寸映射回原图
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        inv = 1.0 / scale
        return (int(round(fx * inv)), int(round(fy * inv)),
                int(round(fw * inv)), int(round(fh * inv)))

    def expand_box(self, box, shape, pad=None):
        """
        按比例扩边人脸框，并裁剪到图像范围内。

        Haar 检出的框通常紧贴五官，而 FER+ 训练样本包含更多额头/下巴。
        扩边后再缩放到 64x64，输入分布更接近训练集，是提升准确率的
        低成本手段（不增加任何推理开销）。

        Args:
            box: 人脸框 (x, y, w, h)，原图坐标
            shape: 图像 shape（用于边界裁剪）
            pad: 扩边比例，None 时用 self.face_padding

        Returns:
            扩边后的 (x, y, w, h)
        """
        pad = self.face_padding if pad is None else pad
        if not pad or pad <= 0:
            return box
        x, y, w, h = box
        dx, dy = int(round(w * pad)), int(round(h * pad))
        ih, iw = shape[:2]
        x0, y0 = max(0, x - dx), max(0, y - dy)
        x1, y1 = min(iw, x + w + dx), min(ih, y + h + dy)
        return (x0, y0, x1 - x0, y1 - y0)

    def preprocess(self, face_img):
        """
        预处理人脸图像为模型输入格式

        Args:
            face_img: 人脸区域图像 (BGR)

        Returns:
            预处理后的 numpy 数组 [1, 1, 64, 64]
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.input_size, self.input_size))
        # FER+ 训练时使用 0-255 原始像素尺度（FER2013 像素即 0-255，CNTK 管线
        # 未做 /255）。经板端实测：喂 0-1 时模型对任意输入都恒定输出
        # neutral~74%/sadness~21%（跨输入标准差 0.0004，等同瘫痪）；0-255 时
        # 输出随输入正常分化（标准差 0.0377）。切勿除以 255。
        face_data = resized.astype(np.float32)
        # 添加 batch 和 channel 维度
        return face_data.reshape(1, 1, self.input_size, self.input_size)

    def postprocess(self, output):
        """
        后处理模型输出

        Args:
            output: 模型原始输出 [1, 8]

        Returns:
            (predicted_label, confidence, all_probabilities)
        """
        probs = output[0]
        probs = np.exp(probs) / np.sum(np.exp(probs))  # softmax
        pred_idx = np.argmax(probs)
        return self.LABELS[pred_idx], float(probs[pred_idx]), probs.tolist()

    def predict(self, image):
        """
        预测图像中人物的表情

        Args:
            image: BGR 格式的 OpenCV 图像

        Returns:
            dict: {
                'success': bool,
                'expression': str,      # 预测表情
                'confidence': float,    # 置信度
                'probabilities': list,  # 各类别概率
                'face_box': tuple,      # 人脸框 (x, y, w, h)
                'message': str          # 错误信息 (如有)
            }
        """
        result = {
            'success': False,
            'expression': None,
            'confidence': 0.0,
            'probabilities': [0.0] * len(self.LABELS),
            'face_box': None,
            'message': ''
        }

        # 检测人脸
        face_box = self.detect_face(image)
        if face_box is None:
            result['message'] = 'No face detected'
            return result

        # 检测框扩边后再裁图：Haar 框偏紧，扩边使输入分布更接近 FER+ 训练集
        bx, by, bw, bh = self.expand_box(face_box, image.shape)
        face_img = image[by:by + bh, bx:bx + bw]

        # 预处理
        input_tensor = self.preprocess(face_img)

        # NPU 推理
        try:
            input_acl_tensor = aclruntime.Tensor(input_tensor)
            outputs = self.session.run(
                [self.output_name], {self.input_name: input_acl_tensor}
            )
            # 输出 tensor 在 NPU 上，需先 to_host()（原地转换）再取 numpy
            out_tensor = outputs[0]
            out_tensor.to_host()
            logits = np.array(out_tensor)
            pred, conf, probs = self.postprocess(logits)
        except Exception as e:
            result['message'] = f'Inference error: {str(e)}'
            return result

        result.update({
            'success': True,
            'expression': pred,
            'confidence': conf,
            'probabilities': probs,
            'face_box': (int(face_box[0]), int(face_box[1]),
                         int(face_box[2]), int(face_box[3])),
            'message': 'OK'
        })
        return result

    def predict_from_file(self, image_path):
        """
        从文件路径预测表情

        Args:
            image_path: 图像文件路径

        Returns:
            同 predict() 返回结果
        """
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'expression': None,
                'confidence': 0.0,
                'probabilities': [0.0] * len(self.LABELS),
                'face_box': None,
                'message': f'Cannot read image: {image_path}'
            }
        return self.predict(image)

    def predict_from_camera(self, camera_id=0, show_preview=True):
        """
        实时摄像头表情识别

        Args:
            camera_id: 摄像头设备 ID
            show_preview: 是否显示预览窗口

        Returns:
            最后预测结果
        """
        cap = open_camera(camera_id, 1280, 720)
        if not cap.isOpened():
            return {'success': False, 'message': f'Cannot open camera {camera_id}'}

        result = {'success': False, 'message': 'No frame captured'}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.predict(frame)

            if result['success']:
                x, y, w, h = result['face_box']
                label = f"{result['expression']} ({result['confidence']:.2f})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if show_preview:
                cv2.imshow('Expression Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # 无头模式只处理一帧
                break

        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        return result

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'session'):
            del self.session


if __name__ == '__main__':
    # 简单测试
    import sys

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'emotion-ferplus.om')
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        recognizer = ExpressionRecognizer(model_path)
        result = recognizer.predict_from_file(image_path)
        print(result)
    else:
        print("Usage: python expression_recognizer.py <image_path>")
