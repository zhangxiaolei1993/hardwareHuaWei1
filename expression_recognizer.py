"""
昇腾 NPU 表情识别模块
基于 FER+ 模型和 Ascend ACL 推理
"""
import os
import numpy as np
import cv2
import aclruntime
from aclruntime import InferenceSession


class ExpressionRecognizer:
    """表情识别器 - 基于昇腾 NPU 加速"""

    # FER+ 8 类表情标签
    LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

    def __init__(self, model_path, device_id=0, input_size=64):
        """
        初始化表情识别器

        Args:
            model_path: OM 模型文件路径
            device_id: NPU 设备 ID (默认 0)
            input_size: 模型输入尺寸 (默认 64x64)
        """
        self.model_path = model_path
        self.device_id = device_id
        self.input_size = input_size

        # 加载人脸检测器 (OpenCV Haar)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # 初始化昇腾推理会话
        options = aclruntime.session_options()
        self.session = InferenceSession(model_path, device_id, options)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect_face(self, image):
        """
        检测图像中的人脸（在缩小图上检测以提速，坐标映射回原图）

        实测：640x480 全分辨率 Haar 约 135ms，
        缩到约 320 宽 + 快速参数仅约 12ms，提速约 10 倍。

        Args:
            image: BGR 格式的 OpenCV 图像

        Returns:
            原图坐标下的人脸区域 (x, y, w, h) 或 None
        """
        h, w = image.shape[:2]
        # 长边缩到 256 左右做人脸检测；太小的图直接用原图。
        # 实测 256 宽 + sf=1.2 单次约 25ms（320 宽 + sf=1.1 需 62ms），
        # 检出率在逆光侧脸样本上与更慢参数基本一致。
        target_w = 256
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

        x, y, w, h = face_box
        face_img = image[y:y + h, x:x + w]

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
            'face_box': (int(x), int(y), int(w), int(h)),
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
        cap = cv2.VideoCapture(camera_id)
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
