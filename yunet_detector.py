"""
YuNet 人脸检测器（昇腾 NPU / OM）

YuNet 2023mar 为 anchor-free 人脸检测，输出三个 stride(8/16/32) 的
cls/obj/bbox/kps。OM 已将 sigmoid 固化在图中，输出直接为 0~1 概率。

实测：640x640 输入单次推理 10.5ms（95 fps），检测含预处理/解码约 24ms，
相比 CPU onnxruntime 的 290ms 快约 12 倍，可与人脸表情推理共用 NPU。
"""
import cv2
import numpy as np
import aclruntime


class YuNetDetector:
    """YuNet 人脸检测器 - 昇腾 NPU 加速"""

    STRIDES = (8, 16, 32)
    # 各 stride 的基准人脸尺度（输入图像素，按 640 输入实测校准）
    # 实测：1280x720 缩到 640 后 IoU 0.68，小脸(153x86)放大后 IoU 0.75
    MIN_SIZES = (24.0, 64.0, 128.0)

    def __init__(self, model_path, device_id=0, input_size=640,
                 score_thr=0.6, nms_thr=0.3):
        """
        Args:
            model_path: YuNet OM 模型路径
            device_id: NPU 设备 ID
            input_size: 模型输入边长（需与 OM 一致）
            score_thr: 置信度阈值（sqrt(cls*obj)）
            nms_thr: NMS IoU 阈值
        """
        self.model_path = model_path
        self.device_id = device_id
        self.input_size = int(input_size)
        self.score_thr = score_thr
        self.nms_thr = nms_thr

        options = aclruntime.session_options()
        self.session = aclruntime.InferenceSession(model_path, device_id, options)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        # 输出名形如 'Sigmoid_85:0:cls_8'，取末段作为键
        self._out = {n.split(':')[-1]: i for i, n in enumerate(self.output_names)}

    def _decode(self, res):
        """解码三个 stride 的输出，返回输入图坐标系下的 (boxes, scores)"""
        n = self.input_size
        boxes, scores = [], []
        for si, stride in enumerate(self.STRIDES):
            fm = n // stride
            cls = res[self._out['cls_%d' % stride]][0, :, 0]
            obj = res[self._out['obj_%d' % stride]][0, :, 0]
            bb = res[self._out['bbox_%d' % stride]][0]
            # OM 输出已过 sigmoid，无需再次激活
            sc = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
            ks = np.where(sc >= self.score_thr)[0]
            if ks.size == 0:
                continue
            pw = self.MIN_SIZES[si] / n
            jj, ii = np.divmod(ks, fm)
            cx = bb[ks, 0] * 0.1 * pw + (ii + 0.5) * stride / n
            cy = bb[ks, 1] * 0.1 * pw + (jj + 0.5) * stride / n
            bw = np.exp(bb[ks, 2] * 0.2) * pw
            bh = np.exp(bb[ks, 3] * 0.2) * pw
            for k in range(ks.size):
                boxes.append([float((cx[k] - bw[k] / 2) * n),
                              float((cy[k] - bh[k] / 2) * n),
                              float(bw[k] * n), float(bh[k] * n)])
            scores.extend(sc[ks].tolist())
        return boxes, scores

    def detect(self, image, letterbox=True):
        """
        检测图像中最大的人脸

        Args:
            image: BGR 图像
            letterbox: True 时等比缩放填充（不变形，精度更稳）；
                False 时直接拉伸到 input_size（速度略快、有纵横比畸变）

        Returns:
            (x, y, w, h, score)，原图坐标；未检出返回 (None, 0.0)
        """
        ih, iw = image.shape[:2]
        n = self.input_size
        if letterbox:
            s = min(n / float(iw), n / float(ih))
            nw, nh = int(round(iw * s)), int(round(ih * s))
            canvas = np.zeros((n, n, 3), np.uint8)
            pad_x, pad_y = (n - nw) // 2, (n - nh) // 2
            canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(image, (nw, nh))
            src = canvas
        else:
            s, pad_x, pad_y = None, 0, 0
            src = cv2.resize(image, (n, n))

        # blobFromImage 在 C++ 侧一次性完成 RGB 交换/NCHW 排布/float32 转换，
        # 比 numpy transpose + ascontiguousarray 少一次 4.9MB 全图拷贝
        blob = cv2.dnn.blobFromImage(src, 1.0, (n, n), (0, 0, 0),
                                     swapRB=True, crop=False)

        outs = self.session.run(self.output_names,
                                {self.input_name: aclruntime.Tensor(blob)})
        for t in outs:
            t.to_host()
        res = [np.array(t) for t in outs]

        boxes, scores = self._decode(res)
        if not boxes:
            return None, 0.0

        idx = cv2.dnn.NMSBoxes(boxes, scores, self.score_thr, self.nms_thr)
        if len(idx) == 0:
            return None, 0.0
        idx = np.array(idx).flatten()
        # 取面积最大的框（与 Haar 版行为一致：聚焦画面主体）
        best = max(idx, key=lambda i: boxes[i][2] * boxes[i][3])
        bx, by, bw, bh = boxes[best]

        if letterbox:
            bx, by = (bx - pad_x) / s, (by - pad_y) / s
            bw, bh = bw / s, bh / s
        else:
            bx, by = bx * iw / n, by * ih / n
            bw, bh = bw * iw / n, bh * ih / n

        x0, y0 = max(0, int(round(bx))), max(0, int(round(by)))
        x1, y1 = min(iw, int(round(bx + bw))), min(ih, int(round(by + bh)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None, 0.0
        return (x0, y0, x1 - x0, y1 - y0), float(scores[best])

    def __del__(self):
        if hasattr(self, 'session'):
            del self.session
