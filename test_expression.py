#!/usr/bin/env python3
"""表情识别测试脚本 - 使用摄像头"""
import os
import sys

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expression_recognizer import ExpressionRecognizer


def test_camera():
    """测试摄像头实时表情识别"""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'emotion-ferplus.om')

    print("Initializing ExpressionRecognizer...")
    recognizer = ExpressionRecognizer(model_path, device_id=0)

    print("Starting camera... Press 'q' to quit")
    result = recognizer.predict_from_camera(camera_id=0, show_preview=True)

    print("\nFinal result:", result)


def test_image(image_path):
    """测试单张图片"""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'emotion-ferplus.om')

    recognizer = ExpressionRecognizer(model_path, device_id=0)
    result = recognizer.predict_from_file(image_path)

    print(f"\nImage: {image_path}")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Expression: {result['expression']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Face box: {result['face_box']}")
        print(f"All probabilities:")
        for label, prob in zip(ExpressionRecognizer.LABELS, result['probabilities']):
            print(f"  {label:12s}: {prob:.2%}")
    else:
        print(f"Error: {result['message']}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_image(sys.argv[1])
    else:
        test_camera()
