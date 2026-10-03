"""下载真实人脸样例图、上传到开发板并运行表情识别测试"""
import os
import ssl
import urllib.request

import paramiko

HOST = "192.168.137.2"
USER = "root"
PASSWORD = "Mind@123"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

# OpenCV 官方样例中的真实人脸照片
FACE_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg",
]
REMOTE_DIR = "/home/HwHiAiUser/expression_recognition"
REMOTE_IMG = REMOTE_DIR + "/test_data/real_face.jpg"


def download_image():
    local_path = os.path.join(LOCAL_DIR, "real_face.jpg")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for url in FACE_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
                data = r.read()
            if len(data) > 10000:
                with open(local_path, "wb") as f:
                    f.write(data)
                print(f"下载成功: {url} ({len(data)} bytes)")
                return local_path
            print(f"文件过小，跳过: {len(data)} bytes")
        except Exception as e:
            print(f"下载失败 {url}: {e}")
    return None


def run_on_board(local_path):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15,
                   look_for_keys=False, allow_agent=False)

    # 上传图片
    sftp = client.open_sftp()
    sftp.put(local_path, REMOTE_IMG)
    sftp.close()
    print(f"已上传: {REMOTE_IMG}")

    # 运行测试
    cmd = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
        f"cd {REMOTE_DIR} && "
        "./venv/bin/python test_expression.py test_data/real_face.jpg 2>&1"
    )
    stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    print("\n===== 推理结果 =====")
    print(out)
    if err:
        print(f"(stderr) {err}")
    print("Exit:", stdout.channel.recv_exit_status())
    client.close()


if __name__ == "__main__":
    path = download_image()
    if path:
        run_on_board(path)
    else:
        print("所有下载源均失败")
