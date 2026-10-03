import paramiko
import sys

HOST = "192.168.137.2"
USER = "root"
PASSWORD = "Mind@123"


def exec_cmd(client, cmd, title="", timeout=120):
    print(f"\n===== {title or cmd[:50]} =====")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    if out:
        print(out)
    if err:
        print(f"(stderr) {err}")
    print(f"Exit code: {exit_code}")
    return exit_code, out, err


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15, look_for_keys=False, allow_agent=False)
    print("SSH connected as root")

    # Generate test image
    exec_cmd(client,
             "/home/HwHiAiUser/expression_recognition/venv/bin/python << 'PYEOF'\n"
             "import cv2\n"
             "import numpy as np\n"
             "img = np.ones((200, 200, 3), dtype=np.uint8) * 200\n"
             "cv2.circle(img, (100, 100), 60, (220, 220, 220), -1)\n"
             "cv2.circle(img, (80, 80), 8, (50, 50, 50), -1)\n"
             "cv2.circle(img, (120, 80), 8, (50, 50, 50), -1)\n"
             "cv2.ellipse(img, (100, 120), (20, 10), 0, 0, 180, (50, 50, 50), 2)\n"
             "cv2.imwrite('/home/HwHiAiUser/expression_recognition/test_data/test_face.jpg', img)\n"
             "print('Test image created')\n"
             "PYEOF",
             "generate test image")

    # Run test
    exec_cmd(client,
             "source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
             "cd /home/HwHiAiUser/expression_recognition && "
             "./venv/bin/python test_expression.py test_data/test_face.jpg 2>&1",
             "run inference test")

    client.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
