import paramiko
import sys

HOST = "192.168.137.2"
USER = "root"
PASSWORD = "Mind@123"

COMMANDS = [
    ("安装onnx阿里云", "/usr/local/miniconda3/bin/pip install onnx -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -q 2>&1 | tail -n 5"),
    ("检查模型结构", "/usr/local/miniconda3/bin/python -c \"import onnx; m=onnx.load('/home/HwHiAiUser/expression_recognition/models/emotion-ferplus-8.onnx'); print('输入:'); [print(' ', i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]; print('输出:'); [print(' ', o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]\" 2>&1"),
]

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASSWORD, timeout=10, look_for_keys=False, allow_agent=False)
    except Exception as e:
        print(f"SSH连接失败: {e}")
        return 1

    for title, cmd in COMMANDS:
        print(f"\n===== {title} =====")
        stdin, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode(errors="replace").strip()
        err = stderr.read().decode(errors="replace").strip()
        if out:
            print(out)
        if err:
            print(f"(stderr) {err}")

    client.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
