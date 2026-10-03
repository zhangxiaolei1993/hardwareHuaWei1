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

    # Create venv
    exec_cmd(client,
             "/usr/local/miniconda3/bin/python -m venv /home/HwHiAiUser/expression_recognition/venv --system-site-packages 2>&1",
             "create venv")

    # Verify venv python
    exec_cmd(client,
             "/home/HwHiAiUser/expression_recognition/venv/bin/python --version",
             "venv python version")

    # Verify venv deps
    exec_cmd(client,
             "/home/HwHiAiUser/expression_recognition/venv/bin/python -c \"import cv2, numpy, aclruntime; print('deps OK', cv2.__version__, numpy.__version__)\"",
             "venv deps")

    client.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
