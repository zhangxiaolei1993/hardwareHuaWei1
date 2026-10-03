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

    # Create activation script
    script_content = '''#!/bin/bash
cd /home/HwHiAiUser/expression_recognition
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source venv/bin/activate
export PYTHONPATH=/usr/local/miniconda3/lib/python3.9/site-packages:$PYTHONPATH
'''
    exec_cmd(client,
             f"cat > /home/HwHiAiUser/expression_recognition/activate.sh << 'SCRIPT_EOF'\n{script_content}\nSCRIPT_EOF",
             "create activate.sh")

    exec_cmd(client, "chmod +x /home/HwHiAiUser/expression_recognition/activate.sh", "chmod")

    # Quick test import
    exec_cmd(client,
             "source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
             "/home/HwHiAiUser/expression_recognition/venv/bin/python -c \"import aclruntime; print('aclruntime OK')\" 2>&1",
             "test aclruntime")

    # List final files
    exec_cmd(client,
             "ls -la /home/HwHiAiUser/expression_recognition/",
             "final files")

    client.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
