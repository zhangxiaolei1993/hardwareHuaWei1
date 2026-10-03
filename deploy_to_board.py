import paramiko
import sys

HOST = "192.168.137.2"
USER = "root"
PASSWORD = "Mind@123"


def exec_cmd(client, cmd, title="", timeout=300):
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

    # Convert OM
    exec_cmd(client,
             "source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
             "cd /home/HwHiAiUser/expression_recognition/models && "
             "/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc "
             "--model=emotion-ferplus-8.onnx "
             "--framework=5 "
             "--output=emotion-ferplus "
             "--input_shape='Input3:1,1,64,64' "
             "--soc_version=Ascend310B4 "
             "--input_format=ND 2>&1",
             "ATC convert", timeout=300)

    # Check result
    exec_cmd(client,
             "ls -la /home/HwHiAiUser/expression_recognition/models/",
             "check output")

    client.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
