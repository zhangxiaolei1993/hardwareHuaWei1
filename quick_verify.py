import paramiko

HOST = "192.168.137.2"
USER = "root"
PASSWORD = "Mind@123"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASSWORD, timeout=15, look_for_keys=False, allow_agent=False)

# Test 1: Import check
cmd1 = '''source /usr/local/Ascend/ascend-toolkit/set_env.sh && cd /home/HwHiAiUser/expression_recognition && ./venv/bin/python -c "from expression_recognizer import ExpressionRecognizer; print('Import OK')" 2>&1'''
stdin, stdout, stderr = client.exec_command(cmd1, timeout=30)
print("=== Import Test ===")
print(stdout.read().decode(errors='replace').strip())
err = stderr.read().decode(errors='replace').strip()
if err:
    print(f"(stderr) {err}")
print(f"Exit: {stdout.channel.recv_exit_status()}")

# Test 2: Full pipeline test
cmd2 = '''source /usr/local/Ascend/ascend-toolkit/set_env.sh && cd /home/HwHiAiUser/expression_recognition && ./venv/bin/python << 'PYEOF'
import numpy as np
from expression_recognizer import ExpressionRecognizer

# Create recognizer
rec = ExpressionRecognizer('models/emotion-ferplus.om', device_id=0)
print('Model loaded OK')

# Create dummy image
img = np.ones((200, 200, 3), dtype=np.uint8) * 200
import cv2
cv2.circle(img, (100, 100), 60, (220, 220, 220), -1)
cv2.circle(img, (80, 80), 8, (50, 50, 50), -1)
cv2.circle(img, (120, 80), 8, (50, 50, 50), -1)

# Run prediction
result = rec.predict(img)
print('Prediction result:', result)
PYEOF'''
stdin2, stdout2, stderr2 = client.exec_command(cmd2, timeout=60)
print("\n=== Full Pipeline Test ===")
print(stdout2.read().decode(errors='replace').strip())
err2 = stderr2.read().decode(errors='replace').strip()
if err2:
    print(f"(stderr) {err2}")
print(f"Exit: {stdout2.channel.recv_exit_status()}")

client.close()
print("\nVerification complete.")
