import paho.mqtt.client as mqtt
import json
import time

# MQTT 服务器配置
MQTT_BROKER = "192.168.31.145"  # 机械臂的IP地址
MQTT_PORT = 1883  # 端口
MQTT_KEEP_ALIVE_INTERVAL = 60  # 保持连接时间间隔
CLIENT_ID = "arm_client_001"  # 客户端ID

# 定义 topic
# 发送
TOPIC_COMMAND = "arm_card_dealer/command"
TOPIC_CAMERA_RESPONSE = "arm_card_dealer/camera/response"

# 接收
TOPIC_STATUS = "arm_card_dealer/status"
TOPIC_ERROR = "arm_card_dealer/error"
TOPIC_CAMERA_COMMAND = "arm_card_dealer/camera/command"


# 定义发送指令函数
def send_command(cmd, request_id, cycles=-1):
    payload = {
        "cmd": cmd,
        "cycles": cycles,
        "request_id": request_id,
        "timestamp": int(time.time() * 1000)  # 当前时间戳
    }
    client.publish(TOPIC_COMMAND, json.dumps(payload), qos=1)
    print(f"Sent command: {payload}")

# success	integer	拍照是否成功（1=成功，0=失败）
def send_camera_response(success=1, error_message=''):
    payload = {
        "success": success,
        "error_message": "",
        "timestamp": int(time.time() * 1000)
    }
    client.publish(TOPIC_CAMERA_RESPONSE, json.dumps(payload), qos=1)
    print(f"Sent camera response: {payload}")

# 定义处理接收到的消息的回调函数
def on_message(client, userdata, msg):
    print('------------------')
    print(f"[接收订阅主题: {msg.topic}: {msg.payload.decode()}]")


    # 处理状态消息
    if msg.topic == TOPIC_STATUS:
        status_data = json.loads(msg.payload.decode())
        print("状态:", status_data)

    # 处理错误消息
    elif msg.topic == TOPIC_ERROR:
        error_data = json.loads(msg.payload.decode())
        print("错误:", error_data)

    elif msg.topic == TOPIC_CAMERA_COMMAND:
        camera_command_data = json.loads(msg.payload.decode())
        print("机械臂请求拍照: ", camera_command_data)

        # 通知拍照完成
        send_camera_response(success=1)


# 初始化 MQTT 客户端
client = mqtt.Client(CLIENT_ID)

# 设置回调函数
client.on_message = on_message

# 连接到 MQTT broker
client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEP_ALIVE_INTERVAL)

# 订阅相关主题
client.subscribe(TOPIC_STATUS, qos=1)
client.subscribe(TOPIC_ERROR, qos=1)
client.subscribe(TOPIC_CAMERA_COMMAND, qos=1)

print("--已订阅主题--")

# 发送 START 命令，假设 request_id 为 "req-2024-001"
send_command("start", "req-2026-001", cycles=1)

# 开始监听 MQTT 消息
client.loop_start()

# 在这里保持长时间运行，等待响应
time.sleep(70)  # 可以根据实际需求调整

# 停止监听
client.loop_stop()