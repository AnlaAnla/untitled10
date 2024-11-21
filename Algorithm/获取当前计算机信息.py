import os
import psutil

# CPU 信息
cpu_count = psutil.cpu_count(logical=False)
cpu_logical_count = psutil.cpu_count(logical=True)
cpu_freq = psutil.cpu_freq().current
cpu_usage = psutil.cpu_percent(interval=1)

# 内存信息
mem = psutil.virtual_memory()
mem_total = mem.total / (1024 ** 3)  # 总内存 (GB)
mem_used = mem.used / (1024 ** 3)    # 已使用内存 (GB)
mem_free = mem.free / (1024 ** 3)    # 可用内存 (GB)
mem_usage = mem.percent               # 内存使用率

# 磁盘信息
disk_partitions = psutil.disk_partitions()
for partition in disk_partitions:
    if os.path.ismount(partition.mountpoint):
        try:
            disk_usage = psutil.disk_usage(partition.mountpoint)
            disk_total = disk_usage.total / (1024 ** 3)  # 总磁盘空间 (GB)
            disk_used = disk_usage.used / (1024 ** 3)    # 已使用磁盘空间 (GB)
            disk_free = disk_usage.free / (1024 ** 3)    # 可用磁盘空间 (GB)
            disk_percent = disk_usage.percent             # 磁盘使用率
            print(f"\n磁盘信息 ({partition.mountpoint}):")
            print(f"  总磁盘空间: {disk_total:.2f} GB")
            print(f"  已使用磁盘空间: {disk_used:.2f} GB")
            print(f"  可用磁盘空间: {disk_free:.2f} GB")
            print(f"  磁盘使用率: {disk_percent:.2f}%")
        except Exception as e:
            # 如果没有权限访问该分区,则跳过
            continue

# 网络信息
net_io_counters = psutil.net_io_counters()
bytes_sent = net_io_counters.bytes_sent / (1024 ** 2)  # 已发送的字节数 (MB)
bytes_recv = net_io_counters.bytes_recv / (1024 ** 2)  # 已接收的字节数 (MB)

# 打印信息
print(f"\nCPU 信息:")
print(f"  CPU 逻辑核心数: {cpu_logical_count}")
print(f"  CPU 物理核心数: {cpu_count}")
print(f"  CPU 频率: {cpu_freq:.2f} MHz")
print(f"  CPU 使用率: {cpu_usage:.2f}%")
print(f"\n内存信息:")
print(f"  总内存: {mem_total:.2f} GB")
print(f"  已使用内存: {mem_used:.2f} GB")
print(f"  可用内存: {mem_free:.2f} GB")
print(f"  内存使用率: {mem_usage:.2f}%")
print(f"\n网络信息:")
print(f"  已发送的字节数: {bytes_sent:.2f} MB")
print(f"  已接收的字节数: {bytes_recv:.2f} MB")