import posix_ipc
import mmap
import os
import struct
import time


class Listener:
    def __init__(self):
        # 创建或打开信号量
        self.semaphore = posix_ipc.Semaphore("shm_card_semaphore", posix_ipc.O_CREAT, initial_value=0)

        # 创建或打开共享内存
        self.memory = posix_ipc.SharedMemory("shm_card_memory", posix_ipc.O_CREAT, size=1024)
        self.map_file = mmap.mmap(self.memory.fd, self.memory.size)
        self.memory.close_fd()

        self.wait = True

    def set_wait(self, is_wait):
        self.wait = is_wait

    def listening(self, wait=True):
        if self.wait:
            self.semaphore.acquire()

        # 读取数据
        self.map_file.seek(0)
        size = struct.unpack('i', self.map_file.read(4))[0]
        data = self.map_file.read(size).decode('utf-8')
        return data


# if __name__ == '__main__':
#     app = Listener()
#     while True:
#         data = app.listening()
#         if data == 'start':
#             app.set_wait(False)
#         if data == 'stop':
#             app.set_wait(True)
#         print(data)