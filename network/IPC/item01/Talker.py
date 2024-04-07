from multiprocessing import shared_memory, Manager

SHM_KEY = 'SHM_CARD_PROCESS'


class Talker:
    def __init__(self):
        self.shl = shared_memory.ShareableList([0, ''], name=SHM_KEY)
        self.round_counter = 0
        manager = Manager()
        self.lock = manager.Lock()
        self.condition = manager.Condition(self.lock)

        print(self.shl.shm.name)

    def handle_input(self):
        data = input("place input data:")
        with self.lock:  # 获取锁
            self.shl[0] = self.round_counter + 1
            self.shl[1] = data
            self.round_counter += 1
        with self.condition:
            print('notify all')
            self.condition.notify_all()

    def talk(self, data=''):
        while True:
            try:
                self.handle_input()
            except KeyboardInterrupt:
                break
        self.shl.shm.close()
        self.shl.shm.unlink()


if __name__ == '__main__':
    app = Talker()
    app.talk()