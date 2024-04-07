from multiprocessing import shared_memory, Manager
from time import sleep

SHM_KEY = 'SHM_CARD_PROCESS'


class Listener:
    def __init__(self):
        self.shl = shared_memory.ShareableList(None, name=SHM_KEY)
        self.round_counter = 0

        manager = Manager()
        self.lock = manager.Lock()
        self.condition = manager.Condition(self.lock)

    def __del__(self):
        self.shl.shm.close()

    def listen(self):

        while True:
            with self.lock:
                print('触发')
                r, data = self.shl[0], self.shl[1]
                if r != self.round_counter:
                    print(data)
                    self.round_counter = r
                else:
                    with self.condition:
                        print('wait...')
                        self.condition.wait()

            sleep(0.5)


if __name__ == '__main__':
    app = Listener()
    app.listen()