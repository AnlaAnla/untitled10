from multiprocessing import shared_memory, Condition, Lock
from time import sleep

SHM_KEY = 'SHM_CARD_PROCESS'



class Listener:
    def __init__(self):
        self.shl = shared_memory.ShareableList(None, name=SHM_KEY)
        self.round_counter = 0

    def __del__(self):
        self.shl.shm.close()

    def listen(self):
        while True:
            print('触发')
            r, data = self.shl[0], self.shl[1]
            if r != self.round_counter:
                print(data)
                self.round_counter = r
            sleep(0.5)


