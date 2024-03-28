import time

def timeit(func):
    print(11)
    func()
    print(22)

@timeit
def test01(name):
    print('test01', name)

test01('啊')