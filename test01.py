import threading
import time

data = [i for i in range(10)]


def task(data, start, end, name, delay=None):
    for i in range(start, end):
        if delay is not None and i > 2:
            time.sleep(delay)
        print(f"{name}: {data[i]}")


t1 = threading.Thread(target=task, args=(data, 0, 5, 't1', 0.1))
t2 = threading.Thread(target=task, args=(data, 5, len(data), 't2'))

t_list = [t1, t2]

for t in t_list:
    t.start()

for t in t_list:
    t.join()

print('结束')
