from multiprocessing import Process, Pipe


def listener(receive_pipe):
    while True:
        data = receive_pipe.recv()  # 阻塞等待数据
        if data == "END":  # 检查是否结束信号
            print("Listener ending.")
            break
        print(f"Listener received: {data}")


if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=listener, args=(child_conn,))
    p.start()
    p.join()
