def log(f):
    def wrapper(*args, **kwargs):
        print('start')
        f()

    return wrapper


@log
def fn():
    print("我的主函数")


if __name__ == '__main__':
    fn()
