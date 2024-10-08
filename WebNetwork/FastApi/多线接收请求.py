import os

from fastapi import FastAPI, BackgroundTasks
import asyncio

app = FastAPI()

# 用于存储正在执行的长时间任务
long_running_tasks = set()


async def train_model(x, y):
    print('train', x, y)

    y = 33
    for i in range(50000000):
        y = y ** i
        if i % 1000000 == 0:
            print('train:', i)
            await asyncio.sleep(0.001)  # 模拟耗时操作
    print('结束:', y)
    long_running_tasks.remove(1)


async def run_tensorboard(cmd):
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    print(f'TensorBoard 进程 ID: {proc.pid}')


@app.post("/post01/{data}")
async def post01(data: str, background_tasks: BackgroundTasks = BackgroundTasks()):
    if 1 in long_running_tasks:
        return {'已经存在训练任务, 可以通过log查看训练过程'}
    else:
        args = {'x': 'aaa', 'y': 'baba'}
        long_running_tasks.add(1)
        background_tasks.add_task(train_model, x=args['x'], y=args['y'])

        return {"message": f"添加任务 {1}"}


@app.post("/post02/{data}")
async def post02(data: str):
    return {"post02": f"接收到 {data}"}


@app.on_event("startup")
async def startup():
    tensorboard_cmd = f"tensorboard --logdir=runs"
    await (run_tensorboard(tensorboard_cmd))
    print('初始化!!!')


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
