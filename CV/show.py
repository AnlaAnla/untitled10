import torch
import time
import random
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs')

for i in range(1000):
    time.sleep(0.5)
    data = i ** 2 * random.random()
    loss = data - i
    print(data)
    writer.add_scalar('data', data, i)
    writer.add_scalar('data', loss, i)

writer.flush()
writer.close()

# tensorboard --logdir=runs
