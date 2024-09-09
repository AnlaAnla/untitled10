import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def data_generator():
    while True:
        x = torch.randint(1, 10, (1,))
        x = torch.tensor(x, dtype=torch.float)
        if x % 2 == 0:
            y = torch.tensor([0., 1])
        else:
            y = torch.tensor([1., 0.])
        yield x, y


# for i in range(10):
#     print(next(data_gen))

def train_model(model, epoch, criterion, optimizer):
    model.train()
    step = 0
    for x, y in data_gen:
        step += 1
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))
            print('Batch number: {}'.format(step))
        if step == epoch:
            return model


def get_accuracy(model, epoch):
    model.eval()
    correct = 0
    step = 0
    with torch.no_grad():
        for x, y in data_gen:
            step += 1

            output = model(x)
            predict = torch.argmax(output, dim=0)
            if predict == torch.argmax(y, dim=0):
                correct += 1
                print(f'{correct}/{step}')

            if step == epoch:
                break

    print(f"test result: {correct}/{step}, accuracy: {correct / step}")


if __name__ == '__main__':
    model = Net()
    data_gen = data_generator()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    model = train_model(model, 8000, criterion, optimizer)
    get_accuracy(model, 15000)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def data_generator():
    while True:
        x = torch.randint(1, 10, (1,))
        x = torch.tensor(x, dtype=torch.float)
        if x % 2 == 0:
            y = torch.tensor([0., 1])
        else:
            y = torch.tensor([1., 0.])
        yield x, y


# for i in range(10):
#     print(next(data_gen))

def train_model(model, epoch, criterion, optimizer):
    model.train()
    step = 0
    for x, y in data_gen:
        step += 1
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))
            print('Batch number: {}'.format(step))
        if step == epoch:
            return model


def get_accuracy(model, epoch):
    model.eval()
    correct = 0
    step = 0
    with torch.no_grad():
        for x, y in data_gen:
            step += 1

            output = model(x)
            predict = torch.argmax(output, dim=0)
            if predict == torch.argmax(y, dim=0):
                correct += 1
                print(f'{correct}/{step}')

            if step == epoch:
                break

    print(f"test result: {correct}/{step}, accuracy: {correct / step}")


if __name__ == '__main__':
    model = Net()
    data_gen = data_generator()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    model = train_model(model, 8000, criterion, optimizer)
    get_accuracy(model, 15000)
