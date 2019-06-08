# -*- coding: utf-8 -*
import argparse
import torch
import torch.nn as nn
from flyai.dataset import Dataset
from torch.optim import Adam

from model import Model
from net import Net
from path import MODEL_PATH

# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=100, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
args = parser.parse_args()

clip = 5

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)


def eval(model, x_test, y_test):
    network.eval()
    total_acc = 0.0
    data_len = len(x_test[0])
    x1, x2 = x_test
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    x1 = x1.float().to(device)
    x2 = x2.float().to(device)
    y_test = torch.from_numpy(y_test)
    y_test = y_test.to(device)
    batch_eval = model.batch_iter(x1, x2, y_test)

    for x_batch1, x_batch2, y_batch in batch_eval:
        outputs = network(x_batch1, x_batch2)
        _, prediction = torch.max(outputs.data, 1)
        correct = (prediction == y_batch).sum().item()
        total_acc += correct
    return total_acc / data_len


# 数据获取辅助类
data = Dataset()
network = Net().to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = Adam(network.parameters(), lr=0.001, weight_decay=1e-4)  # 定义优化器，选用AdamOptimizer

model = Model(data)
iteration = 0

best_accuracy = 0
# 得到训练和测试的数据
for epoch in range(args.EPOCHS):
    network.train()

    # 得到训练和测试的数据
    x_train, y_train, x_test, y_test = data.next_batch(args.BATCH)  # 读取数据

    batch_len = y_train.shape[0]
    x1, x2 = x_train
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    x1 = x1.float().to(device)
    x2 = x2.float().to(device)
    y_train = torch.from_numpy(y_train)
    y_train = y_train.to(device)

    outputs = network(x1, x2)
    _, prediction = torch.max(outputs.data, 1)

    optimizer.zero_grad()
    outputs = outputs.float()
    # calculate the loss according to labels
    loss = loss_fn(outputs, y_train)

    # backward transmit loss
    loss.backward()

    # adjust parameters using Adam
    optimizer.step()

    # 若测试准确率高于当前最高准确率，则保存模型
    train_accuracy = eval(model, x_test, y_test)
    print(train_accuracy)
    if train_accuracy > best_accuracy:
        best_accuracy = train_accuracy
        model.save_model(network, MODEL_PATH, overwrite=True)
        print("Best accuracy: epoch: {}, best accuracy: {}".format(epoch, best_accuracy))
    print("Training data: epoch: {} in EPOCHS {}, best accuracy: {}".format(epoch, args.EPOCHS, best_accuracy))
