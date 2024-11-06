import torch
import torch.nn as nn
import numpy as np
import torchvision

# x = torch.tensor(1., requires_grad=True)
# w = torch.tensor(2., requires_grad=True)
# b = torch.tensor(3., requires_grad=True)
#
# y = w * x + b
#
# y.backward()
#
# print(x.grad)
# print(w.grad)
# print(b.grad)

# x = torch.randn(10, 3)  # (배치 사이즈, 인풋 사이즈)
# y = torch.randn(10, 2)  # (배치 사이즈, 아웃풋 사이즈)
#
# linear = nn.Linear(3, 2)  # Dense(3,2)
# print('w: ', linear.weight)
# print('w: ', linear.bias)
#
# criterion = nn.MSELoss()  # mean square error
# optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
#
# #forward -> loss -> backward -> optimizer
# num_epochs = 100
# for i in range(num_epochs):
#     optimizer.zero_grad()
#     pred = linear(x)
#     loss = criterion(pred,y)
#     loss.backward()
#     optimizer.step()
#     print(i, loss.item())


# x = np.array([[1, 2], [3, 4]])
# y = torch.from_numpy(x)
# z = y.numpy()
# print("a")

# train_dataset = torchvision.datasets.CIFAR10(
#     root='./data/',
#     transform=torchvision.transforms.ToTensor(),
#     download=True
# )

# class CustomDataset(torch.utils.data.DataLoader):
#     def __init__(self):
#         #데이터 초기화
#         #우리 입력 데이터 뭉치기 뭐고
#         #출력(정답지) 데이터 뭉치가 뭐다
#
#     def __getitem__(self, item):
#         #DataLoader Batch 단위로 데이터를 뽑음
#         #100개의 데이터 중에 64 (Btch Size)를 뽑아야하는데
#
#     def __len__(self):
#         #DataLoader가 접근할 수 있는 길이
#         #100
#
# image, label = train_dataset[0]
# print(image.size())
# print(label)
#
# # Data loader
# train_loader = torch.utils.data.DataLoader(
#     dataset=train_dataset,
#     batch_size=64,
#     shuffle=True
# )

resnet = torchvision.models.resnet18(pretrain=True)

for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 180, 12)

torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')
