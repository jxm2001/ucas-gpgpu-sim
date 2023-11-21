import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import os
# from torchsummary import summary
# from torchstat import stat
# from torch.optim.lr_scheduler import StepLR
import time

# 设置随机种子
torch.manual_seed(42)

# 定义LeNet模型
class LeNet(nn.Module):
    # def __init__(self):
    #     super(LeNet, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 4 * 4, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 4 * 4)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x
    
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(14*14, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        # self.fc2 = nn.Linear(128, 10)
        # self.relu4 = nn.ReLU()
        # self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        # x = self.fc2(x)
        return x

script_dir = os.path.dirname(__file__)  # 获取脚本所在的目录

# 数据预处理
transform = transforms.Compose([transforms.ToTensor()])

# 加载数据集
trainset = torchvision.datasets.FashionMNIST(os.path.join(script_dir, '../data'), download=True, train=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(os.path.join(script_dir, '../data'), download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=48, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

val_data_iter = iter(testloader)
val_image, val_label = val_data_iter.__next__()
print(val_image.size())


# 创建模型
model = LeNet()
model = model

# summary(model, input_size=(1, 28, 28))
# stat(LeNet().to('cpu'), (1, 28, 28))
# exit(0)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

loss_list = []

# 训练模型
for epoch in range(17):
# for epoch in range(1):
# if False:
    print('epoch ', epoch, flush=True)  

    running_loss = 0.0 #累加损失
    # for inputs, labels in trainloader:

    start_time = time.time()

    count = 0
    for step, data in enumerate(trainloader, start=0):
        inputs, labels = data
        inputs, labels = inputs, labels
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        count += 1

    end_time = time.time()
    train_time = end_time - start_time

    loss_list.append(running_loss)

    # with torch.no_grad():#上下文管理器
    #     outputs = model(val_image.to('cuda'))  # [batch, 10]
    #     predict_y = torch.max(outputs, dim=1)[1]
    #     accuracy = (predict_y == val_label.to('cuda')).sum().item() / val_label.size(0)

    #     print('[%d] train_loss: %.3f  test_accuracy: %.3f' %
    #           (epoch + 1, running_loss / count, accuracy), flush=True)  
    #     running_loss = 0.0

    start_time = time.time()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images, labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_time = time.time()
    test_time = end_time - start_time

    print('[%d] train_loss: %.3f  test_accuracy: %.3f, train_time: %.3f, test_time: %.3f' %
          (epoch + 1, running_loss / count, correct/total, train_time, test_time), flush=True)  

    # if epoch > 10 and running_loss >= loss_list[-7]:
    #     break

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images, labels
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(correct, correct/total, flush=True)  

# # 从测试集中读取一个测试输入
# dataiter = iter(testloader)
# images, labels = dataiter.__next__()
# test_input = images[0].to('cuda')
# # 打印测试输入的shape
# print("测试输入的shape:", test_input.shape)
# print(test_input)
# # 进行推理
# output = model(test_input.unsqueeze(0))
# # 打印每一层的输出
# for name, module in model.named_children():
#     # print(name)
#     if name == "fc1":
#         test_input = test_input.view(-1)
#     print("input shape:", test_input.shape, end="; ")
#     test_input = module(test_input)
#     print(f"输出层{name}的shape:", test_input.shape)
#     print(test_input)
# print("")

# 导出模型参数，也可以自定义导出模型参数的文件格式，这里使用了最简单的方法，但请注意，如果改动了必须保证程序二能够正常读取
for name, param in model.named_parameters():
    print(f"Layer name: {name}")
    print(f"Parameter shape: {param.shape}")
    np.savetxt(os.path.join(script_dir, f'./{name}.txt'), param.detach().cpu().numpy().flatten())
