import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import torchvision.transforms as transforms
import sys
sys.path.append('../')
import adabound as myoptimizer

input_size = 784
batch_size = 100
num_epochs = 100
learning_rate = 0.001

train_datasets = dsets.MNIST(root='./data',
                             train=True,
                             download=True,
                             transform=transforms.ToTensor())
test_datasets = dsets.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=False)


class feedforward_neural_network(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super(feedforward_neural_network, self).__init__()
        self.linear = nn.Linear(input_size, hidden)
        self.r = nn.ReLU()
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.linear(x)
        x = self.r(x)
        out = self.out(x)

        return out


if torch.cuda.is_available():
    model = feedforward_neural_network(input_size=input_size, hidden=50, num_classes=10).cuda()
else:
    model = feedforward_neural_network(input_size=input_size, hidden=50, num_classes=10)

criterion = nn.CrossEntropyLoss()
optimizer = myoptimizer.Nadabound.AdaBoundW(model.parameters(), lr=1e-3, final_lr=0.1)
    # myoptimizer.Nadabound.NesterovAdaBound(model.parameters(),lr=1e-3, final_lr=0.1)
    # myoptimizer.adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)





    # torch.optim.Adam(model.parameters(), lr=learning_rate)
train_accuracy=0
test_accuracy=0

for epoch in range(num_epochs):
    print("\nepoch: " , epoch)
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.view(-1, 28 * 28)).cuda()
            labels = Variable(labels).cuda()
        else:
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    accuracy = 100. * correct / total
    current_train_accuracy=accuracy
    print('\ntrain acc %.3f' % accuracy)

    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = Variable(images.view(-1, 28 * 28)).cuda()
                labels = Variable(labels).cuda()
            else:
                images = Variable(images.view(-1, 28 * 28))
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    current_test_accuracy=accuracy
    print('\n test acc %.3f' % accuracy)
    # if current_train_accuracy>train_accuracy and current_test_accuracy>test_accuracy:
    #     optimizer = myoptimizer.Nadabound.NesterovAdaBound(model.parameters(), betas=(0.999,0.9999),lr=1e-3, final_lr=0.1)
    #     train_accuracy=current_train_accuracy
    #     test_accuracy=current_test_accuracy
    # else:
    #     optimizer = myoptimizer.Nadabound.NesterovAdaBound(model.parameters(), betas=(0.99, 0.999), lr=1e-3, final_lr=0.1)
    #     train_accuracy = current_train_accuracy
    #     test_accuracy = current_test_accuracy

    # correct = 0
    # total = 0
    # for j, (images, labels) in enumerate(test_loader):
    #     if torch.cuda.is_available():
    #         images = Variable(images.view(-1, 28 * 28)).cuda()
    #         labels = Variable(labels).cuda()
    #     else:
    #         images = Variable(images.view(-1, 28 * 28))
    #     outputs = model(images)
    #     _, predictes = torch.max(outputs.data, 1)
    #     total += labels.shape[0]
    #     correct += predicted.eq(labels).sum().item()
    # print('test acc %.3f' % (100 * correct / total))


torch.save(model.state_dict(), 'model3.pkl')