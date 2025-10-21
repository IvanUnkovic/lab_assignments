import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

class ConvolutionalModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvolutionalModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7*32, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(self.flatten(self.layer2(self.layer1(x))))))
    
config = {
    'num_classes':10,
    'max_epochs': 8,
    'batch_size': 50,
    'weight_decay': 1e-3,
    'lr_policy': 1e-2
}
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False)

model = ConvolutionalModel(num_classes=config['num_classes'])
optimizer = torch.optim.SGD(model.parameters(), lr=config['lr_policy'], weight_decay=config['weight_decay'])
criterion = nn.CrossEntropyLoss()

losses_train=[]
iterations_train = np.arange(config['max_epochs']*len(train_loader))

print("Train:")

for epoch in range(config['max_epochs']):
    for index, (examples, labels) in enumerate(train_loader):
        optimizer.zero_grad() 
        outputs = model(examples)
        loss = criterion(outputs, labels)        
        loss.backward()
        optimizer.step()
        if (index+1)%100==0:
            print("Epoha: {}, iteracija: {}, gubitak:{}".format(epoch+1, index+1, loss.item()))
        losses_train.append(loss.item())
        

plt.plot(iterations_train, losses_train)
plt.title("Train set")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

print("-"*30)
print("Test:")

losses_test=[]
iterations_test = np.arange(config['max_epochs']*len(test_loader))

for epoch in range(config['max_epochs']):
    for index, (examples, labels) in enumerate(test_loader):
        optimizer.zero_grad() 
        outputs = model(examples) 
        loss = criterion(outputs, labels)        
        loss.backward()
        optimizer.step()
        if (index+1)%100==0:
            print("Epoha: {}, iteracija: {}, gubitak:{}".format(epoch+1, index+1, loss.item()))
        losses_test.append(loss.item())

plt.plot(iterations_test, losses_test)
plt.title("Test set")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()