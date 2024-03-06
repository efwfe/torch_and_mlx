
#%%
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

#%%
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 4

trainset = torchvision.datasets.ImageFolder(root="train", transform=transform)
testset  = torchvision.datasets.ImageFolder(root="test", transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

# %%
CLASSES = ['affenpinscher', 'akita', 'corgi']
NUM_CLASSES = len(CLASSES)

class ImageMulticClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()
    
    def forward(self, x):
        x = self.conv1(x) # out: (BS, 6, 48, 48)
        x = self.relu(x)
        x = self.pool(x) # out: (BS, 6, 24, 24)
        x = self.conv2(x) # out: (BS, 16, 22, 22)
        x = self.relu(x)
        x = self.pool(x) # out: (BS, 16, 11, 11)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# %% 
model = ImageMulticClassification()

# %% 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# %% Training
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader):
        inputs, label = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')

# %% test
y_test, y_test_hat = [], []

for i, data in enumerate(testloader):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())
    
# %% 
acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(acc)

# %%
confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))