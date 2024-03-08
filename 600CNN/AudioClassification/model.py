
# %% load package
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as  sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

# %% transform
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 4

trainset = torchvision.datasets.ImageFolder(root="train", transform=transform)
testset  = torchvision.datasets.ImageFolder(root='test', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

#%%
CLASSES =  ['artifact', 'extrahls', 'murmur', 'normal']
NUM_CLASSES = len(CLASSES)

class AudioClassification(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100* 100, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()
        
    def forward(self, x):
        x = self.conv1(x) # (BS, 6, 100, 100)
        x = self.relu(x)
        x = self.pool(x) # (BS, 6, 50, 50)
        
        x = self.conv2(x) # (BS, 16, 50, 50)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
# %%
model = AudioClassification()

# %% 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# %% Training
losses_epoch_mean = []
NUM_EPOCHS =100

for epoch in range(NUM_EPOCHS):
    loss_epoch = []
    
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        loss_epoch.append(loss.item())
    losses_epoch_mean.append(np.mean(loss_epoch))
    print(f'Epoch: {epoch}/{NUM_EPOCHS}, Loss: {np.mean(loss_epoch):.4f}')
    
# %% PLOT CLOSSES
sns.lineplot(x=list(range(len(losses_epoch_mean))), y=losses_epoch_mean)

# %%
y_test = []
y_test_hat = []

for i, data in enumerate(testloader):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_temp_hat = model(inputs).round()
        
    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_temp_hat.numpy())
    

# %%
Counter(y_test)

# %% Accuracy
acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(f"Accuracy: {acc * 100:.2f}%")

# %%
cm = confusion_matrix(y_test,np.argmax(y_test_hat, axis=1))
sns.heatmap(cm, annot=True, xticklabels=CLASSES, yticklabels=CLASSES)
