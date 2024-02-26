
# %% load packages
from ast import Mult
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import numpy as np
from collections import Counter

# %% data prep
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

#%% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch)
#%%
y_test
# %% dataset
class MutipleLabelDataset(Dataset):
    def __init__(self, X, y ):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

#%% create instance of dataload
dataset = MutipleLabelDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=32)

# %% model
class MutipleLabelModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        return x

# %% model instance
input_dim = 10
hidden_dim = 32
output_dim = 3
model = MutipleLabelModel(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim
    )
#%% loss function optimizer and training loop
lr = 1e-2
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %% train
losses = []
slope, bias = [], []
number_epochs = 200

for epoch in range(number_epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        
        loss.backward()
        optimizer.step()
    
        losses.append(float(loss.detach().numpy()))
    
# %% plot loss
sns.lineplot(x=range(len(losses)), y=losses, alpha=.1)

# %% get most common class count
from collections import Counter
y_test_str = [str(i) for i in y_test.detach().numpy()]

most_common_cnt = Counter(y_test_str).most_common()[0][1]
print(f"Native classifier :{most_common_cnt/ len(y_test_str) * 100} %")
# %%
X_test_torch = torch.FloatTensor(X_test)
with torch.no_grad():
    y_test_hat = model(X_test_torch).round()

test_acc = accuracy_score(y_test, y_test_hat)
print(f"Test acc: {test_acc * 100} %")
# %%
with torch.no_grad():
    print(model(X_test_torch))
# %%
