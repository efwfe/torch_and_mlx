#%% load package
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

# %% data import
iris = load_iris()
X= iris.data
y = iris.target
#%%
iris.keys()
# %% train test split
train_x, test_x, train_y, test_y = train_test_split(X, y)

# %% convert to f32
# train_x.shape, train_y.shape
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
# %%

class IrisData(Dataset):
    def __init__(self, X, y):
        
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

train_loader = DataLoader(
    dataset=IrisData(train_x, train_y),
    batch_size = 32
)

#%% define class 
class MutiClassNet(nn.Module):
    def __init__(self, num_features, num_classes, hidden_features):
        super().__init__()
        self.lin1 = nn.Linear(num_features, hidden_features)
        self.lin2 = nn.Linear(hidden_features, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = self.log_softmax(x)
        return x

#%% hypter parameters
NUM_FEATURES = train_x.shape[1]
HIDDEN =  6
NUM_CLASSES = len(set(iris.target))

# %% create model instance
model = MutiClassNet(
    num_features=NUM_FEATURES,
    num_classes=NUM_CLASSES,
    hidden_features=HIDDEN
)

# %% loss function
criterion = nn.CrossEntropyLoss()
#%% optimizer
lr = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

#%% training
NUM_EPOCHS = 100
losses = []
for epoch in range(NUM_EPOCHS):
    for x, y in train_loader:
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        
        optimizer.step()
    losses.append(
        float(loss.data.detach().numpy()))
    
# %%
sns.lineplot(x=range(len(losses)), y=losses)
# %% test the model
X_test_torch = torch.from_numpy(test_x)
with torch.no_grad():
    y_test_hat_softmax = model(X_test_torch)
    y_test_hat = torch.max(y_test_hat_softmax.data, 1)

# %% Accuracy
accuracy_score(test_y, y_test_hat.indices)
# %%
from collections import Counter
most_common_cnt = Counter(test_y).most_common()[0][1]
print(f"Naive Classifer: {most_common_cnt / len(test_y) * 100} %"
      )
# %%
torch.save(model.state_dict(), 'model_iris.pt')
# %%
