# %% load package
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# %%
df = pd.read_csv("heart.csv")
df.head()
# %%
X = np.array(df.loc[:, df.columns != 'output'])
y = np.array(df['output'])

print(X.shape, y.shape)
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# %%
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale  =  scaler.transform(X_test)

# %% 
X_train_scale.shape

# %% network class

class NeuralNetworkFromScratch:
    def __init__(self, lr, x_train, y_train, x_test, y_test) -> None:
        self.w = np.random.randn(X_train_scale.shape[1])
        self.b = np.random.randn()
        self.lr = lr
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.l_train = []
        self.l_test = []

    
    def activation(self, x):
        # sigmoid
        return  1 / (1 + np.exp(-x))
    
    def dactivation(self, x):
        # derivative of sigmoid
        return self.activation(x) * (1 - self.activation(x))
    
    def forward(self, x):
        hidden_1 = np.dot(x, self.w) + self.b
        activate_1 = self.activation(hidden_1)
        return activate_1
    
    def backward(self, x, y_true):
        # calc gradients
        hidden_1 = np.dot(x, self.w) + self.b
        y_pred = self.forward(x)

        dl_dpred = 2 * (y_pred - y_true)
        dpred_dhidden1 = self.dactivation(hidden_1)
        dhidden1_db = 1
        dhidden1_dw = x

        dl_db = dl_dpred * dpred_dhidden1 * dhidden1_db
        dl_dw = dl_dpred * dpred_dhidden1 * dhidden1_dw
        return dl_db, dl_dw

    def optimizer(self, dl_db, dl_dw):
        self.b = self.b - dl_db * self.lr
        self.w = self.w - dl_dw * self.lr


    def train(self, iterations):
        
        for i in range(iterations):
            # random postion
            random_pos = np.random.randint(len(self.x_train))
            # forward pass
            y_train_true = self.y_train[random_pos]
            y_train_pred = self.forward(self.x_train[random_pos])
            # calc training losses
            l = np.sum(np.square(y_train_pred - y_train_true))
            self.l_train.append(l)

            # calc gradients
            dl_db, dl_dw = self.backward(self.x_train[random_pos], y_train_true)

            # update weights
            self.optimizer(dl_db, dl_dw)

            # calc error for test data
            l_sum = 0
            for j in range(len(self.x_test)):
                y_true = self.y_test[j]
                y_pred = self.forward(self.x_test[j])
                l_sum += np.square(y_pred - y_true)
            self.l_test.append(l_sum)
        return "training success"
# %%
# Hypter parameters
LR = 0.1
ITERRATIONS = 1000

nn = NeuralNetworkFromScratch(
    lr=LR,
    x_train=X_train_scale, 
    x_test=X_test_scale, 
    y_train=y_train, 
    y_test=y_test
)

nn.train(ITERRATIONS)

# %%
sns.lineplot(
    x = list(range(len(nn.l_test))),
    y = nn.l_test
)
# %% iter over test data
total = X_test_scale.shape[0]
correct = 0
y_preds = []

for i in range(total):
    y_true = y_test[i]
    y_pred = np.round(nn.forward(X_test_scale[i]))
    y_preds.append(y_pred)
    correct += 1 if y_true == y_pred else 0


# %%
correct / total
# %% Baseline Classifier
from collections import Counter
Counter(y_test)

# %%
# Confusion Matrix
confusion_matrix(y_true=y_test, y_pred=y_preds)
# %%
