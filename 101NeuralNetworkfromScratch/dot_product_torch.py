
# %% package
import numpy as np

# %%
X = [0, 1]
w1 = [2, 3]
w2 = [0.4, 1.8]

# %%
# wchi weight is more similar to X

dot_x_w1 = X[0] * w1[0] + X[1] * w1[1]
dot_x_w2 = X[0] * w2[0] + X[1] * w2[1]

dot_x_w1, dot_x_w2
# %%
np.dot(X, w1), np.dot(X, w2)
# %%
