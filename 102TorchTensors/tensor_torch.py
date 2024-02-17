# %% load package
import torch
import numpy as np
import seaborn as sns

# %%
x = torch.tensor(5.5, requires_grad=True)
# %%
y = x+ 10
print(y)

# %%
print(x.requires_grad)

# %%
x = torch.tensor(2.0, requires_grad=True)

# %%
def y_func(val):
    return (val -3 ) * (val -6) * (val - 4)

x_range = np.linspace(0, 10, 101)
y_range = y_func(x_range)

sns.lineplot(x=x_range, y=y_range)
# %%

y = y_func(x)
y.backward() # calc gradient
# %%
print(x.grad)
# %%
x = torch.tensor(1.0, requires_grad=True)
y = x**3
z = 5 * x - 4
z.backward()
# %%
print(x.grad)
# %%
x11 = torch.tensor(2.0, requires_grad=True)
x21 = torch.tensor(3.0, requires_grad=True)
x12 = 5 * x11 - 3 * x21
x22 = 2 * x11**2 + 2 * x21
y = 4 * x12 + 3 * x22

y.backward()
# %%
print(x11.grad)
print(x21.grad)
# %%
