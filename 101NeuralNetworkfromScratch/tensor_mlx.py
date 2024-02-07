# %%
import mlx.core as mx
import numpy as np
import seaborn as sns


# %%
x = mx.array(5.0)
# %%
y  = x + 10
print(y)
# %%
mx.grad(lambda: x)
# %%
def func(x):
    return x+ 10

grad_fn = mx.grad(func)
# %%
x = mx.array(1.0)
grad_fn(x)

# %%

def y_function(val):
    return (val - 3) * (val - 6) * (val - 4)

x_range = np.linspace(0, 10, 101)
y_range = y_function(x_range)

# %%
sns.lineplot(x=x_range, y=y_range)
# %%
grad_fn = mx.grad(y_function)
grad_fn(mx.array(1.0))
# %%
x11 = mx.array(2.0)
x21 = mx.array(3.0)

def func(x11, x21):
    x12 = 5 * x11 - 3 * x21
    x22 = 2 * x11 ** 2 + 2 * x21
    return 4 * x12 + 3 * x22

grad_fn = mx.grad(func, argnums=[0, 1])
# %%
x11_grad, x21_grad = grad_fn(x11, x21)
# %%
print(x11_grad)
print(x21_grad)
# %%
