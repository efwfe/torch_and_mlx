# %%
import mlx.core as mx

num_features = 100
num_examples = 1_000
num_iters = 1000
lr = 0.01

# %%
# True paramters
w_start = mx.random.normal((num_features,))
# Input examples
X = mx.random.normal((num_examples, num_features))

# Noisy labels
eps = 1e-2 * mx.random.normal((num_examples,))
y = X @ w_start + eps


# %%
def loss_fn(w):
    return 0.5 * mx.mean(mx.square(X @ w - y))


# %% use sgd to find the optimal weights. define the squared loss and get the gradient function
grad_fn = mx.grad(loss_fn)

#%%
w = 1e-2 * mx.random.normal((num_features,))

for _ in range(num_iters):
    loss = loss_fn(w)
    error_norm = mx.sum(mx.square(w-w_start),).item() ** 0.5
    print(f"Loss {loss.item():.5f}, |w-w*| = {error_norm:.5f},")

    grad = grad_fn(w)
    w = w - lr * grad
    mx.eval(w)
