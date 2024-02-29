
# %% load package

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from tqdm.notebook import trange, tqdm
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR
from einops import rearrange
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

# %%

def forward_diffusion_1D(x0, noise_strength_fn, t0, nsteps, dt):
    """
    Simulate forward diffusion for N steps.
    
    
    x0: initial sample value, scalar
    noise_strength_fn: function of time, outputs scalar noise strength
    t0: initial time
    nsteps: number of diffusion steps
    dt: time step size
    """
    
    # initialize trajectory
    x = np.zeros(nsteps + 1)
    x[0] = x0
    t = t0 + np.arange(nsteps + 1) * dt
    
    # perform many elua-maruyama time steps
    for i in range(nsteps):
        noise_strength = noise_strength_fn(t[i])
        random_normal = np.random.randn()
        x[i+1] = x[i] + np.sqrt(dt) * noise_strength * random_normal
    return x, t

def noise_strength_constant(t):
    return 1


# %% 
nsteps = 100
t0 = 0
dt = 0.1
noise_strength_fn = noise_strength_constant
x0 = 0
num_tries = 5

for i in range(num_tries):
    x, t = forward_diffusion_1D(x0, noise_strength_fn, t0, nsteps, dt)
    plt.plot(t, x)
    plt.xlabel("time", fontsize=20)
    plt.ylabel("x", fontsize=20)

plt.title("Forward diffusion visualized", fontsize=20)
plt.show()
# %%
# Basic reverse diffusion

# Simulate forward diffusion for N steps.

def reverse_diffusion_1D(x0, noise_strength_fn, score_fn, T, nsteps, dt):
    """
    x0: init sample value
    noise_strength_fn: function of time, outputs scalar noise strength
    score_fn: score function
    T: final time
    nsteps: number of diffusion steps
    dt: time step size
    """
    
    # Initialize trajectory
    x = np.zeros(nsteps + 1)
    x[0] = x0
    t = np.arange(nsteps + 1) * dt
    
    # Perform many Elur-a time steps
    for i in range(nsteps):
        noise_strength = noise_strength_fn(T - t[i])
        score = score_fn(x[i], 0, noise_strength, T-t[i])
        random_normal = np.random.randn()
        x[i+1] = x[i] + (noise_strength ** 2) * score * dt + np.sqrt(dt)*noise_strength*random_normal
    return x, t

def score_simple(x, x0, noise_strength, t):
    score = -(x - x0) / ((noise_strength ** 2)*  t)
    return score
# %%
nsteps = 100
t0 = 0
dt = 0.1
noise_strength_fn = noise_strength_constant
score_fn = score_simple
x0 = 0
T = 11

num_tries = 5

for i in range(num_tries):
    x0 = np.random.normal(loc=0, scale=T)
    x,t = reverse_diffusion_1D(x0, noise_strength_fn, score_fn, T, nsteps, dt)
    
    plt.plot(t, x)
    plt.xlabel("time", fontsize=20)
    plt.ylabel("x", fontsize=20)

plt.title("Reverse diffusion visualized", fontsize=20)
plt.show()
# %%
# 2. Working with images via U-Nets

# helping our neural network work with time

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps"""
    def __init__(self, embed_dim, scale=30):
        super().__init__()
        # Randomly sample weights during initialization
        # These weights (frequencies) are fixed during optimization and are not trainable
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=True)
        
    def forward(self, x):
        # Cosine(2 pi freq x) , Sine(2 pi freq x)
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([
            torch.sin(x_proj),
            torch.cos(x_proj)
        ], dim=-1)

class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    Allow time repr to input additively from the side of a convolution layer.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.dense(x)[..., None, None]
    

# %%
# Defining the U-Net architecture

class Unet(nn.Module):
    def __init__(self, marginal_prob_std, channels=[32, 64,128,256], embed_dim=256):
        """
        Initialize a time-dependent score-based network.
        
        Args:
            marginal_prob_std: A function that takes time t and gives the standard deviation
            of the perturbation kernel p_{0t}(x(t) | x(0)).
            channels: The number of channels for feature maps of each resolution.
            embed_dim: The dimensionality of Gaussian random feature embeddings.
            
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2= nn.GroupNorm(32, num_channels=channels[1])
        
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        
        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3 , stride=2, bias=False)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
        
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        
    
    def forward(self, x, t, y=None):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.time_embed(t))
        # Encoding path
        h1 = self.conv1(x) + self.dense1(embed)
        # Incorporate information from t
        # Group normalization
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        
        h3 = self.act(self.gnorm2(h2))
        h3 = self.conv3(h3) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm2(h4))
        
        h = self.tconv4(h4)
        
        # skip connction from the encoding path
        h += self.dense5(embed)
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        
        h = self.act(self.tgnorm3(h))
        h += self.dense7(embed)
        
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(torch.cat([h, h1]), dim=1)
        
        # normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h

# %%
# Train the U-Net to learn a score function

device = 'cuda'

def marginal_prob_std(t, sigma):
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    return torch.tensor(sigma ** t, device=device)


sigma = 25.0 
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigm=sigma)
# %%
# Defining the loss function

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """
    The loss function for training score-based generative models.
    Args:
        model: torch model instance
        x: a mini-batch of tarining data
        marginal_prob_std: a function gives the standard deviation of the perturbation kernel
        eps: A tolerance value for numerical stability.
    """
    # Sample time uniformly in 0,1
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    # Find the noise std at the time
    std = marginal_prob_std(random_t)
    z = torch.randn_like(x)
    perturbed_x = x + std[:, None, None, None] * z
    
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum(
        (score * std[:, None, None, None] + z) ** 2,
        dim=(1,2,3)
    ))
    return loss
# %%
# Defining the sampler
num_steps = 500

def Eular_Maruyama_sampler(
    score_model,
    marginal_prob_std,
    diffusion_coeff,
    batch_size=64,
    x_shape=(1, 28, 28),
    num_steps=num_steps,
    device='cuda',
    eps=1e-3,
    y=None
):
    """
    Generate samples from score-based models with the Euler-Maruyama solver.
    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        diffusion_coeff: A function give the diffusion coefficient of the SDE
        num_steps: the number of samplers to generate by calling this function once. 
        eps: The samllest time step for numerical stability
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, *x_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(
                batch_size,
                device=device
            ) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g ** 2 )[:, None, None, None] * score_model(x, batch_time_step, y=y) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
    return mean_x

# %%
# Training on MNIST
# We will train on MNIST
score_model = torch.nn.DataParallel(Unet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

n_epochs = 50
batch_size = 2048
lr = 5e-4

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
optimizer = Adam(score_model.parameters(), lr=lr)
tqdm_epoch = trange(n_epochs)

for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    for x, y in tqdm(data_loader):
        x = x.to(device)
        loss = loss_fn(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
        
    tqdm_epoch.set_description("Average Loss: {:5f}".format(avg_loss / num_items))
    torch.save(score_model.state_dict(), "ckpt.pth")
# %%

# ATTENTION Modules


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=1):
        """
        just 1 head attention
        """
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.query = nn.Liear(hidden_dim, embed_dim, bias=False)
        
        if context_dim is None:
            self.self_attn = True
            self.key = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.self_attn = False
            self.key = nn.Linear(context_dim, embed_dim, bias=False)
            self.value = nn.Linear(context_dim, hidden_dim, bias=False)
        
    
    def forward(self, tokens, context=None):
        # tokens: with shape [batch, sequence_len, hidden_dim]
        # context: with shape [ batch, context_seq_len, context_dim]
        if self.self_attn:
            Q = self.query(tokens)
            K = self.key(tokens)
            V = self.value(tokens)
        else:
            # impl Q, K , V for the Cross attention
            Q = self.query(tokens)
            K = self.key(context)
            V= self.value(context)
            
        scoremats = torch.einsum("BTH,BSH->BTS", Q, K)
        attnmats = F.softmax(scoremats / np.sqrt(self.embed_dim), dim=-1)
        ctx_vecs = torch.einsum("BTS,BSH->BTH", attnmats, V)
        return ctx_vecs


class TransformerBlock(nn.Module):
    """
    The transformer block that combines self-attn, cross-attn and feed forward neural net
    """
    def __init__(self, hidden_dim, context_dim):
        super().__init__()
        self.attn_self = CrossAttention(hidden_dim, hidden_dim)
        self.attn_cross = CrossAttention(hidden_dim, hidden_dim, context_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # imp a 2 layer MLP with k * hidden_dim hidden units, and nn.GeLU nonliearity
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 3 * hidden_dim),
            nn.GELU(),
            nn.Linear(3 * hidden_dim, hidden_dim)
        )

    
    def forward(self, x, context=None):
        # Notice the + x as residue connections
        x = self.attn_self(self.norm1(x)) + x
        x = self.attn_cross(self.norm2(x), context=context) + x
        x = self.ffn(self.norm3(x)) + x
        return x
    

class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super().__init__()
        self.transformer = TransformerBlock(hidden_dim, context_dim)
        
    
    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x + x_in
    

# %%
# UNet transformer
class UNet_Transfomer(nn.Module):
    
    def __init__(
        self,
        marginal_prob_std,
        channels=[32, 64, 128, 256],
        embed_dim=256,
        text_dim=256,
        nClass=10
    ):
        
        """
        Initialize a time-dependent score-based network.
        """
        
        super().__init__()
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.attn3 = SpatialTransformer(channels[2], text_dim)
        
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        self.attn4 = SpatialTransformer(channels[3], text_dim)
        
        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2],3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        
        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])        
        
        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1)
        
        self.act = nn.SiLU()
        self.marginal_prob_std = marginal_prob_std
        self.cond_embed = nn.Embedding(nClass, text_dim)
        
    
    def forward(self, x, t, y=None):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.time_embed(t))
        y_embed = self.cond_embed(y).unsqueeze(1)
        
        # Encoding path
        h1 = self.conv1(x) + self.dense1(x)
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        h3 = self.attn3(h3, y_embed) # Use your attention layers
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))
        h4 = self.attn4(h4, y_embed)
        
        # Decode path
        h = self.tconv4(h4) + self.dense5(embed)
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(h + h3) + self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        h = self.tconv2(h + h2) + self.dense7(embed)
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1)
        
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
# %%
# Conditional Denoising Loss

def loss_fn_cond(model, x, y, marginal_prb_std, eps=1e-5):
    """
    loss for training score-based generative models
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. -eps) + eps
    z = torch.randn_like(x)
    std = marginal_prb_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t, y=y)
    loss = torch.mean(torch.sum((score * std[:, None, None,None]), dim=(1,2,3)))
    return loss
# %%
# Training model of attention

continue_training = False
if not continue_training:
    print("initilize new score  model ....")
    score_model = torch.nn.DataParallel(
        UNet_Transfomer(
            marginal_prob_std=marginal_prob_std_fn
        )
    )
    score_model = score_model.to(device)
    
n_epochs = 100
batch_size = 1024
lr= 10e-4

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

optimizer = Adam(score_model.parameters(), lr=lr)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))
tqdm_epoch = trange(n_epochs)

for epoch in tqdm_epoch:
    avg_loss = 0.
    num_iterms = 0
    for x,y in tqdm(data_loader):
        x = x.to(device)
        loss = loss_fn_cond(score_model, x, y, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    scheduler.step()
    lr_current = scheduler.get_last_lr()[0]
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    tqdm_epoch.set_description("average loss: {:5f}".format(avg_loss / num_items))
    torch.save(score_model.state_dict(), 'ckpt_transformer.pth') 
     
# %%
# Latent space diffusion using an autoencoder

class AutoEncoder(nn.Module):
    def __init__(self, channels=[4, 8, 32]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, stride=1, bias=True),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(),
            nn.Conv2d(channels[1], channels[2],3, stride=1, bias=True),
            nn.BatchNorm2d(channels[2])
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[1], 3, stride=1, bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(),
            nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=True, output_padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
            nn.ConvTranspose2d(channels[0], 1, 3, stride=1, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

# %%
# Train the autoencoder with the help of perceptual loss

x_tmp = torch.randn(1, 1, 28, 28)
assert AutoEncoder()(x_tmp).shape == x_tmp.shape

# %% train the autoencoder with preceptual loss

from lpips import LPIPS

# %%
lpips = LPIPS(net='squeeze').cuda()
loss_fn_ae = lambda x, xhat: nn.functional.mse_loss(x, xhat) + lpips(x.repeat(1, 3, 1,1), x_hat.repeat(1, 3, 1, 1)).mean()

# %% 
ae_model = AutoEncoder([4, 4, 4]).cuda()
n_epochs = 50
batch_size = 2048
lr = 10e-4

dataset = MNIST(".", train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

optimizer = Adam(ae_model.parameters(), lr=lr)
tqdm_epoch = trange(n_epochs)

for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    for x, y in data_loader:
        x = x.to(device)
        z = ae_model.encoder(x)
        x_hat = ae_model.decoder(x)
        loss = loss_fn_ae(x, x_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    print('{} Average Loss: {:5f}'.format(epoch, avg_loss / num_items))
    # Print the averaged training loss so far.
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Update the checkpoint after each epoch of training.
    torch.save(ae_model.state_dict(), 'ckpt_ae.pth')


# %% create latent state dataset
batch_size = 4096
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
ae_model.requires_grad_(False)
ae_model.eval()

zs = []
ys = []

for x,y in tqdm(data_loader):
    z = ae_model.encoder(x.to(device)).cpu()
    zs.append(z)
    ys.append(y)

zdata = torch.cat(zs)
ydata = torch.cat(ys)

# %%
from torch.utils.data import TensorDataset

latent_dataset = TensorDataset(zdata, ydata)

# %%