
# %% load package
import torch
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# %%
image_path = "./image.png"
image = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

X = transforms(image).unsqueeze(0)

# %%
model = resnet18(pretrained=True)

class MyHook:
    def __init__(self) -> None:
        self.layer_out = []
        self.layer_shape = []
        
    def __call__(self, module, module_in, module_out):
        self.layer_out.append(module_out)
        self.layer_shape.append(module_out.shape)
        
# %%
my_hook = MyHook()

for l in model.modules():
    if isinstance(l, torch.nn.modules.conv.Conv2d):
        handle = l.register_forward_hook(my_hook)
        
# %%
y_pred = model(X)

# %%
layer_num = 2
layer_images = my_hook.layer_out[layer_num].detach().numpy()

for i in range(4):
    plt.imshow(layer_images[0, i, :, :])
    plt.show()