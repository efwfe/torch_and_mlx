#%%
from  huggingface_hub import notebook_login
notebook_login()
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import mediapy as media
import math
import itertools

def plt_show_image(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# %%
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16", torch_dtype=torch.float16
).to("cuda")


# %%
image_reservoir = []
latents_reservoir = []

@torch.no_grad()
def plot_show_callback(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())
    # 0.18215 ensures that the initial latent space on which the diffusion model is operating has approximately unit variance
    # https://github.com/huggingface/diffusers/issues/437#issuecomment-1241827515
    image = pipe.vae.decode(1 / 0.18215 * latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    plt.imsave(f"diffprocess/sample_{i:02d}.png", image)
    image_reservoir.append(image)

@torch.no_grad()
def save_latents(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())
    

# %%
prompt = "a handsome cat dressed like Lincoln, trending art."
with torch.no_grad():
    image = pipe(prompt, callback=plot_show_callback).images[0]

image.save("lovely_cat.png")

# %%
media.show_video(image_reservoir, fps=5)

# %% 4 channel in the latent tensor, we choose to visualize any 3 of them as RGB
Chan2RGB = [0, 1 ,2]
latents_np_seq = [
    tsr[0, Chan2RGB].permute(1, 2, 0).numpy()
    for tsr in latents_reservoir
]

# %%
media.show_video(latents_np_seq, fps=5)


# %% a text2image sampling function

@torch.no_grad()
def generate_simplified(
    prompt,
    negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
):
    batch_size = 1
    height, width = 512, 512
    generator = None
    
    # gen text embeddings
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors='pt'
    )
    text_input_ids = text_inputs.input_ids
    text_embeddings = pipe.text_encoder(
        text_input_ids.to(pipe.device)
    )[0]
    bs_embed, seq_len, _ = text_embeddings.shape
    
    # gen negative prompts
    max_length = text_input_ids.shape[-1]
    uncond_input = pipe.tokenizer(
        negative_prompt,
        padding="max_length",
        max_length= max_length,
        truncation=True,
        return_tensors='pt'
    )
    uncond_embeddings = pipe.text_encoder(
        uncond_input.input_ids.to(pipe.device)
    )[0]
    
    # duplicate unconditional embeddings 
    seq_len = uncond_embeddings.shape[1]
    uncond_embeddings = uncond_embeddings.repeat(batch_size, 1, 1)
    uncond_embeddings = uncond_embeddings.view(batch_size, seq_len, -1)
    
    # for classifier gree guidance , we need to do two forward passes.
    # we cancatenate the unconditional and text embeddings a single batch
    # to aviod doing two forward passes
    text_embddings = torch.cat([
        uncond_embeddings,
        text_embddings
    ])
    
    # 
    latents_shape = (
        batch_size, 
        pipe.unet.in_channels,
        height // 8,
        width //8
    )
    latents_dtype = text_embddings.dtype
    latents = torch.randn(latents_shape, generator=generator, device=pipe.device, dtype=latents_dtype)
    
    # set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps)
    # some schedulers like PNDM have timesteps as arrays
    # it's more optimized to move all timesteps to correct device beforehand
    timesteps_tensor = pipe.scheduler.timesteps.to(pipe.device)
    # scale the initial noise by the standard deviation required by she scheduler
    latents = latents * pipe.scheduler.init_noise_sigma
    
    # Main diffusion process
    for i, t in enumerate(pipe.progress_bar(timesteps_tensor)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embddings).sample
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t - 1
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead
    # and is compatible with bfloat 16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image

# %%
image = generate_simplified(
    prompt=["a lovely cat"],
    negative_prompt=["sunshine"]
)

# %% Image to Image
from diffusers import StableDiffusionImg2ImgPipeline


device = "cuda"
model_path = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    revision="fp16",  torch_dtype=torch.float16,
    use_auth_token=True
)
pipe = pipe.to(device)

# %%
import requests
from io import BytesIO
from PIL import Image

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize(768, 512)
# %%
prompt = "A fantasy landscape, trending on artstation"
generator = torch.Generator(device=device).manual_seed(1024)
with autocast("cuda"):
    image = pipe(prompt=prompt, 
                 init_image=init_image,
                 strength=0.75,
                 guidance_scale=7.5,
                 generator=generator).images[0]

image

# %% simple img2img sampling function

@torch.no_grad()
def generate_img2img_simplified(
    num_inference_steps=30,
    guidance_scale=0.7,
    do_classifer_free_guidance=True,
    **extra_step_kwargs
):
    prompt = ["A fantasy landscape, trending on artstation"]
    negative_prompt = [""]
    strength = 0.5 # strength of the image conditioning
    batch_size = 1
    
    
    # 
    pipe.scheduler.set_timesteps(num_inference_steps)
    # gen text prompts
    text_inputs = pipe.tokenizer(
        prompt,
        padding='max_length',
        max_length=pipe.tokenizer.model_max_length,
        return_tensors='pt',
    )
    
    text_input_ids = text_inputs.input_ids
    text_embeddings = pipe.text_encoder(text_input_ids.to(pipe.device))[0]
    
    # uncond embeddings for classifier free guidance
    uncondi_tokens = negative_prompt
    max_length = text_input_ids.shape[-1]
    uncond_input = pipe.tokenizer(
        uncondi_tokens,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]
    
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    # encode the init image into latents and scale the latents
    latents_dtype = text_embeddings.dtype
    if isinstance(init_image, Image.Image):
        init_image = preprocess(init_image)
    
    init_image = init_image.to(device=pipe.device, dtype=latents_dtype)
    init_latent_dist = pipe.vae.encode(init_image).latent_dist
    init_latents = init_latent_dist.sampe(generator=generator)
    init_latents = 0.18215 * init_latents # todo
    
    # get the original timestep using init_timestep
    offset = pipe.scheduler.config.get("steps_offset", 0)
    init_timestep = int(init_timestep * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)
    
    timesteps = pipe.scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps] * batch_size, device=pipe.device)
    
    # add noise to latents using the timesteps
    noise = torch.randn(init_latents.shape, generator=generator, device=pipe.device, dtype=latents_dtype)
    init_lantents = pipe.scheduler.add_noise(init_latents, noise, timesteps)
    lantents = init_lantents
    
    t_start = max(num_inference_steps - init_timestep + offset, 0)
    timesteps = pipe.scheduler.timesteps[t_start:].to(pipe.device)
    
    for i, t in enumerate(pipe.process_bar(timesteps)):
        # expand the lantetns if web doing clasifier free guidance
        latent_model_input = torch.cat([latents]*2) if do_classifer_free_guidance else lantents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # predict the nodise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previouse noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, lantents, **extra_step_kwargs).prev_sample
        
    latents = 1 / 0.18215 * lantents
    image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return image