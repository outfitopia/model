import IPython.display 
import gc
import torch
from PIL import Image
import IPython.display 
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler , PNDMScheduler
import warnings

warnings.filterwarnings('ignore')

class config : 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    HEIGHT = 512                        
    WIDTH = 512                         
    NUM_INFERENCE_STEPS = 500            
    GUIDANCE_SCALE = 15                
    GENERATOR = torch.manual_seed(50)   
    BATCH_SIZE = 1
    

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size = (cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box = (i % cols * w, i // cols * h))
    return grid


vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder = "vae")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder = "unet")
vae = vae.to(config.DEVICE)
text_encoder = text_encoder.to(config.DEVICE)
unet = unet.to(config.DEVICE) 


scheduler = LMSDiscreteScheduler(beta_start = 0.00085, beta_end = 0.012, beta_schedule = "scaled_linear", num_train_timesteps = 1000)
print(f'\033[94mThe scheduler loaded is K-LMS Sceheduler')

prompt = ["large size blue suit outfit"]

text_input = tokenizer(prompt, padding = "max_length", max_length = tokenizer.model_max_length, truncation = True, return_tensors = "pt")
max_length = text_input.input_ids.shape[-1]

with torch.no_grad():
      text_embeddings = text_encoder(text_input.input_ids.to(config.DEVICE))[0]
uncond_input = tokenizer(
    [""] * config.BATCH_SIZE, padding = "max_length", max_length = max_length, return_tensors = "pt"
)

with torch.no_grad():
      uncond_embeddings = text_encoder(uncond_input.input_ids.to(config.DEVICE))[0]   
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
# print(f'\033[94mText Embeddings shape: {text_embeddings.shape}')

latents = torch.randn(
  (config.BATCH_SIZE, unet.in_channels, config.HEIGHT // 8, config.WIDTH // 8),
  generator=config.GENERATOR,
)

latents = latents.to(config.DEVICE)

# print(f'\033[94mLatent shape: {latents.shape}')

scheduler.set_timesteps(config.NUM_INFERENCE_STEPS)
latents = latents * scheduler.sigmas[0]

with autocast(config.DEVICE):
      for i, t in tqdm(enumerate(scheduler.timesteps)):
        
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        with torch.no_grad():
              noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, i, latents).prev_sample

latents = 1 / 0.18215 * latents

with torch.no_grad():
  image = vae.decode(latents).sample
print(f'\033[94mImage shape: {image.shape}')

image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0]


prompt = ["extra large size blue kurta with red buttons"]

text_input = tokenizer(prompt, padding = "max_length", max_length = tokenizer.model_max_length, truncation = True, return_tensors = "pt")

max_length = text_input.input_ids.shape[-1]
with torch.no_grad():
      text_embeddings = text_encoder(text_input.input_ids.to(config.DEVICE))[0]
uncond_input = tokenizer(
    [""] * config.BATCH_SIZE, padding = "max_length", max_length = max_length, return_tensors = "pt"
)

with torch.no_grad():
      uncond_embeddings = text_encoder(uncond_input.input_ids.to(config.DEVICE))[0]   
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

latents = torch.randn(
  (config.BATCH_SIZE, unet.in_channels, config.HEIGHT // 8, config.WIDTH // 8),
  generator=config.GENERATOR,
)
latents = latents.to(config.DEVICE)

print(f'\033[94mLatent shape: {latents.shape}')

scheduler.set_timesteps(config.NUM_INFERENCE_STEPS)
latents = latents * scheduler.sigmas[0]

with autocast(config.DEVICE):
      for i, t in tqdm(enumerate(scheduler.timesteps)):
        
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)

        with torch.no_grad():
              noise_pred = unet(latent_model_input, t, encoder_hidden_states = text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, i, latents).prev_sample

latents = 1 / 0.18215 * latents

with torch.no_grad():
  image = vae.decode(latents).sample
print(f'\033[94mImage shape: {image.shape}')

image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0].save("img2.jpg")
pil_images[0]

# to avoid memory leaks
del latents
del vae
del text_encoder
del unet
gc.collect()


def getImages(prompt) :
    prompt_array =[prompt] * 4

    with autocast("cuda"):
        images = pipe(prompt_array , num_inference_steps = 50).images

    return image_grid(images, rows = 2, cols = 2)