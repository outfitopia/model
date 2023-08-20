pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision = "fp16", torch_dtype = torch.float16, use_auth_token = Hugging_face)  
pipe = pipe.to(config.DEVICE)