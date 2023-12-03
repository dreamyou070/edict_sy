from edict_functions import (coupled_stablediffusion, load_im_into_format_from_path,
                             prep_image_for_return, init_attention_func,init_attention_weights)
import matplotlib.pyplot as plt
import torch, random
import argparse
from transformers import CLIPModel, CLIPTokenizer
from EDICT.my_diffusers import AutoencoderKL, UNet2DConditionModel
from EDICT.my_diffusers import LMSDiscreteScheduler, PNDMScheduler, DDPMScheduler, DDIMScheduler
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

def plot_EDICT_outputs(im_tuple):
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(im_tuple[0])
    ax1.imshow(im_tuple[1])
    plt.show()
def image_to_latent(im, vae, generator, device, width=256, height=256):
    if isinstance(im, torch.Tensor):
        # assume it's the latent
        # used to avoid clipping new generation before inversion
        init_latent = im.to(device)
    else:
        # Resize and transpose for numpy b h w c -> torch b c h w
        im = im.resize((width, height), resample=Image.Resampling.LANCZOS)
        im = np.array(im).astype(np.float64) / 255.0 * 2.0 - 1.0
        # check if black and white
        if len(im.shape) < 3:
            im = np.stack([im for _ in range(3)], axis=2)  # putting at end b/c channels

        im = torch.from_numpy(im[np.newaxis, ...].transpose(0, 3, 1, 2))

        # If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
        if im.shape[1] > 3:
            im = im[:, :3] * im[:, 3:] + (1 - im[:, 3:])

        # Move image to GPU
        im = im.to(device)
        # Encode image
        init_latent = vae.encode(im).latent_dist.sample(generator=generator) * 0.18215
        return init_latent
def main(args) :

    print(f'step 1. model build')
    device = args.device
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16)
    clip = clip_model.text_model
    clip.double().to(device)
    with open('hf_auth', 'r') as f:
        auth_token = f.readlines()[0].strip()
    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    # Build our SD model
    unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet",
                                                use_auth_token=auth_token,
                                                revision="fp16", torch_dtype=torch.float16)
    unet.double().to(device)
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae",
                                        use_auth_token=auth_token,
                                        revision="fp16", torch_dtype=torch.float16)
    vae.double().to(device)

    print(f'step 2. load image')
    init_image = load_im_into_format_from_path(args.img_dir)
    prompt = 'A church'
    run_baseline = False
    latents = coupled_stablediffusion(prompt,reverse=True,init_image=init_image,run_baseline=run_baseline,)

    print(f' (1) set seed')
    generator = torch.cuda.manual_seed(args.seed)
    width = 512
    height = 512
    steps = 50,
    init_image_strength = 1.0,
    reverse = True
    fixed_starting_latent = None
    # Preprocess image if it exists (img2img)
    if init_image is not None:
        if isinstance(init_image, list):
            if isinstance(init_image[0], torch.Tensor):
                init_latent = [t.clone() for t in init_image]
            else:
                init_latent = [image_to_latent(im) for im in init_image]
        else:
            init_latent = image_to_latent(init_image)
        t_limit = steps - int(steps * init_image_strength)
    else:
        init_latent = torch.zeros((1, unet.in_channels, height // 8, width // 8), device=device)
        t_limit = 0

    if reverse:
        latent = init_latent
    else:
        noise = torch.randn(init_latent.shape,generator=generator,device=device,dtype=torch.float64)
        if fixed_starting_latent is None:
            latent = noise
        else:
            if isinstance(fixed_starting_latent, list):
                latent = [l.clone() for l in fixed_starting_latent]
            else:
                latent = fixed_starting_latent.clone()
            t_limit = steps - int(steps * init_image_strength)
    if isinstance(latent, list):  # initializing from pair of images
        latent_pair = latent
    else:  # initializing from noise
        # make coupled latent list
        print(f'make coupled latent list')
        latent_pair = [latent.clone(), latent.clone()]


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='EDICT')
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--seed', type=int, default=64)
    parser.add_argument('--img_dir', type=str, default='experiment_images/church.jpg')
    args=parser.parse_args()
    main(args)