from edict_functions import coupled_stablediffusion, load_im_into_format_from_path
import matplotlib.pyplot as plt
import torch, random

def plot_EDICT_outputs(im_tuple):
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(im_tuple[0])
    ax1.imshow(im_tuple[1])
    plt.show()

"""
@torch.no_grad()
def coupled_stablediffusion(prompt="",
                            prompt_edit=None,
                            null_prompt='',
                            prompt_edit_token_weights=[],
                            prompt_edit_tokens_start=0.0,
                            prompt_edit_tokens_end=1.0,
                            prompt_edit_spatial_start=0.0,
                            prompt_edit_spatial_end=1.0,
                            guidance_scale=7.0, steps=50,
                            seed=1, width=512, height=512,
                            init_image=None, init_image_strength=1.0,
                            run_baseline=False,
                            use_lms=False,
                            leapfrog_steps=True,
                            reverse=False,
                            return_latents=False,
                            fixed_starting_latent=None,
                            beta_schedule='scaled_linear',
                            mix_weight=0.93):
    # If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None: seed = random.randrange(2 ** 32 - 1)
    generator = torch.cuda.manual_seed(seed)

    def image_to_latent(im):
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

    assert not use_lms, "Can't invert LMS the same as DDIM"
    if run_baseline: leapfrog_steps = False
    # Change size to multiple of 64 to prevent size mismatches inside model
    width = width - width % 64
    height = height - height % 64

    # Preprocess image if it exists (img2img)
    if init_image is not None:
        assert reverse  # want to be performing deterministic noising 
        # can take either pair (output of generative process) or single image
        if isinstance(init_image, list):
            if isinstance(init_image[0], torch.Tensor):
                init_latent = [t.clone() for t in init_image]
            else:
                init_latent = [image_to_latent(im) for im in init_image]
        else:
            init_latent = image_to_latent(init_image)
        # this is t_start for forward, t_end for reverse
        t_limit = steps - int(steps * init_image_strength)
    else:
        assert not reverse, 'Need image to reverse from'
        init_latent = torch.zeros((1, unet.in_channels, height // 8, width // 8), device=device)
        t_limit = 0

    if reverse:
        latent = init_latent
    else:
        # Generate random normal noise
        noise = torch.randn(init_latent.shape,
                            generator=generator,
                            device=device,
                            dtype=torch.float64)
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
        latent_pair = [latent.clone(), latent.clone()]

    if steps == 0:
        if init_image is not None:
            return image_to_latent(init_image)
        else:
            image = vae.decode(latent.to(vae.dtype) / 0.18215).sample
            return prep_image_for_return(image)

    # Set inference timesteps to scheduler
    schedulers = []
    for i in range(2):
        # num_raw_timesteps = max(1000, steps)
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                  beta_schedule=beta_schedule,
                                  num_train_timesteps=1000,
                                  clip_sample=False,
                                  set_alpha_to_one=False)
        scheduler.set_timesteps(steps)
        schedulers.append(scheduler)

    # CLIP Text Embeddings
    tokens_unconditional = clip_tokenizer(null_prompt, padding="max_length",
                                          max_length=clip_tokenizer.model_max_length,
                                          truncation=True, return_tensors="pt",
                                          return_overflowing_tokens=True)
    embedding_unconditional = clip(tokens_unconditional.input_ids.to(device)).last_hidden_state

    tokens_conditional = clip_tokenizer(prompt, padding="max_length",
                                        max_length=clip_tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt",
                                        return_overflowing_tokens=True)
    embedding_conditional = clip(tokens_conditional.input_ids.to(device)).last_hidden_state

    # Process prompt editing (if running Prompt-to-Prompt)
    if prompt_edit is not None:
        tokens_conditional_edit = clip_tokenizer(prompt_edit, padding="max_length",
                                                 max_length=clip_tokenizer.model_max_length,
                                                 truncation=True, return_tensors="pt",
                                                 return_overflowing_tokens=True)
        embedding_conditional_edit = clip(tokens_conditional_edit.input_ids.to(device)).last_hidden_state

        init_attention_edit(tokens_conditional, tokens_conditional_edit)

    init_attention_func()
    init_attention_weights(prompt_edit_token_weights)

    timesteps = schedulers[0].timesteps[t_limit:]
    if reverse: timesteps = timesteps.flip(0)

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        t_scale = t / schedulers[0].num_train_timesteps

        if (reverse) and (not run_baseline):
            # Reverse mixing layer
            new_latents = [l.clone() for l in latent_pair]
            new_latents[1] = (new_latents[1].clone() - (1 - mix_weight) * new_latents[0].clone()) / mix_weight
            new_latents[0] = (new_latents[0].clone() - (1 - mix_weight) * new_latents[1].clone()) / mix_weight
            latent_pair = new_latents

        # alternate EDICT steps
        for latent_i in range(2):
            if run_baseline and latent_i == 1: continue  # just have one sequence for baseline
            # this modifies latent_pair[i] while using 
            # latent_pair[(i+1)%2]
            if reverse and (not run_baseline):
                if leapfrog_steps:
                    # what i would be from going other way
                    orig_i = len(timesteps) - (i + 1)
                    offset = (orig_i + 1) % 2
                    latent_i = (latent_i + offset) % 2
                else:
                    # Do 1 then 0
                    latent_i = (latent_i + 1) % 2
            else:
                if leapfrog_steps:
                    offset = i % 2
                    latent_i = (latent_i + offset) % 2

            latent_j = ((latent_i + 1) % 2) if not run_baseline else latent_i

            latent_model_input = latent_pair[latent_j]
            latent_base = latent_pair[latent_i]

            # Predict the unconditional noise residual
            noise_pred_uncond = unet(latent_model_input, t,
                                     encoder_hidden_states=embedding_unconditional).sample

            # Prepare the Cross-Attention layers
            if prompt_edit is not None:
                save_last_tokens_attention()
                save_last_self_attention()
            else:
                # Use weights on non-edited prompt when edit is None
                use_last_tokens_attention_weights()

            # Predict the conditional noise residual and save the cross-attention layer activations
            noise_pred_cond = unet(latent_model_input, t,
                                   encoder_hidden_states=embedding_conditional).sample

            # Edit the Cross-Attention layer activations
            if prompt_edit is not None:
                t_scale = t / schedulers[0].num_train_timesteps
                if t_scale >= prompt_edit_tokens_start and t_scale <= prompt_edit_tokens_end:
                    use_last_tokens_attention()
                if t_scale >= prompt_edit_spatial_start and t_scale <= prompt_edit_spatial_end:
                    use_last_self_attention()

                # Use weights on edited prompt
                use_last_tokens_attention_weights()

                # Predict the edited conditional noise residual using the cross-attention masks
                noise_pred_cond = unet(latent_model_input,
                                       t,
                                       encoder_hidden_states=embedding_conditional_edit).sample

            # Perform guidance
            grad = (noise_pred_cond - noise_pred_uncond)
            noise_pred = noise_pred_uncond + guidance_scale * grad

            step_call = reverse_step if reverse else forward_step
            new_latent = step_call(schedulers[latent_i],
                                   noise_pred,
                                   t,
                                   latent_base)  # .prev_sample
            new_latent = new_latent.to(latent_base.dtype)

            latent_pair[latent_i] = new_latent

        if (not reverse) and (not run_baseline):
            # Mixing layer (contraction) during generative process
            new_latents = [l.clone() for l in latent_pair]
            new_latents[0] = (mix_weight * new_latents[0] + (1 - mix_weight) * new_latents[1]).clone()
            new_latents[1] = ((1 - mix_weight) * new_latents[0] + (mix_weight) * new_latents[1]).clone()
            latent_pair = new_latents

    # scale and decode the image latents with vae, can return latents instead of images
    if reverse or return_latents:
        results = [latent_pair]
        return results if len(results) > 1 else results[0]

    # decode latents to iamges
    images = []
    for latent_i in range(2):
        latent = latent_pair[latent_i] / 0.18215
        image = vae.decode(latent.to(vae.dtype)).sample
        images.append(image)

    # Return images
    return_arr = []
    for image in images:
        image = prep_image_for_return(image)
        return_arr.append(image)
    results = [return_arr]
    return results if len(results) > 1 else results[0]

diffusion_result = coupled_stablediffusion('A black bear')
#plot_EDICT_outputs()
"""
import argparse
from transformers import CLIPModel, CLIPTokenizer
from EDICT.my_diffusers import AutoencoderKL, UNet2DConditionModel
from EDICT.my_diffusers import LMSDiscreteScheduler, PNDMScheduler, DDPMScheduler, DDIMScheduler

def main(args) :

    # Push to devices w/ double precision
    device = args.device

    print(f'step 1. model build')

    # Build our CLIP model
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

    print(f'step 2. set seed')
    if args.seed is None: seed = random.randrange(2 ** 32 - 1)
    generator = torch.cuda.manual_seed(seed)

    print(f'step 3. set start latent')
    width, height = 512, 512
    init_latent = torch.zeros((1, unet.in_channels, height // 8, width // 8), device=device)
    t_limit = 0
    noise = torch.randn(init_latent.shape,generator=generator,device=device,dtype=torch.float64)
    latent = noise
    latent_pair = [latent.clone(), latent.clone()]


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='EDICT')
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--seed', type=int, default=64)
    args=parser.parse_args()
    main(args)