import inspect
from typing import List, Optional, Union
import torch
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import CrossAttention
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler

from .attention import CrossAttention_forward_lite

class StableDiffusionPipelineLite(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.torch_device = "cpu"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # loads the model
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # applies patch to all CrossAttention forward calls 
        CrossAttention.forward = CrossAttention_forward_lite
        return model

    def prompt_to_embedding(self, prompt, batch_size, do_classifier_free_guidance):
        """
        takes a prompt and generate an embedding
        """
        # move model to device
        self.text_encoder.to(self.torch_device)

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.torch_device))[0]

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.torch_device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get model back
        self.text_encoder.to("cpu")
        return text_embeddings

    def initialize_latents(self, batch_size, height, width, generator):
        """initialize random latents"""
        # get the intial random noise
        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.torch_device,
        )

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents.mull_(self.scheduler.sigmas[0])

        return latents

 
    def initialise_scheduler(self, num_inference_steps, eta):
        """initialize the scheduler"""
        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        return extra_step_kwargs

    def update_latents(self, i, t, latents, text_embeddings, guidance_scale, do_classifier_free_guidance, extra_step_kwargs):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[i]
            latent_model_input.div_((sigma**2 + 1) ** 0.5)

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
        else:
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]
        
        return latents

    def latent_to_image(self, latents, output_type):
        """taken a latent and returns an image"""
        # moves model to device
        self.vae.to(self.torch_device)

        # scale and decode the image latents with vae
        latents.div_(0.18215)
        image = self.vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # get model back
        self.vae.to("cpu")
        return self.numpy_to_pil(image) if (output_type == "pil") else image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        **kwargs,
    ):
        """
        runs the full pipeline
        """
        # NOTE: we default to a batch size of one as we are memory constrained anyway
        if not isinstance(prompt, str):
            raise ValueError(f"`prompt` has to be of type `str` but is {type(prompt)}")
        batch_size = 1

        # make sure that our sizes is correct
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # turns the prompt into an embedding
        text_embeddings = self.prompt_to_embedding(prompt, batch_size, do_classifier_free_guidance)

        # gets the scheduler ready
        extra_step_kwargs = self.initialise_scheduler(num_inference_steps, eta)

        # get the intial random noise
        latents = self.initialize_latents(batch_size, height, width, generator)

        # updates the latents
        self.unet.to(self.torch_device) # moves model to device
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            latents = self.update_latents(i, t, latents, text_embeddings, guidance_scale, do_classifier_free_guidance, extra_step_kwargs)
        self.unet.to("cpu") # gets model back

        # this gets filled with nan!
        print(f"latents end:{latents}")

        # decodes the latent
        image = self.latent_to_image(latents, output_type)
        return {"sample": image, "nsfw_content_detected": False}
   
    def to(self, torch_device):
        """
        stores information on the device to be used
        the model will be moved piece-wise to minimise GPU memory usage
        """
        self.torch_device = torch_device
        return self