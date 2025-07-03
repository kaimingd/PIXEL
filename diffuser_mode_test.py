from pipeline_prompt2prompt import Prompt2PromptPipeline
from ptp_utils import AttentionStore, AttentionReplace, LocalBlend, AttentionRefine, AttentionReweight, view_images, get_equalizer
import torch
import numpy as np
from diffusers import StableDiffusionPipeline,DDIMScheduler
from condition_adaptor_src.condition_adaptor_model import ConditionAdaptor

random_int = 300000
g_cpu = torch.Generator().manual_seed(random_int)
device = "cuda"
pipe = Prompt2PromptPipeline.from_pretrained("pretrained_mimic_diffusion" ).to(device)
prompts = ["a photo of xray with no finding"]
NUM_DIFFUSION_STEPS = 500
outputs = pipe(prompt=prompts, height=512, width=512, num_inference_steps=NUM_DIFFUSION_STEPS, generator=g_cpu)
outputs['images'][0].save('reference_data/image_300000_ref.jpg')


