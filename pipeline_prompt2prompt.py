from typing import Callable, List, Optional, Union
import numpy as np
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.models.cross_attention import CrossAttention

#from pipeline_sd import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
import torch.nn.functional as F
from ptp_utils import AttentionStore
import ptp_utils
from PIL import Image

from condition_adaptor_src.condition_adaptor_model import ConditionAdaptor
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler,DDIMSampler2
import torch.nn as nn
import cv2

class Prompt2PromptPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]
    
    def __init__(
        self,
        vae: None,
        text_encoder: None,
        tokenizer: None,
        unet: None,
        scheduler: None,
        safety_checker: None,
        feature_extractor: None,
        requires_safety_checker: bool = True,
        model_config: Optional[str] = 'configs/lcdg/sdv2_512_t2i_mask_std_pl.yaml',
        model_weight: Optional[str] = 'workdir_std/no_finding/checkpoints/LAST_epoch_23.pth',
    ):
        super(Prompt2PromptPipeline, self).__init__(vae,text_encoder,tokenizer,unet,scheduler,safety_checker,feature_extractor,requires_safety_checker)
        
        
        model_configs = OmegaConf.load(model_config)
        self.diffusion_model = instantiate_from_config(model_configs['model'])  #在这里是training True 
        self.diffusion_model.eval().cuda()   #在这里就是training False，后面都遍历一遍加上no grad试一下
        
        self.vq_model = self.diffusion_model.first_stage_model
        self.vq_model.cuda().eval()
        
        
        self.lcdg_model = ConditionAdaptor(
        time_channels=model_configs['condition_adaptor_config']['time_channels'],
        in_channels=model_configs['condition_adaptor_config']['in_channels'],
        out_channels=model_configs['condition_adaptor_config']['out_channels'],).cuda()
        self.lcdg_model.init_weights(init_type='xavier')
        self.lcdg_model.cuda()
        self.lcdg_model.load_state_dict(torch.load(model_weight, map_location='cpu')['model_state_dict'])
        self.CA_configs = model_configs['condition_adaptor_config']
        

        
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        controller: AttentionStore = None, # 传入attention_store作为p2p的控制。
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,  
        
    ############### modify list
        condition_adapter: bool = False,
        p2p: bool = False,
        target_cond: Optional[torch.FloatTensor] = None,
        plmaskno: Optional[int] = 1,
        cond_weight: Optional[float] = 2.0,
        label:Optional[str] = "pil",
        iter: Optional[int] = 300000
        
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        #dkm 初始化lcdg参数
        if condition_adapter:
            target_cond = self.vq_model.encode_to_codebook(target_cond)     #这时候没有图像了      
            cond_criterion = nn.MSELoss()
            cond_configs = self.CA_configs
            blocks_indexes = self.CA_configs['blocks']
            cond_scale = cond_weight
        
        if p2p:
            self.register_attention_control(controller) # add attention controller  #在这里已经给出了替换的约束了

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0    #调这个参数

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )#[4,77,768] [[neg,neg,pos1,pos2]]

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )  #[2,4,64,64]不一样的

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                #[4,4,64,64]
                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  
                #[rand1,rand2,rand1,rand2]

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                #[4,4,64,64]

                # perform guidance
                if do_classifier_free_guidance:
                    
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)   #输出分别是[2,4,64,64]
                    #这里之前都可以看做两句话并行处理
                    ##################### dkm 计算并添加梯度
                    if condition_adapter:      
                        #后来复制的两份
                        #dkm 初始化lcdg输入
                        x_in = latent_model_input    #对X也展示一下
                        t_in = torch.cat([t.unsqueeze(0)] * 2)
                        c_in = text_embeddings
                        # target_cond0 = torch.cat([target_cond]*2)
                        # gradient,mask_pred = self.cond_fn(x=x_in.chunk(2)[0], c=c_in.chunk(2)[1], t=t_in.chunk(2)[0], cond_configs=cond_configs, blocks_indexes=blocks_indexes,
                        #         target_cond=target_cond,criterion=cond_criterion)#有信息的部分才输入进来
                       
                        #x:[rand1,rand2],c:[pos1,pos2]
                        
                        # gradient1, gradient2 = gradient.chunk(2)
                        
                        # gradient2 = torch.zeros([1,4,64,64]).cuda()
                        # gradient1 = gradient1 * cond_scale * torch.norm(input=noise_pred_text.chunk(2)[0] - x_in.chunk(2)[0].chunk(2)[0], p=2) / torch.norm(input=gradient1, p=2)  
                        
                        #plmaskno
                        if plmaskno == 1:
                            #plmaskpl
                            gradient1 = torch.zeros([1,4,64,64]).cuda()
                            gradient2,mask_pred = self.cond_fn(x=x_in.chunk(2)[0].chunk(2)[1], c=c_in.chunk(2)[1].chunk(2)[1], t=t_in.chunk(2)[0].chunk(2)[1], cond_configs=cond_configs, blocks_indexes=blocks_indexes,
                                    target_cond=target_cond,criterion=cond_criterion)#有信息的部分才输入进来
                            gradient2 = gradient2 * cond_scale * torch.norm(input=noise_pred_text.chunk(2)[1] - x_in.chunk(2)[0].chunk(2)[1], p=2) / torch.norm(input=gradient2, p=2)                        # TODO: normalization weight: important to keep the values balance
                            gradient = torch.cat([gradient1,gradient2])
                            
                        elif plmaskno == 5:
                            #plmasknopl
                            gradient1 = torch.zeros([1,4,64,64]).cuda()
                            gradient2,mask_pred = self.cond_fn(x=x_in.chunk(2)[0].chunk(2)[1], c=c_in.chunk(2)[1].chunk(2)[1], t=t_in.chunk(2)[0].chunk(2)[1], cond_configs=cond_configs, blocks_indexes=blocks_indexes,
                                    target_cond=target_cond,criterion=cond_criterion)#有信息的部分才输入进来
                            gradient2 = gradient2 * cond_scale * torch.norm(input=noise_pred_text.chunk(2)[1] - x_in.chunk(2)[0].chunk(2)[1], p=2) / torch.norm(input=gradient2, p=2)                        # TODO: normalization weight: important to keep the values balance
                            gradient = torch.cat([gradient2,gradient2])
                        
                        elif plmaskno == 3:
                            #noplmasknopl
                            gradient1,mask_pred = self.cond_fn(x=x_in.chunk(2)[0].chunk(2)[0], c=c_in.chunk(2)[1].chunk(2)[0], t=t_in.chunk(2)[0].chunk(2)[0], cond_configs=cond_configs, blocks_indexes=blocks_indexes,
                            target_cond=target_cond,criterion=cond_criterion)#有信息的部分才输入进来
                            gradient2,mask_pred = self.cond_fn(x=x_in.chunk(2)[0].chunk(2)[1], c=c_in.chunk(2)[1].chunk(2)[1], t=t_in.chunk(2)[0].chunk(2)[1], cond_configs=cond_configs, blocks_indexes=blocks_indexes,
                            target_cond=target_cond,criterion=cond_criterion)#有信息的部分才输入进来
                            gradient1 = gradient1 * cond_scale * torch.norm(input=noise_pred_text.chunk(2)[0] - x_in.chunk(2)[0].chunk(2)[0], p=2) / torch.norm(input=gradient1, p=2)
                            gradient2 = gradient2 * cond_scale * torch.norm(input=noise_pred_text.chunk(2)[1] - x_in.chunk(2)[0].chunk(2)[1], p=2) / torch.norm(input=gradient2, p=2)                       
                            gradient = torch.cat([gradient1,gradient2])
                        
                        elif plmaskno == 4:
                            #nomasknopl
                            gradient1,mask_pred = self.cond_fn(x=x_in.chunk(2)[0].chunk(2)[0], c=c_in.chunk(2)[1], t=t_in.chunk(2)[0], cond_configs=cond_configs, blocks_indexes=blocks_indexes,
                            target_cond=target_cond,criterion=cond_criterion)#有信息的部分才输入进来
                            gradient2 = torch.zeros([1,4,64,64]).cuda()
                            gradient1 = gradient1 * cond_scale * torch.norm(input=noise_pred_text.chunk(2)[0] - x_in.chunk(2)[0].chunk(2)[0], p=2) / torch.norm(input=gradient1, p=2)  
                            gradient = torch.cat([gradient1,gradient1])
                            
                            
                        #mask_pred       
                        #nomaskno
                        
                        else:
                            gradient1,mask_pred = self.cond_fn(x=x_in.chunk(2)[0], c=c_in.chunk(2)[1].chunk(2)[0], t=t_in.chunk(2)[0].chunk(2)[0], cond_configs=cond_configs, blocks_indexes=blocks_indexes,
                                    target_cond=target_cond,criterion=cond_criterion)#有信息的部分才输入进来
                            gradient = gradient1 * cond_scale * torch.norm(input=noise_pred_text.chunk(2)[0] - x_in.chunk(2)[0].chunk(2)[0], p=2) / torch.norm(input=gradient1, p=2)  
                            
                            

                        # TODO: normalization weight: important to keep the values balance

                        # gradient = gradient * cond_scale * torch.norm(input=noise_pred_text - x_in.chunk(2)[0], p=2) / torch.norm(input=gradient, p=2)                        # TODO: normalization weight: important to keep the values balance
                        
                        
                        #g:[2,4,64,64]
                        #把g分成两部分，分别叠加到noise_pred的两个dim上
                        
                        
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)+ gradient
                        #不确定能否支持dim=2的计算，支持，但是效果可能不佳
                    ##################### dkm 计算并添加梯度
                    
                    else:
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                # 又回到了[2,4,64,64]，而且这里面已经完成了p2p的约束

                # step callback
                if p2p:
                    latents = controller.step_callback(latents)     #dkm P2P多在了这一步上  每走一步进行call back

                
                # call the callback, if provided  如果不call back呢
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)   #无关
                        
                # if (i % 200 == 0) or (i >= 450 and i % 10 == 0):
                #     cond_image = self.decode_latents(mask_pred)
                #     cond_image = self.numpy_to_pil(cond_image)
                #     cond_image[0].save('work_dir_diffuser/sketch/adapter_chestxray11/image_'+label+'_'+str(iter)+'_upsamplestep_'+str(i)+'.jpg')
                #     mid_image = self.decode_latents(latents)
                #     mid_image = self.numpy_to_pil(mid_image)
                #     # mid_image[1].save('work_dir_diffuser/sketch/adapter_chestxray6/image_'+label+'_300000_imgstep_'+str(i)+'.jpg')
                #     mid_image[0].save('work_dir_diffuser/sketch/adapter_chestxray11/image_'+label+'_'+str(iter)+'_imgstep_'+str(i)+'.jpg')
        # if condition_adapter:           
        #     controller.reset()

        # 8. Post-processing
        image = self.decode_latents(latents)    #都需要在这里解码

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def cond_fn(self, x, c, t, cond_configs, blocks_indexes,
            target_cond, criterion, scale=1.0):   #默认scale为1.0

        with torch.enable_grad():
            target_cond = target_cond.requires_grad_(True)    #target_cond 是mask
            x = x.requires_grad_(True)     #x 随机生成的噪声； c 是文本特征
            
            features = self.diffusion_model.model.diffusion_model.forward_return_features(x=x, timesteps=t, context=c, block_indexes=blocks_indexes)#这一步都是正常的生成采样
            # upsample features
            upsampled_features = []
            for feat in features:
                feat = F.interpolate(input=feat, size=cond_configs['size'], mode='bilinear')
                upsampled_features.append(feat)
            upsampled_features = torch.cat(upsampled_features, dim=1)   #大面积上采样
            
            # compute the gradient
            x_pred = self.lcdg_model(upsampled_features, t)#预测出一个mask，需要查看! 这是从有病的图像中预测出来的sketch，是否准确？
            #[1,7040,64,64]
            #即使输出变了，输入并没有发生大变化？
            
            
            # compute loss value
            loss = criterion(target_cond, x_pred)  #target_cond是通过计算梯度传的.这个loss一定要改,相当于是有病的sketch和无病的sketch直接做loss，难道要warping后再做loss？
            
            grad = torch.autograd.grad(loss, x, allow_unused=True)[0] * scale      # original: loss.sum()
            
            return grad,x_pred

    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]  #得到attention模块的每一个隐藏层
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )#得到attention模块的每一个隐藏层

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count


    def aggregate_attention(self, prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
        out = []
        attention_maps = attention_store.get_average_attention()
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out.cpu()


    def show_cross_attention(self, prompts, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
        tokens = self.tokenizer.encode(prompts[select])
        decoder = self.tokenizer.decode
        attention_maps = self.aggregate_attention(prompts, attention_store, res, from_where, True, select)
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)
        ptp_utils.view_images(np.stack(images, axis=0))
        

    def show_self_attention_comp(self, prompts, attention_store: AttentionStore, res: int, from_where: List[str],
                            max_com=10, select: int = 0):
        attention_maps = self.aggregate_attention(prompts, attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
        u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
        images = []
        for i in range(max_com):
            image = vh[i].reshape(res, res)
            image = image - image.min()
            image = 255 * image / image.max()
            image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
            image = Image.fromarray(image).resize((256, 256))
            image = np.array(image)
            images.append(image)
        ptp_utils.view_images(np.concatenate(images, axis=1))


class P2PCrossAttnProcessor:

    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states