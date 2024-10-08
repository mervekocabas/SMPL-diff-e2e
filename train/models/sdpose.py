import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, StableDiffusionPipeline, DDPMScheduler

from .head.smplx_cam_head import SMPLXCamHead
from .head.smplx_cam_head_proj import SMPLXCamHeadProj
from .head.smplx_head import SMPLXHead
from .head.smpl_cam_head import SMPLCamHead
from ..utils.rotations import axis_angle_to_rotation_6d, rotation_6d_to_matrix
from ..core.config import SMPL_MEAN_PARAMS
from ..core.constants import NUM_JOINTS_SMPLX, BN_MOMENTUM


class SDPose(nn.Module):
    def __init__(self, hparams=None, precision=torch.float32, img_res=224,):
        super(SDPose, self).__init__()
        self.hparams = hparams
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")

        # pipe.enable_sequential_cpu_offload()
        # pipe.unet.to(memory_format=torch.channels_last)
        # pipe.enable_attention_slicing(1)
        pipe.enable_vae_slicing()
            
        empty_text_embed_path = 'data/empty_text_embedding.pt'
        self.register_buffer('empty_text_embed', torch.load(empty_text_embed_path))
        
        self.vae = pipe.vae
        del self.vae.decoder
        self.vae.requires_grad_(False)
        
        if hparams.DATASET.proj_verts:
            self.smpl = SMPLXCamHeadProj(img_res=img_res) 
        else:
            self.smpl = SMPLXCamHead(img_res=img_res)
                    
        self.noise_scheduler = pipe.scheduler
        self.unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-2", 
            torch_dtype=precision, 
            subfolder="unet",
            in_channels=4, out_channels=1,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        )
        print('Initializing the unet conv_out with pretrained weights')
        
        self.unet.conv_out.weight.data[:] = pipe.unet.conv_out.weight.data.mean(dim=0, keepdim=True).detach().clone() 
        self.unet.conv_out.bias.data[:] = pipe.unet.conv_out.bias.data.mean(dim=0, keepdim=True).detach().clone() 
        
        self.unet.requires_grad_(True)
        # self.unet.enable_attention_slicing(1)
        self.unet.to(memory_format=torch.channels_last)
        self.unet.enable_xformers_memory_efficient_attention()
        self.unet.enable_gradient_checkpointing()
        
        del pipe
    
    def forward(
        self, 
        rgb_latents, 
        bbox_scale, 
        bbox_center,
        img_w, 
        img_h, 
        fl,
        gt_joints2d=None,
        gt_cam=None,
    ):
        noise_pred = self.denoise_forward(rgb_latents, 1000)
        pred_smplx_params = self.decode_smplx_params(noise_pred)
        pred = self.forward_smplx_params(
            pred_smplx_params, 
            bbox_scale=bbox_scale, 
            bbox_center=bbox_center, 
            img_w=img_w, 
            img_h=img_h, 
            fl=fl,
            gt_joints2d=gt_joints2d,
            gt_cam=gt_cam,
        )
        pred['noise_pred'] = noise_pred
        return pred
        
    def denoise_forward(self, noisy_latents, timestep):
        # (noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
        tt = torch.cat([timestep] * 1)
        embed = self.empty_text_embed.repeat(noisy_latents.shape[0], 1, 1)
        denoised_latents = self.unet(noisy_latents, tt, encoder_hidden_states=embed, return_dict=False)[0]
        return denoised_latents
    
    @torch.no_grad()
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
    
    def forward_smplx_params(
        self,
        pred_smplx_params,
        bbox_scale=None,
        bbox_center=None,
        img_w=None,
        img_h=None,
        fl=None,
        gt_joints2d=None,
        gt_cam=None,
    ):
        
        batch_size = pred_smplx_params['pred_pose'].shape[0] # images.shape[0]

        if fl is not None:
            # GT focal length
            focal_length = fl
        else:
            # Estimate focal length
            focal_length = (img_w * img_w + img_h * img_h) ** 0.5
            focal_length = focal_length.repeat(2).view(batch_size, 2)

        # Initialze cam intrinsic matrix
        cam_intrinsics = torch.eye(3).repeat(batch_size, 1, 1).cuda().float()
        cam_intrinsics[:, 0, 0] = focal_length[:, 0]
        cam_intrinsics[:, 1, 1] = focal_length[:, 1]
        cam_intrinsics[:, 0, 2] = img_w/2.
        cam_intrinsics[:, 1, 2] = img_h/2.
        
        if self.hparams.TRIAL.bedlam_bbox:
            # Assuming prediction are in camera coordinate
            smpl_output = self.smpl(
                rotmat=pred_smplx_params['pred_pose'],
                shape=pred_smplx_params['pred_shape'],
                cam=pred_smplx_params['pred_cam'],
                cam_intrinsics=cam_intrinsics,
                bbox_scale=bbox_scale,
                bbox_center=bbox_center,
                img_w=img_w,
                img_h=img_h,
                normalize_joints2d=False,
                gt_joints2d=gt_joints2d,
                gt_cam=gt_cam,
            )
        else:
            smpl_output = self.smpl(
                rotmat=pred_smplx_params['pred_pose'],
                shape=pred_smplx_params['pred_shape'],
                cam=pred_smplx_params['pred_cam'],
                normalize_joints2d=True,
            )
        smpl_output.update(pred_smplx_params)
        return smpl_output
    
    def decode_smplx_params(self, smplx_latents):
        batch_size = smplx_latents.shape[0]
        smplx_latents_vec = smplx_latents.reshape(batch_size, -1)
        
        res_pred_pose_6d = smplx_latents_vec[:, :132]
        res_pred_shape = smplx_latents_vec[:, 132:132+11]
        res_pred_cam = smplx_latents_vec[:, 132+11:132+11+3]
        
        pred_rotmat = rotation_6d_to_matrix(pred_pose_6d.reshape(-1, 6)).reshape(batch_size, -1, 3, 3)
        pred_smplx_params = {
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose_6d': pred_pose_6d,
        }
        
        return pred_smplx_params
    