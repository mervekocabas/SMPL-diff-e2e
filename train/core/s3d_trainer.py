import os
import cv2
import torch
import smplx
import pickle
import numpy as np
from loguru import logger
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F

from . import constants
from . import config
from ..losses.losses import HMRLoss
from ..utils.renderer import Renderer
from .constants import NUM_JOINTS_SMPLX
from ..utils.train_utils import set_seed
from ..dataset.dataset_sd import DatasetSDPose
from ..utils.image_utils import denormalize_images
from ..utils.eval_utils import reconstruction_error
from ..utils.renderer_cam import render_image_group
from ..utils.geometry import estimate_translation_cam
from ..models.head.smplx_cam_head import convert_full_img_cam_t_to_weak_cam, convert_pare_to_full_img_cam
from diffusers import DDIMScheduler

class SDPoseTrainer(pl.LightningModule):

    def __init__(self, hparams):
        super(SDPoseTrainer, self).__init__()

        self.hparams.update(hparams)

        self.model = SDPose(
            hparams=self.hparams,
            img_res=self.hparams.DATASET.IMG_RES,
        )
        self.loss_fn = HMRLoss(hparams=self.hparams)

        self.smplx = smplx.SMPLX(config.SMPLX_MODEL_DIR, batch_size=self.hparams.DATASET.BATCH_SIZE, create_transl=False, num_betas=11)
        self.add_module('smplx', self.smplx)
        self.smpl = smplx.SMPL(config.SMPL_MODEL_DIR, batch_size=self.hparams.DATASET.BATCH_SIZE, create_transl=False)

        # Initialize the training datasets only in training mode
        if not hparams.RUN_TEST:
            self.train_ds = self.train_dataset()

        self.val_ds = self.val_dataset()
        self.save_itr = 0
        batch_size = self.hparams.DATASET.BATCH_SIZE
        self.smplx2smpl = pickle.load(open(config.SMPLX2SMPL, 'rb'))
        self.smplx2smpl = torch.tensor(self.smplx2smpl['matrix'][None], dtype=torch.float32).cuda()

        self.register_buffer(
            'J_regressor',
            torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
        )

        self.renderer = Renderer(
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.IMG_RES,
            faces=self.smplx.faces,
            mesh_color=self.hparams.DATASET.MESH_COLOR,
        )
        self.testing_gt_vis = self.hparams.TESTING.GT_VIS
        self.testing_wp_vis = self.hparams.TESTING.WP_VIS
        self.testing_fp_vis = self.hparams.TESTING.FP_VIS
        self.testing_mesh_vis = self.hparams.TESTING.MESH_VIS

        self.training_gt_vis = self.hparams.TRAINING.GT_VIS
        self.training_wp_vis = self.hparams.TRAINING.WP_VIS
        self.training_fp_vis = self.hparams.TRAINING.FP_VIS
        self.training_mesh_vis = self.hparams.TRAINING.MESH_VIS
        self.max_epochs = self.hparams.TRAINING.MAX_EPOCHS
        # they will be hparams
        # gaussian noise parameters
        self.noise_offset = 0.1
        self.input_perturbation = 0
        # epsilon v-prediction
        self.noise_prediction_type = "epsilon"
        # gaussian multi-resolution multi-resolution-annealed
        self.noise_type = "multi-resolution-annealed"
        self.gen = torch.tensor(0, dtype=torch.float32).cuda()
        self.generator = torch.Generator(device=self.gen.device)
        # bedlam seed 
        self.generator.manual_seed(1234)
        self.inference_noise_scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2", subfolder="scheduler")

    def training_step(self, batch, batch_nb, dataloader_nb=0):
        
        # GT data
        images = batch['img']
        gt_betas = batch['betas']
        bbox_scale = batch['scale']
        bbox_center = batch['center']
        img_h = batch['orig_shape'][:, 0]
        img_w = batch['orig_shape'][:, 1]
        fl = batch['focal_length']
        gt_pose = batch['pose']
        gt_cam_t = batch['translation'][:,:3]
        # Calculate joints and vertices using just 22 pose param for SMPL
        gt_out = self.smplx(
            betas=gt_betas,
            body_pose=gt_pose[:, 3:NUM_JOINTS_SMPLX*3],
            global_orient=gt_pose[:, :3]
        )
        
        batch['vertices'] = gt_out.vertices
        batch['joints3d'] = gt_out.joints
        joints3d = batch['joints3d']
        joints2d = batch['keypoints']
        joints2d_fullimg = batch['keypoints_orig']
        
        gt_cam = convert_full_img_cam_t_to_weak_cam(
            gt_cam_t, 
            bbox_height=bbox_scale * 200.,
            bbox_center=bbox_center,
            img_w=img_w,
            img_h=img_h,
            focal_length=fl[..., 0],
            crop_res=self.hparams.DATASET.IMG_RES,
        )
        # # sanity check cam_transl conversion
        # gt_cam_t_inv = convert_pare_to_full_img_cam(
        #     pare_cam=gt_cam,
        #     bbox_height=bbox_scale * 200.,
        #     bbox_center=bbox_center,
        #     img_w=img_w,
        #     img_h=img_h,
        #     focal_length=fl[..., 0],
        #     crop_res=self.hparams.DATASET.IMG_RES,
        # )
        # torch.testing.assert_close(gt_cam_t_inv, gt_cam_t, rtol=1e-5, atol=1e-5)

        bsz = batch['img'].shape[0]
       
        with torch.no_grad():
            rgb_latents = self.model.encode_imgs(images)
            
        pred = self.model.forward(
            rgb_latents, # input to the model
            bbox_scale, 
            bbox_center,
            img_w, 
            img_h, 
            fl,
            gt_joints2d=joints2d,
            gt_cam=gt_cam,
        )
        pred['pred_cam'] = gt_cam
        
        # Visualization for debugging
        if self.training_gt_vis:
                self.gt_projection(batch, pred, batch_nb)
        if self.training_wp_vis:
                self.weak_perspective_projection(batch, pred, batch_nb, dataloader_nb)
        if self.training_fp_vis:
                self.perspective_projection(batch, pred, batch_nb)
        if self.training_mesh_vis and self.global_step % 100 == 0:
                self.visualize_mesh(batch, pred, batch_nb, dataloader_nb, pred['vertices'], batch['vertices'])
     
        loss, loss_dict = self.loss_fn(pred=pred, gt=batch)
        
        for k, v in loss_dict.items():
            # print(f'{k}, val={v.item():.3f}, isnan={torch.isnan(v).any().item()}')
            if torch.isnan(v).any():
                print(f'{k} is nan')
                import ipdb; ipdb.set_trace()

        self.log('train_loss', loss, logger=True, sync_dist=True)

        # for k, v in loss_dict.items():
        #     self.log(k, v, logger=True, sync_dist=True)
        #     print(f'{k}, val={v.item():.3f}, isnan={torch.isnan(v).any().item()}')
        # print('--------')
        # print(pred['pred_cam'][0])
        # import ipdb; ipdb.set_trace()
        return {'loss': loss}

    def validation_step(self, batch, batch_nb, dataloader_nb=0, vis=False, save=True, mesh_save_dir=None):
        
        images = batch['img']
        batch_size = images.shape[0]
        bbox_scale = batch['scale']
        bbox_center = batch['center']
        dataset_names = batch['dataset_name']
        dataset_index = batch['dataset_index'].detach().cpu().numpy()
        val_dataset_names = self.hparams.DATASET.VAL_DS.split('_')
        img_h = batch['orig_shape'][:, 0]
        img_w = batch['orig_shape'][:, 1]
        J_regressor_batch_smpl = self.J_regressor[None, :].expand(batch['img'].shape[0], -1, -1)
        
        gt_cam_t = batch['translation'][:,:3]
        fl = batch['focal_length']
        gt_cam = convert_full_img_cam_t_to_weak_cam(
            gt_cam_t, 
            bbox_height=bbox_scale * 200.,
            bbox_center=bbox_center,
            img_w=img_w,
            img_h=img_h,
            focal_length=fl[..., 0],
            crop_res=self.hparams.DATASET.IMG_RES,
        )
        with torch.no_grad():
            # Encode image
            rgb_latents = self.model.encode_imgs(images)

        accumulated_smplx_params = None

        with torch.no_grad():
            # Set timesteps
            self.inference_noise_scheduler.set_timesteps(num_inference_steps=50, device = self.gen.device)
            timesteps = self.inference_noise_scheduler.timesteps  # T
            
            # Initial sp latents (noise)
                        
            smplx_latents = torch.randn(
                (batch_size, 1, rgb_latents[0][1][0].shape[0], rgb_latents[0][1][1].shape[0]),
                device=self.gen.device,
                dtype=torch.float16,
                generator=self.generator,
            )
            
            # Denoising loop
                        
            iterable = enumerate(timesteps)
            for i,t in iterable:
                unet_input = torch.cat((rgb_latents, smplx_latents), dim=1) # this order is important
                        
                # predict the noise residual
                embed = self.model.empty_text_embed.repeat(unet_input.shape[0], 1, 1)
                # noise_pred = self.finetuned_unet(unet_input, t, encoder_hidden_states=embed).sample
                noise_pred = self.model.unet(unet_input, t, encoder_hidden_states=embed).sample

                # compute the previous noisy sample x_t -> x_t-1
                smplx_latents = self.inference_noise_scheduler.step(
                    noise_pred, t, smplx_latents, generator=self.generator
                ).prev_sample
                    
        pred_smplx_params = self.model.decode_smplx_params(smplx_latents)
                #     import ipdb; ipdb.set_trace()
                #     if accumulated_smplx_params is None:
                #         accumulated_smplx_params = {key: value.clone() for key, value in pred_smplx_params.items()}
                #     else:
                #         for key in accumulated_smplx_params:
                #             accumulated_smplx_params[key] += pred_smplx_params[key]

                # # Compute the average of the predictions
                # averaged_smplx_params = {key: value / self.ensemble_size for key, value in accumulated_smplx_params.items()}
 
        pred = self.model.forward_smplx_params(
            pred_smplx_params, 
            bbox_scale=bbox_scale, 
            bbox_center=bbox_center, 
            img_w=img_w,                 
            img_h=img_h, 
            # 3DPW gt_joints2d = batch['keypoints'],
            gt_cam=gt_cam,
        )

        pred_cam_vertices = pred['vertices']
        joint_mapper_gt = constants.J24_TO_J14
        joint_mapper_h36m = constants.H36M_TO_J14
        
        gt_out_cam = self.smplx(
                betas=batch['betas'],
                body_pose=batch['pose'][:, 3:NUM_JOINTS_SMPLX*3],
                global_orient=batch['pose'][:, :3],
            )
        gt_cam_vertices = gt_out_cam.vertices
        gt_keypoints_3d = gt_out_cam.joints[:, :24]
        pred_keypoints_3d = pred['joints3d'][:, :24]
        gt_pelvis = (gt_keypoints_3d[:, [1], :] + gt_keypoints_3d[:, [2], :]) / 2.0
        pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0

        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
        pred_cam_vertices = pred_cam_vertices - pred_pelvis
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
        gt_cam_vertices = gt_cam_vertices - gt_pelvis
        
        if 'bedlam' in dataset_names[0]:
            gt_out_cam = self.smplx(
                betas=batch['betas'],
                body_pose=batch['pose'][:, 3:NUM_JOINTS_SMPLX*3],
                global_orient=batch['pose'][:, :3],
            )
            gt_cam_vertices = gt_out_cam.vertices
            gt_keypoints_3d = gt_out_cam.joints[:, :24]
            pred_keypoints_3d = pred['joints3d'][:, :24]
            gt_pelvis = (gt_keypoints_3d[:, [1], :] + gt_keypoints_3d[:, [2], :]) / 2.0
            pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0

            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_cam_vertices = pred_cam_vertices - pred_pelvis
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_cam_vertices = gt_cam_vertices - gt_pelvis
        elif 'rich' in dataset_names[0]:
            # For rich vertices are generated in dataset.py because gender is needed
            gt_cam_vertices = batch['vertices']
            gt_keypoints_3d = batch['joints']
            pred_cam_vertices = torch.matmul(self.smplx2smpl.repeat(batch_size, 1, 1), pred_cam_vertices)
            pred_keypoints_3d = torch.matmul(self.smpl.J_regressor, pred_cam_vertices)
            gt_pelvis = (gt_keypoints_3d[:, [1], :] + gt_keypoints_3d[:, [2], :]) / 2.0
            pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_cam_vertices = pred_cam_vertices - pred_pelvis
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_cam_vertices = gt_cam_vertices - gt_pelvis
        elif 'h36m' in dataset_names[0]:
            gt_cam_vertices = batch['vertices']
            # # Get 14 predicted joints from the mesh
            gt_keypoints_3d = batch['joints']
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            gt_keypoints_3d = gt_keypoints_3d - ((gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2).unsqueeze(1)
            pred_cam_vertices = torch.matmul(self.smplx2smpl.repeat(batch_size, 1, 1).cuda(), pred_cam_vertices)
            # # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch_smpl, pred_cam_vertices)
            # pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            # pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_keypoints_3d = pred_keypoints_3d - ((pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2).unsqueeze(1)
        # else:
        #     # For 3dpw vertices are generated in dataset.py because gender is needed
        #     gt_cam_vertices = batch['vertices']
        #     # Get 14 predicted joints from the mesh
        #     gt_keypoints_3d = torch.matmul(J_regressor_batch_smpl, gt_cam_vertices)
        #     gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
        #     gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
        #     gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
        #     gt_cam_vertices = gt_cam_vertices - gt_pelvis
        #     # Convert predicted vertices to SMPL Fromat
        #     pred_cam_vertices = torch.matmul(self.smplx2smpl.repeat(batch_size, 1, 1), pred_cam_vertices)
        #     # Get 14 predicted joints from the mesh
        #     pred_keypoints_3d = torch.matmul(J_regressor_batch_smpl, pred_cam_vertices)
        #     pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
        #     pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
        #     pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
        #     pred_cam_vertices = pred_cam_vertices - pred_pelvis

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).cpu().numpy()
        error_verts = torch.sqrt(((pred_cam_vertices - gt_cam_vertices) ** 2).sum(dim=-1)).cpu().numpy()

        # Reconstuction_error (PA-MPJPE)
        r_error, _ = reconstruction_error(
            pred_keypoints_3d.cpu().numpy(),
            gt_keypoints_3d.cpu().numpy(),
            reduction=None
        )
        val_mpjpe = error.mean(-1)
        val_pampjpe = r_error.mean(-1)
        val_pve = error_verts.mean(-1)

        # Visualize results
        if self.testing_gt_vis:
            self.gt_projection(batch, pred, batch_nb)
        if self.testing_mesh_vis:
            self.visualize_mesh(batch, pred, batch_nb, dataloader_nb, pred_cam_vertices, gt_cam_vertices)
        if self.testing_wp_vis:
            self.weak_perspective_projection(batch, pred, batch_nb, dataloader_nb)
        if self.testing_fp_vis:
            self.perspective_projection(batch, pred, batch_nb)

        loss_dict = {}

        for ds_idx, ds in enumerate(self.val_ds):
            ds_name = ds.dataset
            ds_idx = val_dataset_names.index(ds.dataset)
            idxs = np.where(dataset_index == ds_idx)
            loss_dict[ds_name + '_mpjpe'] = list(val_mpjpe[idxs])
            loss_dict[ds_name + '_pampjpe'] = list(val_pampjpe[idxs])
            loss_dict[ds_name + '_pve'] = list(val_pve[idxs])

        return loss_dict

    def validation_epoch_end(self, outputs):
        logger.info(f'***** Epoch {self.current_epoch} *****')
        val_log = {}

        if len(self.val_ds) > 1:
            for ds_idx, ds in enumerate(self.val_ds):
                ds_name = ds.dataset
                mpjpe = 1000 * np.hstack(np.array([val[ds_name + '_mpjpe'] for x in outputs for val in x])).mean()
                pampjpe = 1000 * np.hstack(np.array([val[ds_name + '_pampjpe'] for x in outputs for val in x])).mean()
                pve = 1000 * np.hstack(np.array([val[ds_name + '_pve'] for x in outputs for val in x])).mean()

                if self.trainer.is_global_zero:
                    logger.info(ds_name + '_MPJPE: ' + str(mpjpe))
                    logger.info(ds_name + '_PA-MPJPE: ' + str(pampjpe))
                    logger.info(ds_name + '_PVE: ' + str(pve))
                    val_log[ds_name + '_val_mpjpe'] = mpjpe
                    val_log[ds_name + '_val_pampjpe'] = pampjpe
                    val_log[ds_name + '_val_pve'] = pve
        else:
            for ds_idx, ds in enumerate(self.val_ds):
                ds_name = ds.dataset
                mpjpe = 1000 * np.hstack(np.array([x[ds_name + '_mpjpe'] for x in outputs])).mean()
                pampjpe = 1000 * np.hstack(np.array([x[ds_name + '_pampjpe'] for x in outputs])).mean()
                pve = 1000 * np.hstack(np.array([x[ds_name + '_pve'] for x in outputs])).mean()

                if self.trainer.is_global_zero:
                    logger.info(ds_name + '_MPJPE: ' + str(mpjpe))
                    logger.info(ds_name + '_PA-MPJPE: ' + str(pampjpe))
                    logger.info(ds_name + '_PVE: ' + str(pve))

                    val_log[ds_name + '_val_mpjpe'] = mpjpe
                    val_log[ds_name + '_val_pampjpe'] = pampjpe
                    val_log[ds_name + '_val_pve'] = pve

        self.log('val_loss', val_log[self.val_ds[0].dataset + '_val_pampjpe'], logger=True, sync_dist=True)
        self.log('val_loss_mpjpe', val_log[self.val_ds[0].dataset + '_val_mpjpe'], logger=True, sync_dist=True)
        for k, v in val_log.items():
            self.log(k, v, logger=True, sync_dist=True)

    def gt_projection(self, input_batch, output, batch_idx, max_save_img=1):
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images_gt')
        os.makedirs(save_dir, exist_ok=True)
        focal_length = input_batch['focal_length']

        gt_out = self.smplx(
            betas=input_batch['betas'],
            body_pose=input_batch['pose'][:, 3:NUM_JOINTS_SMPLX*3],
            global_orient=input_batch['pose'][:, :3]
            )

        gt_vertices = gt_out.vertices
        translation = input_batch['translation'][:, :3]
        for i in range(len(input_batch['imgname'])):
            dataset_name = input_batch['dataset_name'][i]
            imgname = input_batch['imgname'][i]
            cv_img = cv2.imread(imgname)
            if 'closeup' in dataset_name:
                cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
            cy, cx = input_batch['orig_shape'][i] // 2
            save_filename = os.path.join(save_dir, f'{self.global_step:08d}_{i:02d}_{os.path.basename(imgname)}')

            rendered_img = render_image_group(
                image=cv_img,
                camera_translation=translation[i],
                vertices=gt_vertices[i],
                focal_length=focal_length[i],
                camera_center=(cx, cy),
                camera_rotation=None,
                save_filename=save_filename,
                keypoints_2d=input_batch['keypoints_orig'][i].cpu().numpy(),
                faces=self.smplx.faces,
            )
            if i >= (max_save_img - 1):
                break

    def perspective_projection(self, input_batch, output, batch_idx, max_save_img=1):
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images_cliff')
        os.makedirs(save_dir, exist_ok=True)
        
        focal_length = input_batch['focal_length']
        translation = output['pred_cam_t'].detach()
        vertices = output['vertices'].detach()

        for i in range(len(input_batch['imgname'])):
            cy, cx = input_batch['orig_shape'][i] // 2
            img_h, img_w = cy*2, cx*2
            imgname = input_batch['imgname'][i]
            save_filename = os.path.join(save_dir, f'{self.global_step:08d}_{i:02d}_{os.path.basename(imgname)}')
            #focal_length_ = (img_w * img_w + img_h * img_h) ** 0.5  # Assumed fl
            #focal_length = (focal_length_, focal_length_)

            rendered_img = render_image_group(
                image=cv2.imread(imgname),
                camera_translation=translation[i],
                vertices=vertices[i],
                focal_length=focal_length[i],
                camera_center=(cx, cy),
                camera_rotation=None,
                save_filename=save_filename,
                faces=self.smplx.faces,
                keypoints_2d= output['joints2d'][:,:24][i].detach().cpu().numpy(),
            )
            if i >= (max_save_img - 1):
                break

    def visualize_mesh(self, input_batch, output, batch_idx, dataloader_nb, pc, gc, max_save=1):
        import trimesh
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images')
        os.makedirs(save_dir, exist_ok=True)
        
        pred_vertices = pc.detach().cpu().numpy()
        gt_vertices = gc.detach().cpu().numpy()
        for i in range(pred_vertices.shape[0]):
            imgname = input_batch['imgname'][i].split('/')[-1]
            gt = trimesh.Trimesh(vertices=gt_vertices[i]*np.array([1, -1, -1]), faces=self.smplx.faces, process=False)
            gt.visual.face_colors = [200, 200, 250, 100]
            gt.visual.vertex_colors = [200, 200, 250, 100]

            pred = trimesh.Trimesh(vertices=pred_vertices[i]*np.array([1, -1, -1]), faces=self.smplx.faces, process=False)
            save_filename = os.path.join(save_dir, f'{self.current_epoch:04d}_{dataloader_nb:02d}_'
                                f'{batch_idx:05d}_{i:02d}_{os.path.basename(imgname)}')
            gt.export(save_filename+str(i)+'_gt.obj')
            pred.export(save_filename+str(i)+'_pred.obj')   
            if i >= (max_save - 1):
                break

    def weak_perspective_projection(self, input_batch, output, batch_idx, dataloader_nb):
        pred_vertices = output['vertices'].detach()
        images = input_batch['img']
        images = denormalize_images(images)
        pred_cam_t = output['pred_cam_t'].detach()
        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images')

        os.makedirs(save_dir, exist_ok=True)
        for i, _ in enumerate(pred_vertices):
            if i > 1:
                break
            images_pred = self.renderer.visualize_tb(
                pred_vertices[i:i+1],
                pred_cam_t[i:i+1],
                images[i:i+1],
                sideview=True,
            )

            save_filename = os.path.join(save_dir, f'result_{self.current_epoch:04d}_'
                                                f'{dataloader_nb:02d}_{batch_idx:05d}_{i}.jpg')
            if save_filename is not None:
                images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255
                images_pred = np.clip(images_pred, 0, 255).astype(np.uint8)
                cv2.imwrite(save_filename, cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB))

    def test_step(self, batch, batch_nb, dataloader_nb=0):
        return self.validation_step(batch, batch_nb, dataloader_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        if self.hparams.OPTIMIZER.TYPE == 'sgd':
            return torch.optim.SGD(
                self.model.unet.parameters(), 
                lr=self.hparams.OPTIMIZER.LR, 
                momentum=0.9
            )
        elif self.hparams.OPTIMIZER.TYPE == 'adam8bit':
            import bitsandbytes as bnb
            
            return bnb.optim.AdamW8bit(
                self.model.unet.parameters(),
                lr=self.hparams.OPTIMIZER.LR,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-08,
            )
        else:
            return torch.optim.Adam(
                self.model.unet.parameters(),
                lr=self.hparams.OPTIMIZER.LR,
                weight_decay=self.hparams.OPTIMIZER.WD,
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True
                # eps=1.0,
            )

    def train_dataset(self):
        options = self.hparams.DATASET
        dataset_names = options.DATASETS_AND_RATIOS.split('_')
        dataset_list = [DatasetSDPose(options, ds) for ds in dataset_names]
        train_ds = ConcatDataset(dataset_list)

        return train_ds

    def train_dataloader(self):
        self.train_ds = self.train_dataset()
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
            pin_memory=self.hparams.DATASET.PIN_MEMORY,
            shuffle=self.hparams.DATASET.SHUFFLE_TRAIN,
            drop_last=True
        )

    def val_dataset(self):
        datasets = self.hparams.DATASET.VAL_DS.split('_')
        logger.info(f'Validation datasets are: {datasets}')
        val_datasets = []
        for dataset_name in datasets:
            val_datasets.append(
                DatasetSDPose(
                    options=self.hparams.DATASET,
                    dataset=dataset_name,
                    is_train=False,
                )
            )
        return val_datasets

    def val_dataloader(self):
        dataloaders = []
        for val_ds in self.val_ds:
            dataloaders.append(
                DataLoader(
                    dataset=val_ds,
                    batch_size=self.hparams.DATASET.BATCH_SIZE,
                    shuffle=False,
                    num_workers=self.hparams.DATASET.NUM_WORKERS,
                    drop_last=True
                )
            )
        return dataloaders

    def test_dataloader(self):
        return self.val_dataloader()