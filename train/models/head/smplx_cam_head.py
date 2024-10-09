import torch
import torch.nn as nn

from .smplx_local import SMPLX
from ...core import config
from ...utils.geometry import estimate_translation_fullimg


class SMPLXCamHead(nn.Module):
    def __init__(self, img_res=224):

        super(SMPLXCamHead, self).__init__()
        self.smplx = SMPLX(config.SMPLX_MODEL_DIR, num_betas=11)
        self.add_module('smplx', self.smplx)
        self.img_res = img_res

    def forward(self, rotmat, shape, cam, cam_intrinsics,
                bbox_scale, bbox_center, img_w, img_h,
                normalize_joints2d=False, gt_joints2d=None, gt_cam=None, trans=False, trans2=False,
                learned_scale=None):

        smpl_output = self.smplx(
            betas=shape,
            body_pose=rotmat[:, 1:22].contiguous(),
            global_orient=rotmat[:, 0].unsqueeze(1).contiguous(),
            pose2rot=False,
        ) 

        output = {
            'vertices': smpl_output.vertices,
            'joints3d': smpl_output.joints,
        }

        joints3d = output['joints3d']
        batch_size = joints3d.shape[0]
        device = joints3d.device

        # if gt_joints2d is not None:
        #     import ipdb; ipdb.set_trace()
        #     with torch.no_grad():
        #         est_cam = estimate_translation_fullimg(
        #             S=joints3d[:, :24].cpu(),
        #             joints_2d=gt_joints2d.cpu(),
        #             focal_length=torch.stack([cam_intrinsics[:, 0, 0], cam_intrinsics[:, 1, 1]], dim=-1).cpu(),
        #             img_size=torch.stack([img_w, img_h], dim=-1).cpu(),
        #             use_all_joints=True,
        #         )
        # else:
        #     est_cam = cam
        
        cam_t = convert_pare_to_full_img_cam(
            pare_cam=gt_cam, # cam
            bbox_height=bbox_scale * 200.,
            bbox_center=bbox_center,
            img_w=img_w,
            img_h=img_h,
            focal_length=cam_intrinsics[:, 0, 0],
            crop_res=self.img_res,
        )
        
        joints2d = perspective_projection(
            joints3d,
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=cam_t,
            cam_intrinsics=cam_intrinsics,
        )
        if normalize_joints2d:
            # Normalize keypoints to [-1,1]
            joints2d = joints2d / (self.img_res / 2.)
        
        output['joints2d'] = joints2d
        output['pred_cam_t'] = cam_t

        return output


def perspective_projection(points, rotation, translation, cam_intrinsics):

    K = cam_intrinsics
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points.float())
    return projected_points[:, :, :-1]


def convert_pare_to_full_img_cam(
        pare_cam, bbox_height, bbox_center,
        img_w, img_h, focal_length, crop_res=224):

    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    res = 224
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)

    cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

    return cam_t


def convert_full_img_cam_t_to_weak_cam(
        cam_t, bbox_height, bbox_center,
        img_w, img_h, focal_length, crop_res=224):

    res = crop_res
    r = bbox_height / res
    
    s = 2 * focal_length / (r * res * cam_t[:, 2])
    
    # cam_t = [tx + cx, ty + cy, tz]
    cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)
    tx = cam_t[:, 0] - cx
    ty = cam_t[:, 1] - cy

    cam = torch.stack([s, tx, ty], dim=-1)
    return cam