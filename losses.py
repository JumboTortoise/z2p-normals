import torch
import torch.nn.functional as F
import torchvision

import torchvision.models.vgg as vgg
import numpy as np

"""
Note: the images are in BGR format, not RGB

This means the the alpha(blue) channel is the FIRST channel, not the last
"""

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights=vgg.VGG16_Weights.DEFAULT).features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights=vgg.VGG16_Weights.DEFAULT).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights=vgg.VGG16_Weights.DEFAULT).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights=vgg.VGG16_Weights.DEFAULT).features[16:23].eval())

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input_, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input_.shape[1] != 3:
            input_ = input_.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input_ = (input_-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input_ = self.transform(input_, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input_
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

__preceptual_loss_obj = VGGPerceptualLoss()

__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__laplacian_kernel = torch.tensor(np.array([
        [0,-1,0],
        [-1,4,-1],
        [0,-1,0]
    ])).unsqueeze(0).unsqueeze(0).repeat(2,1,1,1).float().to(__device)

__gaussian_kernel = torch.tensor(np.array([
    [1/16,1/8,1/16],
    [1/8,1/4,1/8],
    [1/16,1/8,1/16]
    ])).unsqueeze(0).unsqueeze(0).repeat(2,1,1,1).float().to(__device)



def masked_laplacan_weighted_mse(generated,gt):
    foreground_mask = (gt[:, 0, :, :] > 0).float()
    gt = gt[:,1:,:,:] # remove blue channel (alpha)
    gt.requires_grad = False

    # apply filters to gt image
    scale = 1
    weights = torch.abs(F.conv2d(gt, __laplacian_kernel, bias=None, stride=1, padding=1, dilation=1, groups=2))*scale
    weights = torch.abs(F.conv2d(weights,__gaussian_kernel,bias=None,stride=1,padding=1,dilation=1,groups=2))

    loss = (generated[:, 1:, :, :] - gt[:,:, :, :]) ** 2
    loss = (loss*weights).sum(dim=1)
    loss *= foreground_mask
    loss = loss.sum() / max(foreground_mask.float().sum() , 1)
    return loss

def preceptual(generated, gt):
   return __preceptual_loss_obj(generated,gt)

def mse(generated, gt):
    loss = (generated[:, 1:, :, :] - gt[:, 1:, :, :]) ** 2
    loss = loss.sum(dim=1)
    loss = loss.sum() / (loss.shape[-1] * loss.shape[-2])
    return loss

def masked_mse(generated, gt):
    foreground_mask = (gt[:, 0, :, :] > 0).float()
    loss = (generated[:, 1:, :, :] - gt[:, 1:, :, :]) ** 2
    loss = loss.sum(dim=1)
    loss *= foreground_mask
    loss = loss.sum() / max(foreground_mask.float().sum() , 1)
    return loss


def masked_cosine(generated, gt, eps=1e-5):
    foreground_mask = (gt[:, 0, :, :] > 0).float()

    dot = (generated[:, 1:, :, :] * gt[:, 1:, :, :]).sum(dim=1)
    norm = generated[:, 1:, :, :].norm(dim=1) * gt[:, 1:, :, :].norm(dim=1)

    zero_mask = norm < eps
    dot *= (~zero_mask).type(dot.dtype)  # make sure gradients don't flow to elements considered zero
    norm[zero_mask] = 1  # avoid division by zero
    cosine_similarity = dot / norm
    loss = 1 - cosine_similarity

    loss *= foreground_mask
    loss = loss.sum() / foreground_mask.float().sum()
    return loss


def intensity(generated, gt):
    intensity_generated = generated[:, 0, :, :]
    intensity_gt = gt[:,0, :, :]
    loss = (intensity_generated - intensity_gt).abs()
    loss = loss.sum() / (loss.shape[-1] * loss.shape[-2])
    return loss


def mse_intensity(generated, gt):
    intensity_generated = generated[:, 0, :, :]
    intensity_gt = gt[:, 0, :, :]
    loss = (intensity_generated - intensity_gt) ** 2
    loss = loss.sum() / (loss.shape[-1] * loss.shape[-2])
    return loss


def pixel_intensity(generated, gt):
    loss = (generated[:, 1:, :, :].norm(dim=1) - gt[:, 1:, :, :].norm(dim=1)).abs()
    loss = loss.sum() / (loss.shape[-1] * loss.shape[-2])
    return loss


def masked_pixel_intensity(generated, gt):
    foreground_mask = (gt[:, 0, :, :] > 0).float()
    loss = (generated[:, 1:, :, :].norm(dim=1) - gt[:, 1:, :, :].norm(dim=1)).abs() * foreground_mask
    loss = loss.sum() / foreground_mask.sum()
    return loss


def background(generated, gt):
    background_mask = (gt == 0)
    loss = generated[background_mask].sum(dim=1).mean()
    return loss
