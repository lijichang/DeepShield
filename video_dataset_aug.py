import multiprocessing
from typing import List

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
import random
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from video_transforms import (GroupRandomHorizontalFlip, GroupOverSample,
                               GroupMultiScaleCrop, GroupScale, GroupCenterCrop, GroupRandomCrop,
                               GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomScale,
                               get_source_transforms, get_transform, randaffine)


def source_augment(img_group): # img_group is a list，has num_clips*duration [h,w,c] np.ndarray
    seed = np.random.randint(2147483647) # To ensure that the num_clips*duration video frames from a video are enhanced with the same data augmentation
    
    source_trans = get_source_transforms()
    
    transformed_images = []
    for img in img_group:
        random.seed(seed)
        np.random.seed(seed)
        transformed = source_trans(image=img)
        transformed_images.append(transformed['image'])

    return transformed_images


def augment(imgs, masks, size=224, is_train=True): # img_group is a list，has num_clips*duration [h,w,c] np.ndarray
    seed = np.random.randint(2147483647) # To ensure that the num_clips*duration video frames from a video are enhanced with the same data augmentation
    
    trans = get_transform(size=size, is_train=is_train)

    transformed_images = []
    transformed_masks = []

    if masks is not None:
        for img, mask in zip(imgs, masks):
            random.seed(seed)
            np.random.seed(seed)
            transformed =trans(image=img, mask=mask)
            transformed_images.append(transformed['image'])
            transformed_masks.append(transformed['mask'])
        
        stacked_images = torch.stack(transformed_images)
        stacked_masks = torch.stack(transformed_masks)
        
        # reshape to [num_clips*duration, c, h, w]
        return stacked_images.view(-1, stacked_images.shape[-3], stacked_images.shape[-2], stacked_images.shape[-1]), stacked_masks.view(-1, stacked_masks.shape[-2], stacked_masks.shape[-1])
    else:
        for img in imgs:
            random.seed(seed)
            np.random.seed(seed)
            transformed =trans(image=img)
            transformed_images.append(transformed['image'])

        stacked_images = torch.stack(transformed_images)
        
        # reshape to [num_clips*duration, c, h, w]
        return stacked_images.view(-1, stacked_images.shape[-3], stacked_images.shape[-2], stacked_images.shape[-1])


def group_randaffine(img_group, mask_group):
    affined_imgs = []
    affined_masks = []
    seed = np.random.randint(2147483647)
    for img, mask in zip(img_group, mask_group):
        affined_img, affined_mask = randaffine(img, mask, seed)
        affined_imgs.append(affined_img)
        affined_masks.append(affined_mask)
    return affined_imgs, affined_masks


def get_augmentor(is_train: bool, image_size: int, mean: List[float] = None,
                  std: List[float] = None, disable_scaleup: bool = False,
                  threed_data: bool = True, version: str = 'v1', scale_range: [int] = None,
                  modality: str = 'rgb', num_clips: int = 1, num_crops: int = 1, cut_out=False,dataset: str = ''):

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    scale_range = [256, 320] if scale_range is None else scale_range


    augments = []
    if is_train:
        if version == 'v1':
            augments += [
                GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
            ]
        elif version == 'v2':
            augments += [
                GroupRandomScale(scale_range),
                GroupRandomCrop(image_size),
            ]
        if not (dataset.startswith('ststv') or 'jester' in dataset or 'mini_ststv' in dataset):
            augments += [GroupRandomHorizontalFlip(is_flow=(modality == 'flow'))]
    else:
        scaled_size = image_size if disable_scaleup else int(image_size / 0.875 + 0.5)
        if num_crops == 1:
            augments += [
                GroupScale(scaled_size),
                GroupCenterCrop(image_size)
            ]
        else:
            flip = True if num_crops == 10 else False
            augments += [
                GroupOverSample(image_size, scaled_size, num_crops=num_crops, flip=flip),
            ]
    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(num_clips_crops=num_clips * num_crops),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]
    # if cut_out:
    #     augments += [GroupCutout(n_holes=1,length=16)] 

    augmentor = transforms.Compose(augments)
    return augmentor


def build_dataflow(dataset, is_train, batch_size, workers=16, is_distributed=False):
    workers = min(workers, multiprocessing.cpu_count())
    shuffle = False

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    if is_train:
        shuffle = sampler is None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn,
                                              num_workers=workers, drop_last = True,pin_memory=True, sampler=sampler)

    return data_loader


def build_dataflow_nosbi(dataset, is_train, batch_size, workers=16, is_distributed=False):
    workers = min(workers, multiprocessing.cpu_count())
    shuffle = False

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    if is_train:
        shuffle = sampler is None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=workers, drop_last = True,pin_memory=True, sampler=sampler)

    return data_loader

