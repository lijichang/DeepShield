import os
import six
from typing import Union
import random
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import cv2
from video_dataset_aug import source_augment, augment, group_randaffine
from sbi.sbi import group_dynamic_blend
from sbi.bi_online_generation import random_get_hull

try:
    import lmdb
    import pyarrow as pa
    _HAS_LMDB = True
except ImportError as e:
    _HAS_LMDB = False
    _LMDB_ERROR_MSG = e

try:
    import av
    _HAS_PYAV = True
except ImportError as e:
    _HAS_PYAV = False
    _PYAV_ERROR_MSG = e


def random_clip(video_frames, sampling_rate, frames_per_clip, fixed_offset=False, start_frame_idx=0, end_frame_idx=None):
    """

    Args:
        video_frames (int): total frame number of a video
        sampling_rate (int): sampling rate for clip, pick one every k frames
        frames_per_clip (int): number of frames of a clip
        fixed_offset (bool): used with sample offset to decide the offset value deterministically.

    Returns:
        list[int]: frame indices (started from zero)
    """
    new_sampling_rate = sampling_rate
    highest_idx = video_frames - new_sampling_rate * frames_per_clip if end_frame_idx is None else end_frame_idx
    if highest_idx <= 0:
        random_offset = 0
    else:
        if fixed_offset:
            random_offset = (video_frames - new_sampling_rate * frames_per_clip) // 2
        else:
            random_offset = int(np.random.randint(start_frame_idx, highest_idx, 1))
    # print(start_frame_idx, highest_idx, random_offset)
    frame_idx = [int(random_offset + i * sampling_rate) % video_frames for i in range(frames_per_clip)]
    return frame_idx


def compute_img_diff(image_1, image_2, bound=255.0):
    image_diff = np.asarray(image_1, dtype=np.float) - np.asarray(image_2, dtype=np.float)
    image_diff += bound
    image_diff *= (255.0 / float(2 * bound))
    image_diff = image_diff.astype(np.uint8)
    image_diff = Image.fromarray(image_diff)
    return image_diff


def load_image(root_path, directory, image_tmpl, idx, modality):
    """

    :param root_path:
    :param directory:
    :param image_tmpl:
    :param idx: if it is a list, load a batch of images
    :param modality:
    :return:
    """

    def _safe_load_image(img_path):
        img = None
        num_try = 0
        while num_try < 10:
            try:
                img_tmp = Image.open(img_path)
                img = img_tmp.copy()
                img = np.array(img)
                img_tmp.close()
                break
            except Exception as e:
                print('[Will try load again] error loading image: {}, '
                      'error: {}'.format(img_path, str(e)))
                num_try += 1
        if img is None:
            raise ValueError('[Fail 10 times] error loading image: {}'.format(img_path))
        return img

    if not isinstance(idx, list):
        idx = [idx]
    out = []
    if modality == 'rgb':
        for i in idx:
            image_path_file = os.path.join(root_path, directory, image_tmpl.format(i))
            out.append(_safe_load_image(image_path_file))
    elif modality == 'rgbdiff':
        tmp = {}
        new_idx = np.unique(np.concatenate((np.asarray(idx), np.asarray(idx) + 1)))
        for i in new_idx:
            image_path_file = os.path.join(root_path, directory, image_tmpl.format(i))
            tmp[i] = _safe_load_image(image_path_file)
        for k in idx:
            img_ = compute_img_diff(tmp[k + 1], tmp[k])
            out.append(img_)
        del tmp
    elif modality == 'flow':
        for i in idx:
            flow_x_name = os.path.join(root_path, directory, "x_" + image_tmpl.format(i))
            flow_y_name = os.path.join(root_path, directory, "y_" + image_tmpl.format(i))
            out.extend([_safe_load_image(flow_x_name), _safe_load_image(flow_y_name)])

    return out

# -----------------------------------add for distort-------------------------------------------
def load_distort_image(root_path, directory, image_tmpl, idx, modality, distort_func, distort_param):
    """

    :param root_path:
    :param directory:
    :param image_tmpl:
    :param idx: if it is a list, load a batch of images
    :param modality:
    :return:
    """

    def _safe_load_image(img_path):
        img = None
        num_try = 0
        while num_try < 10:
            try:
                img_tmp = Image.open(img_path)
                img = img_tmp.copy()
                img = np.array(img)
                img_tmp.close()
                # RGB to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = distort_func(img, distort_param)
                # BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                break
            except Exception as e:
                print('[Will try load again] error loading image: {}, '
                      'error: {}'.format(img_path, str(e)))
                num_try += 1
        if img is None:
            raise ValueError('[Fail 10 times] error loading image: {}'.format(img_path))
        return img

    if not isinstance(idx, list):
        idx = [idx]
    out = []
    if modality == 'rgb':
        for i in idx:
            image_path_file = os.path.join(root_path, directory, image_tmpl.format(i))
            out.append(_safe_load_image(image_path_file))
    elif modality == 'rgbdiff':
        tmp = {}
        new_idx = np.unique(np.concatenate((np.asarray(idx), np.asarray(idx) + 1)))
        for i in new_idx:
            image_path_file = os.path.join(root_path, directory, image_tmpl.format(i))
            tmp[i] = _safe_load_image(image_path_file)
        for k in idx:
            img_ = compute_img_diff(tmp[k + 1], tmp[k])
            out.append(img_)
        del tmp
    elif modality == 'flow':
        for i in idx:
            flow_x_name = os.path.join(root_path, directory, "x_" + image_tmpl.format(i))
            flow_y_name = os.path.join(root_path, directory, "y_" + image_tmpl.format(i))
            out.extend([_safe_load_image(flow_x_name), _safe_load_image(flow_y_name)])

    return out
# -----------------------------------add for distort-------------------------------------------

# -----------------------------------add for self blend-----------------------------------------
def load_image_lands(root_path, directory, image_tmpl, idx, modality):
    """

    :param root_path:
    :param directory:
    :param image_tmpl:
    :param idx: if it is a list, load a batch of images
    :param modality:
    :return:
    """

    def _safe_load_image(img_path):
        img = None
        num_try = 0
        while num_try < 10:
            try:
                img_tmp = Image.open(img_path)
                img = img_tmp.copy()
                img = np.array(img)
                img_tmp.close()
                break
            except Exception as e:
                print('[Will try load again] error loading image: {}, '
                      'error: {}'.format(img_path, str(e)))
                num_try += 1
        if img is None:
            raise ValueError('[Fail 10 times] error loading image: {}'.format(img_path))
        return img

    if not isinstance(idx, list):
        idx = [idx]
    out = []
    lands = []
    if modality == 'rgb':
        for i in idx:
            image_path_file = os.path.join(root_path, directory, image_tmpl.format(i))
            out.append(_safe_load_image(image_path_file))
            lands_path_file = image_path_file.replace('frames', 'landmarks').replace('png', 'npy')
            lands.append(np.load(lands_path_file))

    return out, lands
# -----------------------------------add for self blend-----------------------------------------


def sample_train_clip(video_length, num_consecutive_frames, num_frames, sample_freq, dense_sampling, num_clips=1):
    max_frame_idx = max(1, video_length - num_consecutive_frames + 1)
    if dense_sampling:
        frame_idx = np.zeros((num_clips, num_frames), dtype=int)
        if num_clips == 1:  # backward compatibility
            frame_idx[0] = np.asarray(random_clip(max_frame_idx, sample_freq, num_frames, False))
        else:
            max_start_frame_idx = max_frame_idx - sample_freq * num_frames
            frames_per_segment = max_start_frame_idx // num_clips
            for i in range(num_clips):
                if frames_per_segment <= 0:
                    frame_idx[i] = np.asarray(random_clip(max_frame_idx, sample_freq, num_frames, False))
                    #frame_idx[i] = [frame_idx[i][2],frame_idx[i][0],frame_idx[i][1],frame_idx[i][3]]
                    #frame_idx[i] = [frame_idx[i][3],frame_idx[i][2],frame_idx[i][1],frame_idx[i][0]]
                else:
                    frame_idx[i] = np.asarray(random_clip(max_frame_idx, sample_freq, num_frames, False, i * frames_per_segment, (i + 1) * frames_per_segment))
                    #1423
                    #frame_idx[i] = [frame_idx[i][2],frame_idx[i][0],frame_idx[i][1],frame_idx[i][3]]
                    #frame_idx[i] = [frame_idx[i][3],frame_idx[i][2],frame_idx[i][1],frame_idx[i][0]]
        frame_idx = frame_idx.flatten()
        """
        def _check_interval_overlapped(int_1, int_2):
            if int_1[0] < int_2[0]:
                int_l, int_r = int_1, int_2
            else:
                int_l, int_r = int_2, int_1

            return True if int_l[-1] > int_r[0] else False

        clips = 0
        num_tries = 0
        #all_frame_idx = np.arange(max_frame_idx - sample_freq * num_frames)
        while clips < num_clips and num_tries < 1000:
            curr_clips = np.asarray(random_clip(max_frame_idx, sample_freq, num_frames))
            overlap = False
            for i in range(clips):
                overlap = _check_interval_overlapped((frame_idx[i][0], frame_idx[i][-1]), (curr_clips[0], curr_clips[-1]) ) 
                if overlap:
                    break
            if overlap:
                num_tries += 1
                continue
            else:
                frame_idx[clips] = curr_clips
                clips += 1
        for i in range(clips, num_clips):
            frame_idx[i] = np.asarray(random_clip(max_frame_idx, sample_freq, num_frames))

        # sort the intervals
        frame_idx = frame_idx[np.argsort(frame_idx[:, 0]), ...]
        frame_idx = frame_idx.flatten()
        """

    else:  # uniform sampling
        # import pdb;pdb.set_trace()
        total_frames = num_frames * sample_freq
        ave_frames_per_group = max_frame_idx // num_frames
        if ave_frames_per_group >= sample_freq:
            # randomly sample f images per segement
            frame_idx = np.arange(0, num_frames) * ave_frames_per_group
            frame_idx = np.repeat(frame_idx, repeats=sample_freq)
            offsets = np.random.choice(ave_frames_per_group, sample_freq, replace=False)
            offsets = np.tile(offsets, num_frames)
            frame_idx = frame_idx + offsets
        elif max_frame_idx < total_frames:
            # need to sample the same images
            frame_idx = np.random.choice(max_frame_idx, total_frames)
        else:
            # sample cross all images
            frame_idx = np.random.choice(max_frame_idx, total_frames, replace=False)
        frame_idx = np.sort(frame_idx)
    # print(frame_idx)
    frame_idx = frame_idx + 1
    # random.shuffle(frame_idx)
    return frame_idx


def sample_val_test_clip(video_length, num_consecutive_frames, num_frames, sample_freq, dense_sampling,
                         fixed_offset, num_clips, whole_video):
    max_frame_idx = max(1, video_length - num_consecutive_frames + 1)
    # import pdb;pdb.set_trace()
    if whole_video:
        return np.arange(1, max_frame_idx, step=sample_freq, dtype=int)
    if dense_sampling:
        if fixed_offset:
            sample_pos = max(1, 1 + max_frame_idx - sample_freq * num_frames)
            t_stride = sample_freq
            start_list = np.linspace(0, sample_pos - 1, num=num_clips, dtype=int)
            frame_idx = []
            for start_idx in start_list.tolist():
                frame_idx += [(idx * t_stride + start_idx) % max_frame_idx for idx in
                              range(num_frames)]
        else:
            frame_idx = []
            for i in range(num_clips):
                frame_idx.extend(random_clip(max_frame_idx, sample_freq, num_frames))
        frame_idx = np.asarray(frame_idx) + 1
    else:  # uniform sampling
        if fixed_offset:
            frame_idices = []
            sample_offsets = list(range(-num_clips // 2 + 1, num_clips // 2 + 1))
            for sample_offset in sample_offsets:
                if max_frame_idx > num_frames:
                    tick = max_frame_idx / float(num_frames)
                    curr_sample_offset = sample_offset
                    if curr_sample_offset >= tick / 2.0:
                        curr_sample_offset = tick / 2.0 - 1e-4
                    elif curr_sample_offset < -tick / 2.0:
                        curr_sample_offset = -tick / 2.0
                    frame_idx = np.array([int(tick / 2.0 + curr_sample_offset + tick * x) for x in
                                          range(num_frames)])
                else:
                    np.random.seed(sample_offset - (-num_clips // 2 + 1))
                    frame_idx = np.random.choice(max_frame_idx, num_frames)
                frame_idx = np.sort(frame_idx)
                frame_idices.extend(frame_idx.tolist())
        else:
            frame_idices = []
            for i in range(num_clips):
                total_frames = num_frames * sample_freq
                ave_frames_per_group = max_frame_idx // num_frames
                if ave_frames_per_group >= sample_freq:
                    # randomly sample f images per segment
                    frame_idx = np.arange(0, num_frames) * ave_frames_per_group
                    frame_idx = np.repeat(frame_idx, repeats=sample_freq)
                    offsets = np.random.choice(ave_frames_per_group, sample_freq,
                                               replace=False)
                    offsets = np.tile(offsets, num_frames)
                    frame_idx = frame_idx + offsets
                elif max_frame_idx < total_frames:
                    # need to sample the same images
                    np.random.seed(i)
                    frame_idx = np.random.choice(max_frame_idx, total_frames)
                else:
                    # sample cross all images
                    np.random.seed(i)
                    frame_idx = np.random.choice(max_frame_idx, total_frames, replace=False)
                frame_idx = np.sort(frame_idx)
                frame_idices.extend(frame_idx.tolist())
        frame_idx = np.asarray(frame_idices) + 1
    return frame_idx


class VideoRecord(object):
    def __init__(self, path, start_frame, end_frame, label, reverse=False):
        self.path = path
        self.video_id = os.path.basename(path)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.label = label
        self.reverse = reverse

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1

    def __str__(self):
        return self.path


class VideoDataSet(data.Dataset):

    def __init__(self, root_path, list_file, num_groups=64, frames_per_group=1, sample_offset=0, num_clips=1,
                 modality='rgb', image_size=224, dense_sampling=True, fixed_offset=True,
                 image_tmpl='{:05d}.jpg', transform=None, is_train=True, test_mode=False, seperator=' ',
                 filter_video=0, num_classes=None, whole_video=False,
                 fps=29.97, audio_length=1.28, resampling_rate=24000):
        """

        Arguments have different meaning when dense_sampling is True:
            - num_groups ==> number of frames
            - frames_per_group ==> sample every K frame
            - sample_offset ==> number of clips used in validation or test mode

        Args:
            root_path (str): the file path to the root of video folder
            list_file (str): the file list, each line with folder_path, start_frame, end_frame, label_id
            num_groups (int): number of frames per data sample
            frames_per_group (int): number of frames within one group
            sample_offset (int): used in validation/test, the offset when sampling frames from a group
            modality (str): rgb or flow
            dense_sampling (bool): dense sampling in I3D
            fixed_offset (bool): used for generating the same videos used in TSM
            image_tmpl (str): template of image ids
            transform: the transformer for preprocessing
            is_train (bool): shuffle the video but keep the causality
            test_mode (bool): testing mode, no label
            whole_video (bool): take whole video
            fps (float): frame rate per second, used to localize sound when frame idx is selected.
            audio_length (float): the time window to extract audio feature.
            resampling_rate (int): used to resampling audio extracted from wav
        """
        if modality not in ['flow', 'rgb', 'rgbdiff', 'sound']:
            raise ValueError("modality should be 'flow' or 'rgb' or 'rgbdiff' or 'sound'.")

        self.root_path = root_path
        self.list_file = os.path.join(root_path, list_file)
        self.num_groups = num_groups
        self.num_frames = num_groups
        self.frames_per_group = frames_per_group
        self.sample_freq = frames_per_group
        self.num_clips = num_clips
        self.sample_offset = sample_offset
        self.fixed_offset = fixed_offset
        self.dense_sampling = dense_sampling
        self.modality = modality.lower()
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.is_train = is_train
        self.test_mode = test_mode
        self.separator = seperator
        self.filter_video = filter_video
        self.whole_video = whole_video
        self.fps = fps
        self.audio_length = audio_length
        self.resampling_rate = resampling_rate
        self.video_length = (self.num_frames * self.sample_freq) / self.fps
        self.image_size = image_size

        if self.modality in ['flow', 'rgbdiff']:
            self.num_consecutive_frames = 5
        else:
            self.num_consecutive_frames = 1

        # self.video_list, self.multi_label = self._parse_list()
        self.real_video_list, self.fake_video_list, self.multi_label = self._parse_list()
        self.real_len = len(self.real_video_list)
        self.fake_len = len(self.fake_video_list)
        self.num_classes = num_classes

    def _parse_list(self):
        # usually it is [video_id, num_frames, class_idx]
        # or [video_id, start_frame, end_frame, list of class_idx]
        tmp = []
        original_video_numbers = 0
        for x in open(self.list_file):
            elements = x.strip().split(self.separator)
            start_frame = int(elements[1])
            end_frame = int(elements[2])
            total_frame = end_frame - start_frame + 1
            original_video_numbers += 1
            if self.test_mode:
                tmp.append(elements)
            else:
                if total_frame >= self.filter_video:
                    tmp.append(elements)

        num = len(tmp)
        print("The number of videos is {} (with more than {} frames) "
              "(original: {})".format(num, self.filter_video, original_video_numbers), flush=True)
        assert (num > 0)
        # TODO: a better way to check if multi-label or not
        multi_label = np.mean(np.asarray([len(x) for x in tmp])) > 4.0
        file_list = []
        real_file_list = []
        fake_file_list = []
        for item in tmp:
            if self.test_mode:
                file_list.append([item[0], int(item[1]), int(item[2]), -1])
            else:
                labels = []
                for i in range(3, len(item)):
                    labels.append(float(item[i]))
                if not multi_label:
                    labels = labels[0] if len(labels) == 1 else labels
                # file_list.append([item[0], int(item[1]), int(item[2]), labels])
                if labels == 0:
                    real_file_list.append([item[0], int(item[1]), int(item[2]), labels])
                else:
                    fake_file_list.append([item[0], int(item[1]), int(item[2]), labels])

        # video_list = [VideoRecord(item[0], item[1], item[2], item[3]) for item in file_list]
        real_video_list = [VideoRecord(item[0], item[1], item[2], item[3]) for item in real_file_list]
        fake_video_list = [VideoRecord(item[0], item[1], item[2], item[3]) for item in fake_file_list]
        # flow model has one frame less
        # if self.modality in ['rgbdiff']:
        #     for i in range(len(video_list)):
        #         video_list[i].end_frame -= 1

        #if self.is_train:
        #    video_list = video_list[:50000]

        return real_video_list, fake_video_list, multi_label

    def remove_data(self, idx):
        original_video_num = len(self.video_list)
        self.video_list = [v for i, v in enumerate(self.video_list) if i not in idx]
        print("Original videos: {}\t remove {} videos, remaining {} videos".format(original_video_num, len(idx), len(self.video_list)))

    def _sample_indices(self, record):
        return sample_train_clip(record.num_frames, self.num_consecutive_frames, self.num_frames,
                                 self.sample_freq, self.dense_sampling, self.num_clips)

    def _get_val_indices(self, record):
        return sample_val_test_clip(record.num_frames, self.num_consecutive_frames, self.num_frames,
                                    self.sample_freq, self.dense_sampling, self.fixed_offset,
                                    self.num_clips, self.whole_video)

    def __getitem__(self, index):
        """
        Returns:
            torch.FloatTensor: (3xgxf)xHxW dimension, g is number of groups and f is the frames per group.
            torch.FloatTensor: the label
        """
        if index < self.real_len:
            record = self.real_video_list[index]
        else:
            record = self.fake_video_list[index - self.real_len]
        # check this is a legit video folder
        indices = self._sample_indices(record) if self.is_train else self._get_val_indices(record)
        # images = self.get_data(record, indices)
        # images = self.transform(images)
        label = self.get_label(record)
        
        # # re-order data to targeted format.
        # return images, label
        
        # --------------------------add for self blend--------------------------------------------
        if label == 0: #real
            images, landmarks = self.get_data_lands(record, indices)
            real_masks = self.create_mask_list(images)
            real_has_masks = torch.tensor([1] * len(images))
            landmarks = self.reorder_landmark(landmarks)
            fake_images, fake_masks = self.self_blend(images.copy(), landmarks)
            fake_has_masks = torch.tensor([1] * len(fake_images))
        else:
            fake_images = self.get_data(record, indices)
            fake_masks = self.create_mask_list(fake_images)
            fake_has_masks = torch.tensor([0] * len(fake_images))
            real_index = random.randint(0, self.real_len-1)
            real_record = self.real_video_list[real_index]
            real_indices = self._sample_indices(real_record) if self.is_train else self._get_val_indices(real_record)
            images = self.get_data(real_record, real_indices)
            real_masks = self.create_mask_list(images)
            real_has_masks = torch.tensor([1] * len(images))
        images, real_masks = augment(images, real_masks, size=self.image_size, is_train=self.is_train)
        fake_images, fake_masks = augment(fake_images, fake_masks, size=self.image_size, is_train=self.is_train)
        real_masks = self.create_patch_mask(real_masks)
        fake_masks = self.create_patch_mask(fake_masks)
        return images, fake_images, real_masks, fake_masks, real_has_masks, fake_has_masks
        # ----------------------------------------------------------------------------------------
    
    def create_mask_list(self, images):
        mask_list = []
        for img in images:
            mask = np.zeros(img.shape[0:2], dtype=np.float32)
            mask_list.append(mask)
        return mask_list
    
    def create_patch_mask(self, mask, patch_size=16, threshold=10):
        nt, h, w = mask.shape
        
        patches = mask.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        
        patches = patches.contiguous().view(nt, h // patch_size, w // patch_size, -1)
        
        patch_sums = patches.sum(dim=-1)
        
        new_mask = (patch_sums > threshold).long() #[numclips*duration,14,14]
        
        return new_mask
    
    def reorder_landmark(self,landmarks):
        for landmark in landmarks:
            landmark_add=np.zeros((13,2))
            for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
                landmark_add[idx]=landmark[idx_l]
            landmark[68:]=landmark_add
        return landmarks

    def self_blend(self, images, lands):
        if np.random.rand() < 0.25:
            lands = [landmark[:68] for landmark in lands]
        # hull_type = random.choice([0,1,2,3,4])
        hull_type = random.choice([0,1,2,3])
        masks = [random_get_hull(landmark, img, hull_type)[:,:,0] for img, landmark in zip(images, lands)]
        # masks = [np.zeros_like(img[:,:,0]) for img in images]
        # for mask, landmark in zip(masks, lands):
        #     cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)
        
        source = [img.copy() for img in images]
        target = [img.copy() for img in images]
        if np.random.rand() < 0.5:
            source = source_augment(source)
        else:
            target = source_augment(target)
        source, masks = group_randaffine(source, masks)
        images_blended = group_dynamic_blend(source, target, masks)
        
        # images = [image.astype(np.uint8) for image in images]
        images_blended = [image_blended.astype(np.uint8) for image_blended in images_blended]
        
        return images_blended, masks
    
    def get_data_lands(self, record, indices):
        images = []
        if self.whole_video:
            tmp = len(indices) % self.num_frames
            if tmp != 0:
                indices = indices[:-tmp]
            num_clips = len(indices) // self.num_frames
            # print(tmp, indices, self.num_frames, num_clips)
        else:
            num_clips = self.num_clips
        if self.modality == 'sound':
            new_indices = [indices[i * self.num_frames: (i + 1) * self.num_frames]
                           for i in range(num_clips)]
            for curr_indiecs in new_indices:
                center_idx = (curr_indiecs[self.num_frames // 2 - 1] + curr_indiecs[self.num_frames // 2]) // 2 \
                    if self.num_frames % 2 == 0 else curr_indiecs[self.num_frames // 2]
                center_idx = min(record.num_frames, center_idx)
                # seg_imgs = load_sound(self.root_path, record, center_idx,
                #                       self.fps, self.audio_length, self.resampling_rate)
                # images.extend(seg_imgs)
        else:
            images = []
            landmarks = []
            for seg_ind in indices:
                new_seg_ind = [min(seg_ind + record.start_frame - 1 + i, record.num_frames)
                               for i in range(self.num_consecutive_frames)]
                # seg_imgs = load_image(self.root_path, record.path, self.image_tmpl,
                #                       new_seg_ind, self.modality)
                seg_imgs, seg_lands = load_image_lands(self.root_path, record.path, self.image_tmpl,
                                      new_seg_ind, self.modality)
                images.extend(seg_imgs) #is a list，has "num_clips*duration" [h,w,c] np.ndarray
                landmarks.extend(seg_lands)
        return images, landmarks
    
    def get_data(self, record, indices):
        images = []
        if self.whole_video:
            tmp = len(indices) % self.num_frames
            if tmp != 0:
                indices = indices[:-tmp]
            num_clips = len(indices) // self.num_frames
            # print(tmp, indices, self.num_frames, num_clips)
        else:
            num_clips = self.num_clips
        if self.modality == 'sound':
            new_indices = [indices[i * self.num_frames: (i + 1) * self.num_frames]
                           for i in range(num_clips)]
            for curr_indiecs in new_indices:
                center_idx = (curr_indiecs[self.num_frames // 2 - 1] + curr_indiecs[self.num_frames // 2]) // 2 \
                    if self.num_frames % 2 == 0 else curr_indiecs[self.num_frames // 2]
                center_idx = min(record.num_frames, center_idx)
                # seg_imgs = load_sound(self.root_path, record, center_idx,
                #                       self.fps, self.audio_length, self.resampling_rate)
                # images.extend(seg_imgs)
        else:
            images = []
            for seg_ind in indices:
                new_seg_ind = [min(seg_ind + record.start_frame - 1 + i, record.num_frames)
                               for i in range(self.num_consecutive_frames)]
                seg_imgs = load_image(self.root_path, record.path, self.image_tmpl,
                                      new_seg_ind, self.modality)
                images.extend(seg_imgs) #is a list，has "num_clips*duration" [h,w,c] np.ndarray
        return images
    
    def get_label(self, record):
        if self.test_mode:
            # in test mode, return the video id as label
            label = record.video_id
        else:
            if not self.multi_label:
                label = int(record.label)
            else:
                # create a binary vector.
                label = torch.zeros(self.num_classes, dtype=torch.float)
                for x in record.label:
                    label[int(x)] = 1.0
        return label

    def __len__(self):
        # return len(self.video_list)
        return len(self.real_video_list) + len(self.fake_video_list)
    
    # --------------------------add for self blend--------------------------------------------
    def collate_fn(self,batch):
        imgs_r, imgs_f, real_masks, fake_masks, real_has_masks, fake_has_masks = zip(*batch)
        imgs_r = torch.stack(imgs_r)
        imgs_f = torch.stack(imgs_f)
        real_masks = torch.stack(real_masks)
        fake_masks = torch.stack(fake_masks)
        real_has_masks = torch.stack(real_has_masks)
        fake_has_masks = torch.stack(fake_has_masks)
        data = {}
        # data['img']=torch.cat([torch.tensor(imgs_r).float(),torch.tensor(imgs_f).float()],0)
        data['imgs'] = torch.cat([imgs_r, imgs_f], 0)
        data['masks'] = torch.cat([real_masks, fake_masks], 0)
        data['labels'] = torch.tensor([0] * len(imgs_r) + [1] * len(imgs_f))
        data['has_masks'] = torch.cat([real_has_masks, fake_has_masks], 0)
        return data
    # ----------------------------------------------------------------------------------------
    

class VideoDataSet_nosbi(data.Dataset):

    def __init__(self, root_path, list_file, num_groups=64, frames_per_group=1, sample_offset=0, num_clips=1,
                 modality='rgb', image_size=224, dense_sampling=True, fixed_offset=True,
                 image_tmpl='{:05d}.jpg', transform=None, is_train=True, test_mode=False, seperator=' ',
                 filter_video=0, num_classes=None, whole_video=False,
                 fps=29.97, audio_length=1.28, resampling_rate=24000,
                 distort_func=None, distort_param=None):
        """

        Arguments have different meaning when dense_sampling is True:
            - num_groups ==> number of frames
            - frames_per_group ==> sample every K frame
            - sample_offset ==> number of clips used in validation or test mode

        Args:
            root_path (str): the file path to the root of video folder
            list_file (str): the file list, each line with folder_path, start_frame, end_frame, label_id
            num_groups (int): number of frames per data sample
            frames_per_group (int): number of frames within one group
            sample_offset (int): used in validation/test, the offset when sampling frames from a group
            modality (str): rgb or flow
            dense_sampling (bool): dense sampling in I3D
            fixed_offset (bool): used for generating the same videos used in TSM
            image_tmpl (str): template of image ids
            transform: the transformer for preprocessing
            is_train (bool): shuffle the video but keep the causality
            test_mode (bool): testing mode, no label
            whole_video (bool): take whole video
            fps (float): frame rate per second, used to localize sound when frame idx is selected.
            audio_length (float): the time window to extract audio feature.
            resampling_rate (int): used to resampling audio extracted from wav
        """
        if modality not in ['flow', 'rgb', 'rgbdiff', 'sound']:
            raise ValueError("modality should be 'flow' or 'rgb' or 'rgbdiff' or 'sound'.")

        self.root_path = root_path
        self.list_file = os.path.join(root_path, list_file)
        self.num_groups = num_groups
        self.num_frames = num_groups
        self.frames_per_group = frames_per_group
        self.sample_freq = frames_per_group
        self.num_clips = num_clips
        self.sample_offset = sample_offset
        self.fixed_offset = fixed_offset
        self.dense_sampling = dense_sampling
        self.modality = modality.lower()
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.is_train = is_train
        self.test_mode = test_mode
        self.separator = seperator
        self.filter_video = filter_video
        self.whole_video = whole_video
        self.fps = fps
        self.audio_length = audio_length
        self.resampling_rate = resampling_rate
        self.video_length = (self.num_frames * self.sample_freq) / self.fps
        self.image_size = image_size

        if self.modality in ['flow', 'rgbdiff']:
            self.num_consecutive_frames = 5
        else:
            self.num_consecutive_frames = 1

        self.video_list, self.multi_label = self._parse_list()
        self.num_classes = num_classes
        
        self.distort_func = distort_func
        self.distort_param = distort_param

    def _parse_list(self):
        # usually it is [video_id, num_frames, class_idx]
        # or [video_id, start_frame, end_frame, list of class_idx]
        tmp = []
        original_video_numbers = 0
        for x in open(self.list_file):
            elements = x.strip().split(self.separator)
            start_frame = int(elements[1])
            end_frame = int(elements[2])
            total_frame = end_frame - start_frame + 1
            original_video_numbers += 1
            if self.test_mode:
                tmp.append(elements)
            else:
                if total_frame >= self.filter_video:
                    tmp.append(elements)

        num = len(tmp)
        print("The number of videos is {} (with more than {} frames) "
              "(original: {})".format(num, self.filter_video, original_video_numbers), flush=True)
        assert (num > 0)
        # TODO: a better way to check if multi-label or not
        multi_label = np.mean(np.asarray([len(x) for x in tmp])) > 4.0
        file_list = []
        for item in tmp:
            if self.test_mode:
                file_list.append([item[0], int(item[1]), int(item[2]), -1])
            else:
                labels = []
                for i in range(3, len(item)):
                    labels.append(float(item[i]))
                if not multi_label:
                    labels = labels[0] if len(labels) == 1 else labels
                file_list.append([item[0], int(item[1]), int(item[2]), labels])

        video_list = [VideoRecord(item[0], item[1], item[2], item[3]) for item in file_list]
        # flow model has one frame less
        if self.modality in ['rgbdiff']:
            for i in range(len(video_list)):
                video_list[i].end_frame -= 1

        #if self.is_train:
        #    video_list = video_list[:50000]

        return video_list, multi_label

    def remove_data(self, idx):
        original_video_num = len(self.video_list)
        self.video_list = [v for i, v in enumerate(self.video_list) if i not in idx]
        print("Original videos: {}\t remove {} videos, remaining {} videos".format(original_video_num, len(idx), len(self.video_list)))

    def _sample_indices(self, record):
        return sample_train_clip(record.num_frames, self.num_consecutive_frames, self.num_frames,
                                 self.sample_freq, self.dense_sampling, self.num_clips)

    def _get_val_indices(self, record):
        return sample_val_test_clip(record.num_frames, self.num_consecutive_frames, self.num_frames,
                                    self.sample_freq, self.dense_sampling, self.fixed_offset,
                                    self.num_clips, self.whole_video)

    def __getitem__(self, index):
        """
        Returns:
            torch.FloatTensor: (3xgxf)xHxW dimension, g is number of groups and f is the frames per group.
            torch.FloatTensor: the label
        """
        record = self.video_list[index]
        # check this is a legit video folder
        indices = self._sample_indices(record) if self.is_train else self._get_val_indices(record)
        # images = self.get_data(record, indices)
        # images = self.transform(images)
        # label = self.get_label(record)
        
        # # re-order data to targeted format.
        # return images, label
        
        images = self.get_data(record, indices)
        images = augment(images, None, size=self.image_size, is_train=self.is_train)
        label = self.get_label(record)
        return images, label
    
    def get_data(self, record, indices):
        images = []
        if self.whole_video:
            tmp = len(indices) % self.num_frames
            if tmp != 0:
                indices = indices[:-tmp]
            num_clips = len(indices) // self.num_frames
            # print(tmp, indices, self.num_frames, num_clips)
        else:
            num_clips = self.num_clips
        if self.modality == 'sound':
            new_indices = [indices[i * self.num_frames: (i + 1) * self.num_frames]
                           for i in range(num_clips)]
            for curr_indiecs in new_indices:
                center_idx = (curr_indiecs[self.num_frames // 2 - 1] + curr_indiecs[self.num_frames // 2]) // 2 \
                    if self.num_frames % 2 == 0 else curr_indiecs[self.num_frames // 2]
                center_idx = min(record.num_frames, center_idx)
                # seg_imgs = load_sound(self.root_path, record, center_idx,
                #                       self.fps, self.audio_length, self.resampling_rate)
                # images.extend(seg_imgs)
        else:
            images = []
            for seg_ind in indices:
                new_seg_ind = [min(seg_ind + record.start_frame - 1 + i, record.num_frames)
                               for i in range(self.num_consecutive_frames)]
                if self.distort_func:
                    seg_imgs = load_distort_image(self.root_path, record.path, self.image_tmpl,
                                      new_seg_ind, self.modality, self.distort_func, self.distort_param)
                else:
                    seg_imgs = load_image(self.root_path, record.path, self.image_tmpl,
                                        new_seg_ind, self.modality)
                images.extend(seg_imgs)
        return images

    def get_label(self, record):
        if self.test_mode:
            # in test mode, return the video id as label
            label = record.video_id
        else:
            if not self.multi_label:
                label = int(record.label)
            else:
                # create a binary vector.
                label = torch.zeros(self.num_classes, dtype=torch.float)
                for x in record.label:
                    label[int(x)] = 1.0
        return label
    
    def __len__(self):
        return len(self.video_list)