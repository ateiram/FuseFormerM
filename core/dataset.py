import os
import random
import torch
import numpy as np
from itertools import product
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from core.utils import create_random_shape_with_random_motion
from core.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train'):
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']
        self.size = self.w, self.h = (args['w'], args['h'])

        if args['name'] == 'YouTubeVOS':
            vid_lst_prefix = os.path.join(args['data_root'], args['name'], split+'_all_frames/JPEGImages')
            vid_lst = os.listdir(vid_lst_prefix)
            self.video_names = [os.path.join(vid_lst_prefix, name) for name in vid_lst]

            sem_lst_prefix = os.path.join(args['data_root'], args['name'], split+'_all_frames/semantic_maps')
            sem_lst = os.listdir(sem_lst_prefix)
            self.semantic_map_names = [os.path.join(sem_lst_prefix, name) for name in sem_lst]
        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def create_color_map(self):
        predefined_list = [0, 51, 102, 153, 204, 255]
        initial_list = [value for value in predefined_list]
        tuples = list(product(initial_list, repeat=3))
        tuples.remove((0, 0, 0))
        color_map = np.asarray(tuples[0:133])

        return color_map

    def load_item(self, index):
        video_name = self.video_names[index]
        semantic_map_name = self.semantic_map_names[index]
        all_frames = [os.path.join(video_name, name) for name in sorted(os.listdir(video_name))]
        all_semantic_maps = [os.path.join(semantic_map_name, name) for name in sorted(os.listdir(semantic_map_name))]
        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=self.h, imageWidth=self.w)
        ref_index = get_ref_index(len(all_frames), self.sample_length)
        # read video frames
        frames = []
        frames_res = []
        semantic_maps = []
        semantic_maps_res = []
        masks = []
        masks_resized = []

        color_map = self.create_color_map()

        for idx in ref_index:
            img = Image.open(all_frames[idx]).convert('RGB')
            img = img.resize(self.size)

            frames.append(img)

            img_res = img.resize((self.w//4, self.h//4))
            frames_res.append(img_res)

            sem_img = Image.open(all_semantic_maps[idx]).convert('RGB')
            sem_img = sem_img.resize(self.size, resample=0)
            semantic_maps.append(sem_img)

            sem_img_res = sem_img.resize((self.w//4, self.h//4), resample=0)
            semantic_maps_res.append(sem_img_res)

            # to give to swap module, for not to resize while in a batch
            mask_resized = all_masks[idx].resize((self.w//4, self.h//4), resample=0)  # (108, 60)
            masks_resized.append(mask_resized)
            masks.append(all_masks[idx])
        if self.split == 'train':
            frames, semantic_maps = GroupRandomHorizontalFlip()(frames, semantic_maps)
        # To tensors
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0
        semantic_map_tensors = self._to_tensors(semantic_maps)*2.0 - 1.0

        frame_res_tensors = self._to_tensors(frames_res)
        semantic_map_res_tensors = self._to_tensors(semantic_maps_res)

        mask_tensors = self._to_tensors(masks)
        masks_resized_tensors = self._to_tensors(masks_resized)
        return frame_tensors, mask_tensors, semantic_map_tensors, masks_resized_tensors, frame_res_tensors, semantic_map_res_tensors


def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index
