import os
from torch.utils.data import Dataset
import cv2
import numpy as np

import hyperparameters
from utils.util import read_im, random_sample_patch


class XRayDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.loaded_data = {}  # dictionary to load data once and not repeat load
        self.crop_size = hyperparameters.crop_size
        self.bit_depth = hyperparameters.bit_depth

        # find all file pairs
        high_paths = []
        low_paths = []
        for filename in os.listdir(path):
            if "high" in filename:
                high_paths.append(os.path.join(path, filename))
            if "low" in filename:
                low_paths.append(os.path.join(path, filename))

        # remember to sort the lists to have correspondings paired together
        high_paths = sorted(high_paths)
        low_paths = sorted(low_paths)

        self.data = []
        for low, high in zip(low_paths, high_paths):  # just a check
            if low.split("low")[0] == high.split("high")[0]:
                self.data.append((low, high))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        low_paths, high_paths = self.data[index]
        if (low_paths, high_paths) not in self.loaded_data:
            low_im = read_im(low_paths, bit_depth=self.bit_depth)
            high_im = read_im(high_paths, bit_depth=self.bit_depth)

            low_crop, high_crop = random_sample_patch(
                low_im,
                high_im,
                self.crop_size,
                threshold=hyperparameters.sample_threshold,
            )
            self.loaded_data[(low_paths, high_paths)] = low_crop, high_crop
        else:
            low_crop, high_crop = self.loaded_data[(low_paths, high_paths)]
        return low_crop, high_crop
