import os
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from data.base_dataset import BaseDataset, get_transform

class FivekTestDataset(BaseDataset):
    def __init__(self, opt):
        self.root_dir = opt.dataset_root
        self.size = (opt.load_size,opt.load_size)
        self.transform = self.get_transform()
        self.file_list = self._read_file_list()
        self.loader = default_loader
        self.ref_image_paths = opt.ref_image_paths
        self.ref_image = self.transform(self.loader(self.ref_image_paths))

    def _read_file_list(self):
        file_list = []
        for file_name in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file_name)
            if os.path.isfile(file_path):
                file_list.append(file_path)
        return file_list

    def get_transform(self):
        transform_list = []
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Resize(self.size)]
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        dir_split = file_path.split(os.sep)
        image_path = dir_split[-1]
        image = self.loader(file_path)

        if self.transform:
            image = self.transform(image)

        image_pair = [image, self.ref_image]

        return {'image_pair': image_pair, "image_path":image_path}
