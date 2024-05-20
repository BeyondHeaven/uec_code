import os
import random
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from data.base_dataset import BaseDataset, get_transform

class FivekDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.expos = ['neg3', 'neg2', 'neg1', '0', 'pos1', 'pos2','pos3']
        self.expos_len = len(self.expos)
        self.size = (opt.load_size,opt.load_size)
        self.transform = self.get_transform()
        self.root_dir = opt.dataset_root
        self.file_list = self._read_file_list()
        self.loader = default_loader
        self.length = len(self.file_list)

    def _read_file_list(self):
        file_list = []
        with open(os.path.join(self.root_dir, 'train_input.txt'), 'r') as f:
            for line in f:
                file_name = line.strip()
                file_list.append(file_name)
        return file_list

    def get_transform(self):
        transform_list = []
        transform_list += [transforms.Resize(self.size)]
        transform_list += [transforms.ToTensor()]

        return transforms.Compose(transform_list)


    def __len__(self):
        return len(self.file_list)

    def get_data_by_index(self, idx, shuffle = True):
        file_name = self.file_list[idx]

        # sub_dirs = random.sample(dirs, 2)
        index1 = random.randint(0, self.expos_len - 2)
        index2 = random.randint(index1 + 1, self.expos_len - 1)
        sub_dirs = [self.expos[index1], self.expos[index2]]

        if(shuffle):
            random.shuffle(sub_dirs)

        images = []
        file_paths = []
        for sub_dir in sub_dirs:
            file_path = os.path.join(self.root_dir, sub_dir, file_name)
            image = self.loader(file_path)

            if self.transform:
                image = self.transform(image)

            images.append(image)
            file_paths.append(file_path)

        return images, file_paths

    def __getitem__(self, idx):

        images, file_paths = self.get_data_by_index(idx)
        random_idx = random.choice([i for i in range(self.length) if i != idx])
        images2, _ = self.get_data_by_index(random_idx, False)

        return {'image_pair2': images2,'image_pair': images, "image_path":file_paths}