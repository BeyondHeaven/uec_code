import os
import random
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from data.base_dataset import BaseDataset, get_transform

class ExposureDataset(BaseDataset):
    def __init__(self, opt):
        self.dataset_root = opt.dataset_root
        self.groups = self.get_image_groups()
        self.size = (448, 448)
        self.transform = self.get_transform()

    def get_image_groups(self):
        image_files = os.listdir(self.dataset_root)
        groups = {}

        for file_name in image_files:
            parts = file_name.split("_")
            prefix = "_".join(parts[:-2]) + "_"
            suffix = "_".join(parts[-2:])
            value = parts[-2]
            group_key = f"{prefix}"

            if group_key not in groups:
                groups[group_key] = []

            groups[group_key].append(file_name)

        return groups

    def get_transform(self):
        transform_list = []
        transform_list += [transforms.Resize(self.size)]
        transform_list += [transforms.ToTensor()]

        return transforms.Compose(transform_list)

    def __getitem__(self, index):
        group_key = list(self.groups.keys())[index]
        file_names = self.groups[group_key]

        images = []

        for _ in range(2):
            file_name = random.choice(file_names)
            image_path = os.path.join(self.dataset_root, file_name)
            image = self.transform(default_loader(image_path))
            images.append(image)

        return {'image_pair': images, "image_path":file_name}

    def __len__(self):
        return len(self.groups)