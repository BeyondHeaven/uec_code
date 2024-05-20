import os
import random
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from data.base_dataset import BaseDataset, get_transform
import torchsnooper

class ExposureDataset(BaseDataset):
    def __init__(self, opt):
        self.dataset_root = opt.dataset_root
        self.expos = ["N1.5.JPG","N1.JPG","0.JPG","P1.JPG","P1.5.JPG"]
        self.expos_len = len(self.expos)
        self.groups = self.get_image_groups()
        self.length = len(self.groups)
        self.size = (448, 448)
        self.transform = self.get_transform()


    def sort_filenames_by_suffix(self, file_names):
        """
        Sorts a list of file names based on a specified order of suffixes.

        :param file_names: List of file names to be sorted.
        :param suffix_order: List of suffixes in the desired order.
        :return: Sorted list of file names.
        """
        # 将后缀顺序映射到排序索引上
        sort_order = {suffix: index for index, suffix in enumerate(self.expos)}

        # 提取后缀的函数
        def extract_suffix(file_name):
            parts = file_name.split('_')
            return parts[-1] if len(parts) > 1 else ''

        # 根据后缀排序
        return sorted(file_names, key=lambda x: sort_order.get(extract_suffix(x), float('inf')))

    # @torchsnooper.snoop()
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

        for key, file_names in groups.items():
            groups[key] = self.sort_filenames_by_suffix(file_names)

        return groups

    def get_transform(self):
        transform_list = []
        transform_list += [transforms.Resize(self.size)]
        transform_list += [transforms.ToTensor()]

        return transforms.Compose(transform_list)

    def get_data_by_index(self, index, shuffle = True):
        group_key = list(self.groups.keys())[index]
        file_names = self.groups[group_key]

        index1 = random.randint(0, self.expos_len - 2)
        index2 = random.randint(index1 + 1, self.expos_len - 1)

        image_path1 = os.path.join(self.dataset_root,  file_names[index1])
        image_path2 = os.path.join(self.dataset_root,  file_names[index2])
        image1 = self.transform(default_loader(image_path1))
        image2 = self.transform(default_loader(image_path2))

        images = [image1, image2]
        if(shuffle):
            random.shuffle(images)
        file_name = [os.path.basename(image_path1), os.path.basename(image_path2)]

        return images, file_name

    def __getitem__(self, index):
        images, file_paths = self.get_data_by_index(index)
        random_idx = random.choice([i for i in range(self.length) if i != index])
        images2, _ = self.get_data_by_index(random_idx, False)

        return {'image_pair2': images2,'image_pair': images, "image_path":file_paths}

    def __len__(self):
        return len(self.groups)