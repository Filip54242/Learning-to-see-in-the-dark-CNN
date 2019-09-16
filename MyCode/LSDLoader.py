import json
import random

import numpy as np
import torch


class LSDLoader(torch.utils.data.Dataset):
    def __init__(self, path, num_images=-1, crop=True):
        self.target_path = path + "/targets/"
        self.input_path = path + "/inputs/"
        self.file_dir = path + "/correspondence.txt"
        self.data = []
        self.crop = crop

        with open(self.file_dir, 'r') as file:
            contents = file.read().replace("\'", "\"")
            parsed_json = json.loads(contents)

        for key, elements in parsed_json.items():
            if type(elements) is list:
                for element in elements:
                    self.data.append((element, key))
            else:
                self.data.append((elements, key))
        if num_images > 0:
            self.data = random.sample(self.data, num_images)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.input_path + self.data[idx][0]
        label_name = self.target_path + self.data[idx][1]
        image = torch.load(img_name).squeeze()
        target = torch.load(label_name).squeeze()
        if self.crop:
            H = image.shape[0]
            W = image.shape[1]

            xx = np.random.randint(0, W - 512)
            yy = np.random.randint(0, H - 512)
            image = image[yy:yy + 512, xx:xx + 512, :]
            target = target[yy * 2:yy * 2 + 512 * 2, xx * 2:xx * 2 + 512 * 2, :]

        return image, target
