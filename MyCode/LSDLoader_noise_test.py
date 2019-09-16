import json

import torch


class LSDLoader(torch.utils.data.Dataset):
    def __init__(self, path, num_images=-1):
        self.target_path = path + "/targets/"
        self.input_path = path + "/inputs/"
        self.file_dir = path + "/correspondence.txt"
        self.add_to_output_path = '/media/student/2.0 TB Hard Disk/add_to_output/'
        self.data = []

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
            self.data = self.data[::600 // num_images]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.input_path + self.data[idx][0]
        label_name = self.target_path + self.data[idx][1]
        add_name = self.add_to_output_path + self.data[idx][0]
        image = torch.load(img_name).squeeze()
        target = torch.load(label_name).squeeze()
        add = torch.load(add_name).squeeze()

        return image, target, add
