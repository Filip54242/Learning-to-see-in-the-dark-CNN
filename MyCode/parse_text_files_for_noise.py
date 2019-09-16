import numpy as np
import rawpy
import torch


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def parse(path):
    file = open(path, 'r')
    contents = file.read().split("\n")
    contents.pop()
    return [element.split(' ')[:-2] for element in contents]


def get_name(string):
    start = string.find('short/')
    start = start + 6 if start != -1 else string.find('long/') + 5
    end = string.find('.ARW')
    return string[start:end]


def extract_names(item_list):
    result = []
    for element in item_list:
        first, second = element
        result.append([get_name(first), get_name(second)])
    return result


def get_exposure(file_name):
    file_name = file_name.split("_")
    file_name = file_name[2][:-1]
    return float(file_name)


def make_dictionary(item_list):
    dictionary = {}
    for element in item_list:
        if element[1] in dictionary:
            dictionary[element[1]].append(element[0])
        else:
            dictionary[element[1]] = [element[0]]
    return dictionary


def save_second_input(input_path):
    with rawpy.imread(input_path) as raw:
        input_rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        input_rgb = np.expand_dims(np.float32(input_rgb / 65535.0), axis=0)
        torch.save(torch.from_numpy(input_rgb), "/media/student/2.0 TB Hard Disk/add_to_output/" + get_name(input_path))


def process_tensors(input_path, target_path, destination):
    gt_exposure = get_exposure(get_name(target_path))
    in_exposure = get_exposure(get_name(input_path))
    ratio = min(gt_exposure / in_exposure, 300)

    with rawpy.imread(input_path) as raw:
        input_rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        input_rgb = np.expand_dims(np.float32(input_rgb / 65535.0), axis=0)
        input_packed = np.expand_dims(pack_raw(raw), axis=0) * ratio
        input_tensor = torch.from_numpy(input_packed)
        torch.save(input_tensor, destination + "/inputs/" + get_name(input_path))

    with rawpy.imread(target_path) as raw:
        target_rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        target_rgb = np.expand_dims(np.float32(target_rgb / 65535.0), axis=0)
        target = target_rgb - input_rgb
        tensor = torch.from_numpy(target)
        torch.save(tensor, destination + "/targets/" + get_name(input_path))

    return get_name(input_path)


def transform_file_list(txt_file_path, destitation_path):
    files = parse(txt_file_path)

    result = []
    for element in files:
        element = [element[0].replace(".", "../dataset", 1), element[1].replace(".", "../dataset", 1)]
        result.append(element)

    dictionary = make_dictionary(result)
    result = {}
    for element in dictionary.items():
        for sub_element in element[1]:
            #save_second_input(sub_element)
            first = process_tensors(sub_element, element[0], destitation_path)
            result[first] = first
            print(first)

    with open(destitation_path + "/correspondence.txt", 'w') as f:
        f.write(str(result))


transform_file_list("/home/student/Documents/Learning-to-see-in-the-dark-CNN/dataset/Sony_train_list.txt",
                    "/home/student/Documents/Learning-to-see-in-the-dark-CNN/dataset/noise_training/train")
transform_file_list("/home/student/Documents/Learning-to-see-in-the-dark-CNN/dataset/Sony_val_list.txt",
                    "/home/student/Documents/Learning-to-see-in-the-dark-CNN/dataset/noise_training/val")
