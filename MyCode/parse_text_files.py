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


def save_as_target(file_path, destination):
    with rawpy.imread(file_path) as raw:
        image = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        image = np.expand_dims(np.float32(image / 65535.0), axis=0)
        tensor = torch.from_numpy(image)
        destination += "/targets/" + get_name(file_path)
        torch.save(tensor, destination)
        print("parsed to " + destination)


def save_as_input(file_path, destination, ratio):
    with rawpy.imread(file_path) as raw:
        input = np.expand_dims(pack_raw(raw), axis=0) * ratio
        tensor = torch.from_numpy(input)
        destination += "/inputs/" + get_name(file_path)
        torch.save(tensor, destination)
        print("parsed to " + destination)


def remove_duplicates(item_list):
    return list(set([item for sublist in item_list for item in sublist]))


def transform_file_list(txt_file_path, destitation_path):
    files = parse(txt_file_path)

    with open(destitation_path + "/correspondence.txt", 'w') as f:
        f.write(str(make_dictionary(extract_names(files))))

    result = []
    for element in files:
        element = [element[0].replace(".", "../dataset", 1), element[1].replace(".", "../dataset", 1)]
        result.append(element)

    dictionary = make_dictionary(result)
    for element in dictionary.items():
        save_as_target(element[0], destitation_path)
        for sub_element in element[1]:
            gt_exposure = get_exposure(get_name(element[0]))
            in_exposure = get_exposure(get_name(sub_element))
            ratio = min(gt_exposure / in_exposure, 300)
            save_as_input(sub_element, destitation_path, ratio)


#transform_file_list("/home/student/Documents/LSD/dataset/Fuji_train_list.txt",
 #                   "/home/student/Documents/LSD/dataset/train/Fuji")
# transform_file_list("/home/student/Documents/LSD/dataset/Fuji_test_list.txt",
#                     "/home/student/Documents/LSD/dataset/test/Fuji")
# transform_file_list("/home/student/Documents/LSD/dataset/Fuji_val_list.txt",
#                     "/home/student/Documents/LSD/dataset/val/Fuji")
#
transform_file_list("/home/student/Documents/LSD/dataset/Sony_train_list.txt",
                     "/home/student/Documents/LSD/dataset/train/Sony")
# transform_file_list("/home/student/Documents/LSD/dataset/Sony_test_list.txt",
#                     "/home/student/Documents/LSD/dataset/test/Sony")
# transform_file_list("/home/student/Documents/LSD/dataset/Sony_val_list.txt",
#                     "/home/student/Documents/LSD/dataset/val/Sony")
