import numpy as np
from torch import optim
from torch.utils.data import DataLoader

import ToImage
from LSDLoader_noise_test import LSDLoader
from model import *

test_set = LSDLoader("/home/student/Documents/Learning-to-see-in-the-dark-CNN/dataset/test/Sony", num_images=10)

test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
index = 1
network = torch.load("LSD50_noise").cpu()
opt = optim.Adam(network.parameters(), lr=1e-4)
network.eval()
for batch_index, (inputs, targets, add) in enumerate(test_loader):
    input, target, add = inputs.cpu(), targets.cpu(), add.cpu()

    input = np.minimum(input, 1.0)
    add = np.minimum(add, 1.0)
    target = np.maximum(target, 0.0)

    in_img = input.permute(0, 3, 1, 2)

    network.zero_grad()
    out_img = network(in_img)
    out_img = out_img.permute(0, 2, 3, 1).detach().cpu()
    out_img = np.minimum(np.maximum(out_img, 0), 1)
    #output = out_img[0, :, :, :]
    output = (add + out_img)[0, :, :, :]
    add = add[0, :, :, :]
    gt = target[0, :, :, :].detach().cpu()

    opt.step()

    temp = np.concatenate((add, output, gt), axis=1)
    ToImage.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
        "/home/student/Documents/Learning-to-see-in-the-dark-CNN/RESULTS/photos/noise/Image_%d.png" % index)
    index += 1
