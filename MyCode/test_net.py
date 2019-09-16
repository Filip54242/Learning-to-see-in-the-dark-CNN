import numpy as np
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from torch import optim
from torch.utils.data import DataLoader
import csv

from LSDLoader import LSDLoader
from Plotter import VisdomLinePlotter
from model import *


def PSNRV2(img1, img2):
    target = np.array(img1)
    output = np.array(img2)
    my_psnrv2 = compare_psnr(target, output)
    return my_psnrv2


def SSIM(img1, img2):
    target = np.array(img1)
    output = np.array(img2)
    my_ssim = compare_ssim(target, output, multichannel=True)
    return my_ssim


def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


test_set = LSDLoader("/home/student/Documents/Learning-to-see-in-the-dark-CNN/dataset/test/Sony", crop=False)

test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

psnr_list=[]
ssim_list=[]

network = torch.load("LSD50").cpu()
plot1 = VisdomLinePlotter("SSMI and PSNR_baseline")
plot2 = VisdomLinePlotter("SSMI and PSNR_baseline")
opt = optim.Adam(network.parameters(), lr=1e-4)
network.eval()
for batch_index, (inputs, targets) in enumerate(test_loader):
    input, target = inputs.cpu(), targets.cpu()

    input = np.minimum(input, 1.0)
    target = np.maximum(target, 0.0)

    in_img = input.permute(0, 3, 1, 2)

    network.zero_grad()
    out_img = network(in_img)
    out_img = out_img.permute(0, 2, 3, 1).detach().cpu()
    out_img = np.minimum(np.maximum(out_img, 0), 1)
    output = out_img[0, :, :, :]
    gt = target[0, :, :, :].detach().cpu()

    psnr = PSNRV2(output, gt)
    ssim = SSIM(output, gt)
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    plot1.plot("Batch index", "PSNR", 'PSNR', batch_index, psnr)
    plot2.plot("Batch index", "SSIM", 'SSIM', batch_index, ssim)
    opt.step()

with open('psnr_baseline.csv', mode='w') as file:
    psnr_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    psnr_writer.writerow(psnr_list)
with open('ssim_baseline.csv', mode='w') as file:
    ssim_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    ssim_writer.writerow(ssim_list)
