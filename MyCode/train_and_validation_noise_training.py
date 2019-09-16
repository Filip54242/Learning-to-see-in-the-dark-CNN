import os

import numpy as np
from torch import optim
from torch.utils.data import DataLoader

import ToImage
from LSDLoader import LSDLoader
from Plotter import VisdomLinePlotter
from model import *


def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


train_set = LSDLoader("/home/student/Documents/Learning-to-see-in-the-dark-CNN/dataset/noise_training/train")

validation_set = LSDLoader("/home/student/Documents/Learning-to-see-in-the-dark-CNN/dataset/noise_training/val")

validation_loader = DataLoader(validation_set, batch_size=1, shuffle=True)

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

network = SeeInDark()
network.cuda()

network._initialize_weights()

learning_rate = 1e-4
minimum_learning_rate = 1e-5

opt = optim.Adam(network.parameters(), lr=learning_rate)

num_epochs = 501
plotter = VisdomLinePlotter("Train_and_Validation_Noise")

train_loss = np.zeros((5000, 1))
valid_loss = np.zeros((5000, 1))
for epoch in range(num_epochs):
    network.train()
    cnt = 0
    for batch_index, (inputs, targets) in enumerate(train_loader):
        input, target = inputs, targets

        input = np.minimum(input, 1.0)
        target = np.maximum(target, 0.0)

        in_img = input.permute(0, 3, 1, 2).cuda()
        gt_img = target.permute(0, 3, 1, 2).cuda()

        network.zero_grad()
        out_img = network(in_img)

        loss = reduce_mean(out_img, gt_img)
        loss.backward()

        opt.step()
        train_loss[batch_index] = loss.item()
    true_train_loss = np.mean(train_loss[np.where(train_loss)])
    if epoch > 0 and epoch % 5 == 0:
        network.eval()
        for batch_index, (inputs, targets) in enumerate(validation_loader):
            input, target = inputs, targets

            input = np.minimum(input, 1.0)
            target = np.maximum(target, 0.0)

            in_img = input.permute(0, 3, 1, 2).cuda()
            gt_img = target.permute(0, 3, 1, 2).cuda()

            network.zero_grad()
            out_img = network(in_img)

            loss = reduce_mean(out_img, gt_img)

            opt.step()
            valid_loss[batch_index] = loss.item()
        true_valid_loss = np.mean(valid_loss[np.where(valid_loss)])
        print("Validation Loss: %.10f " % true_valid_loss)
        plotter.plot('loss', 'eval', 'Train VS Eval loss', epoch, true_valid_loss)

    if epoch > 0 and epoch % 20 == 0:
        torch.save(network, 'LSD%d_noise' % epoch)
        if not os.path.isdir('%04d_noise' % epoch):
            os.makedirs('%04d_noise' % epoch)
        output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
        output = np.minimum(np.maximum(output, 0), 1)

        temp = np.concatenate((targets[0, :, :, :], output[0, :, :, :]), axis=1)
        ToImage.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
            '%04d_noise/00_train%d.jpg' % (epoch, batch_index))

    torch.save(network, 'LSD_noise')
    print('Results after epoch %d' % (epoch + 1))

    print("Training Loss: %.10f " % true_train_loss)
    plotter.plot('loss', 'train', 'Train VS Eval loss', epoch, true_train_loss)
