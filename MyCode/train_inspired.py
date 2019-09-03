import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from LSDLoader import LSDLoader
from model import *
import os
import ToImage


def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


train_set = LSDLoader("/home/student/Documents/LSD/dataset/train/Sony")

train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

network = SeeInDark()
network.cuda()

network._initialize_weights()

opt = optim.Adam(network.parameters(), lr=1e-4)

num_epochs = 100

for epoch in range(num_epochs):
    network.train()
    cnt = 0
    train_loss = 0
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
        train_loss += loss.item()
        if epoch > 0 and epoch % 10 == 0:
            torch.save(network, 'LSD%d'% epoch)
            if not os.path.isdir('%04d' % epoch):
                os.makedirs('%04d' % epoch)
            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output, 0), 1)

            temp = np.concatenate((targets[0, :, :, :], output[0, :, :, :]), axis=1)
            ToImage.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save('%04d/00_train%d.jpg' % (epoch, batch_index))
        torch.save(network, 'LSD')
    print('Results after epoch %d' % (epoch + 1))

    print("Training Loss: %.10f " % (train_loss / (batch_index + 1)))
