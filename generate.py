'''
Description:
Author: Hejun Jiang
Date: 2020-12-21 14:06:18
LastEditTime: 2020-12-21 14:59:11
LastEditors: Hejun Jiang
Version: v0.0.1
Contact: jianghejun@hccl.ioa.ac.cn
Corporation: hccl
'''
import os
import torch
import model
import numpy as np
from torchvision.utils import save_image


def GetDevice(cuda, dev):
    if cuda:
        device = torch.device("cuda:0")
        print('using gpu', dev)
    else:
        device = torch.device("cpu")
        print('using cpu')
    return device


if __name__ == '__main__':
    modeldir = './model'
    modelname = 'dcgan_generator_epoch2_times8000.0000.pth'
    savedir = './generate'
    gennum = 2
    imgh = 320
    latent_dim = 100
    channels = 1
    os.makedirs(savedir, exist_ok=True)

    dev = '1'
    if dev != '' and dev != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = dev
    cuda = True if torch.cuda.is_available() and dev != 'cpu' else False
    device = GetDevice(cuda, dev)

    generator = model.Generator(imgh, latent_dim, channels)  # h, dim, c
    print(generator)
    pth = torch.load(os.path.join(modeldir, modelname), map_location=device)
    generator.load_state_dict(pth['state_dict'])
    generator.eval()  # 设置为评估模式；如果是resume，则设置为train模式
    print('generator load parameters done')

    for i in range(gennum):
        z = torch.FloatTensor(np.random.normal(0, 1, (1, latent_dim)))  # 输出的值赋在shape里,(batchsize, latent_dim)
        gen_imgs = generator(z)  # [gennum, 1, 320, 320] b, c, h, w
        save_image(gen_imgs.data, os.path.join(savedir, 'generate_{}.png'.format(i)), nrow=1, normalize=True)
        # save_image(gen_imgs.data, os.path.join(savedir, 'generate.png'), nrow=4, normalize=True)
    print('save done')
