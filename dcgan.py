'''
Description: 
Author: Hejun Jiang
Date: 2020-12-16 07:39:47
LastEditTime: 2020-12-21 16:11:41
LastEditors: Hejun Jiang
Version: v0.0.1
Contact: jianghejun@hccl.ioa.ac.cn
Corporation: hccl
'''
import os
import model
import torch
import shutil
import dataset
import argparse
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def GetDevice(cuda, opt):
    if cuda:
        device = torch.device("cuda:0")
        print('using gpu', opt.device)
    else:
        device = torch.device("cpu")
        print('using cpu')
    return device


def GetModelAndLoss(device, opt):
    generator = model.Generator(opt.img_size_h, opt.latent_dim, opt.channels)  # generator
    generator.apply(model.weights_init_normal)

    discriminator = model.Discriminator(opt.img_size_h, opt.channels)  # discriminator
    discriminator.apply(model.weights_init_normal)

    adversarial_loss = torch.nn.BCELoss()  # loss

    return generator.to(device), discriminator.to(device), adversarial_loss.to(device)


def GetDataLoader(opt):
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset.GetDataSet(opt.img_size_h, opt.img_size_w, opt.channels, opt.image_dir),
        batch_size=opt.batch_size,
        shuffle=True,
    )
    return dataloader


def GetOptimizers(opt):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    return optimizer_G, optimizer_D


def DcGANTrain(opt, cuda, dataloader, generator, discriminator, adversarial_loss, optimizer_G, optimizer_D):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)  # batchsize, 1
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)  # batchsize, 1
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # Train Generator
            optimizer_G.zero_grad()
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))  # 输出的值赋在shape里,(batchsize, 100)
            # Generate a batch of images
            gen_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            batches_done = epoch * len(dataloader) + i + 1
            if batches_done % opt.save_img_interval == 0:
                save_image(gen_imgs.data, opt.save_img_dir + "/%d.png" % batches_done, nrow=4, normalize=True)
            if batches_done % opt.save_model_interval == 0:
                torch.save(
                    {
                        "generator_model_state_dict": generator.state_dict(),
                        "generator_optimizer_state_dict": optimizer_G.state_dict(),
                        "discriminator_model_state_dict": discriminator.state_dict(),
                        "discriminator_optimizer_state_dict": optimizer_D.state_dict(),
                        "epoch": epoch + 1,
                        "times": batches_done,
                    },  os.path.join(opt.save_model_dir, "dcgan_model_epoch{}_times{:.4f}.pth".format(epoch + 1, batches_done))
                )
                torch.save(
                    {
                        "generator_model_state_dict": generator.state_dict(),
                        "epoch": epoch + 1,
                        "times": batches_done,
                    },  os.path.join(opt.save_model_dir, "dcgan_generator_epoch{}_times{:.4f}.pth".format(epoch + 1, batches_done))
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")  # 64
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")  # 0.0002
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")  # 潜空间维度
    parser.add_argument("--img_size_h", type=int, default=320, help="size of each image dimension")
    parser.add_argument("--img_size_w", type=int, default=320, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels, 1 or 3")
    parser.add_argument("--image_dir", type=str, default='./copy', help="the train path of images")
    parser.add_argument("--save_img_interval", type=int, default=50, help="interval between image sampling")
    parser.add_argument("--save_img_dir", type=str, default='./images_0.0002_savemd', help="the train images save's dir")
    parser.add_argument("--save_model_interval", type=int, default=1000, help="save train model interval")
    parser.add_argument("--save_model_dir", type=str, default='./model', help="the train images save's dir")
    parser.add_argument("--device", type=str, default='3', help="0 or 0,1,2,3 or cpu")
    opt = parser.parse_args()
    assert opt.channels == 1 or opt.channels == 3, 'image channels must 1 or 3'
    print(opt)

    if opt.device != '' and opt.device != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    cuda = True if torch.cuda.is_available() and opt.device != 'cpu' else False

    if os.path.exists(opt.save_img_dir):
        shutil.rmtree(opt.save_img_dir)
    os.makedirs(opt.save_img_dir)
    os.makedirs(opt.save_model_dir, exist_ok=True)

    generator, discriminator, adversarial_loss = GetModelAndLoss(GetDevice(cuda, opt), opt)
    dataloader = GetDataLoader(opt)
    optimizer_G, optimizer_D = GetOptimizers(opt)
    DcGANTrain(opt, cuda, dataloader, generator, discriminator, adversarial_loss, optimizer_G, optimizer_D)
