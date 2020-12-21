'''
Description: 
Author: Hejun Jiang
Date: 2020-12-15 17:45:12
LastEditTime: 2020-12-21 15:28:48
LastEditors: Hejun Jiang
Version: v0.0.1
Contact: jianghejun@hccl.ioa.ac.cn
Corporation: hccl
'''
import imageio
import os


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def take(elem):
    return elem[1]


if __name__ == '__main__':
    imgdir = './images'
    savedir = './show'
    duration = 0.35
    os.makedirs(savedir, exist_ok=True)

    images = []
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            images.append((os.path.join(root, file), int(file.split('.')[0])))
    images.sort(key=take)

    image_list = []
    for image in images:
        image_list.append(image[0])
    create_gif(image_list, os.path.join(savedir, 'train.gif'), duration)
