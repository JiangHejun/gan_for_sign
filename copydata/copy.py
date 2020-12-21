'''
Description: copy images for gan which can have enough images to train
Author: Hejun Jiang
Date: 2020-12-16 10:20:17
LastEditTime: 2020-12-21 15:01:48
LastEditors: Hejun Jiang
Version: v0.0.1
Contact: jianghejun@hccl.ioa.ac.cn
Corporation: hccl
'''
import os
import cv2
import shutil
import random

if __name__ == '__main__':
    path = './signature'
    cpath = '../copy'
    num = 64000

    plist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            plist.append(cv2.imread(os.path.join(root, file)))

    if os.path.exists(cpath):
        shutil.rmtree(cpath)
    os.makedirs(cpath)

    for i in range(num):
        img = random.choice(plist)
        cv2.imwrite(os.path.join(cpath, str(i) + '.png'), img)
        if (i + 1) % 1000 == 0:
            print('write done', i+1)
    print('write all done')
