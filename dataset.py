'''
Description:
Author: Hejun Jiang
Date: 2020-12-16 07:59:15
LastEditTime: 2020-12-21 14:38:32
LastEditors: Hejun Jiang
Version: v0.0.1
Contact: jianghejun@hccl.ioa.ac.cn
Corporation: hccl
'''
import torch.utils.data as data
import os
import numpy as np
import cv2  # hwc
# import random

imgtype = ['png', 'jpg', 'jpeg']


class GetDataSet(data.Dataset):  # 自己构造Dataset 自己构造子类
    def __init__(self, h, w, channels, imgdir):
        super().__init__()
        self.size_h = h
        self.size_w = w
        self.channels = channels
        self.pathlist = []
        for root, dirs, files in os.walk(imgdir):
            for file in files:
                if file.split('.')[-1] in imgtype:
                    self.pathlist.append(os.path.join(os.path.abspath(root), file))

    def __len__(self):
        return len(self.pathlist)

    def __getitem__(self, index):
        img = cv2.imread(self.pathlist[index], cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        scaleh = self.size_h / h  # 缩放倍数
        scalew = self.size_w / w  # 缩放倍数
        img = cv2.resize(img, (0, 0), fx=scalew, fy=scaleh, interpolation=cv2.INTER_CUBIC)
        if self.channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        img = img.astype(np.float32)
        img = (img/255. - 0.5) / 0.5
        img = img.transpose([2, 0, 1])  # c, h, w
        return img, 1
