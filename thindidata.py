import os
import os.path
import pickle
import cv2
import numpy as np
import torchvision
from torchvision.datasets import CIFAR10
import torch.utils.data as torchdata
import torchvision.transforms as transforms


class Cifar10Data(torchdata.Dataset):
    base_folder = "cifar-10-batches-py"
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
    std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

    def __init__(self, root, datasource="train"):
        self.root = os.path.expanduser(root)
        self.datasource = datasource if datasource in {"train", "test", "all"
                                                       } else "train"
        # train: [len: 50000, shape: (N, H, W, C)]; test: [len: 10000, shape: (N, H, W, C)]
        self.images = list()
        self.labels = list()
        self.__buildset__()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.get_image(index), self.get_label(index)

    def __buildset__(self):
        train_images, test_images, train_labels, test_labels = (list(), list(),
                                                                list(), list())
        if self.datasource in {"train", "all"}:
            for fentry in self.train_list:
                fname = fentry[0]
                fpath = os.path.join(self.root, self.base_folder, fname)
                with open(fpath, "rb") as f:
                    entry = pickle.load(f, encoding="latin1")
                    train_images.append(entry["data"])
                    if "labels" in entry:
                        train_labels.extend(entry["labels"])
                    else:
                        train_labels.extend(entry["fine_labels"])
        if self.datasource in {"test", "all"}:
            fname = self.test_list[0][0]
            fpath = os.path.join(self.root, self.base_folder, fname)
            with open(fpath, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                test_images.append(entry["data"])
                if "labels" in entry:
                    test_labels.extend(entry["labels"])
                else:
                    test_labels.extend(entry["fine_labels"])
        for data in [train_images, test_images]:
            if len(data) != 0:
                self.images.extend(data)
        for data in [train_labels, test_labels]:
            if len(data) != 0:
                self.labels.extend(data)
        self.images = np.concatenate(self.images)
        self.images = self.images.reshape((-1, 3, 32, 32))
        self.images = self.images.transpose((0, 2, 3, 1))

    def get_image(self, index):
        image = self.images[index]
        image_tf = transforms.Compose([
            # transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        return image_tf(image)

    def get_label(self, index):
        return self.labels[index]


class ThindiCifar10Data(Cifar10Data):
    def __init__(self, root, datasource="train"):
        super().__init__(root, datasource=datasource)

    def __getitem__(self, index):
        return (self.get_image(index), self.get_sketch(index),
                self.get_label(index))

    def get_sketch(self, index):
        edge_image = self.sketch(index)[:, :, None]
        edge_tf = transforms.ToTensor()
        return edge_tf(edge_image)

    def sketch(self, index):
        image = self.images[index]  # HWC
        gray_image = cv2.blur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (3, 3))
        high_th = otsu(gray_image)
        edge_image = cv2.Canny(
            gray_image, 0.5 * high_th, high_th, L2gradient=True)
        return edge_image


def get_dataloader(dataset,
                   root,
                   datasource,
                   batch_size,
                   shuffle=True,
                   num_workers=2,
                   pin_memory=True):
    return torchdata.DataLoader(
        dataset(root, datasource),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory)


def otsu(gray_image):
    otsu_ret_val, otsu_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return otsu_ret_val

# def otsu(gray_img):
#     hist = cv2.calcHist([gray_img], [0], None, [256], [0.0, 256.0])
#     hist = hist.reshape(256)
#     p_hist = hist / np.sum(hist)
#     gray_lvl = np.arange(256)
#     sigma_b = np.zeros(256)
#     for t in gray_lvl:
#         q_L = sum(p_hist[:t + 1])
#         q_H = sum(p_hist[t + 1:])
#         if not (q_L == 0 or q_H == 0):
#             miu_L = p_hist[:t + 1].dot(gray_lvl[:t + 1]) / q_L
#             miu_H = p_hist[t + 1:].dot(gray_lvl[t + 1:]) / q_H
#             sigma_b[t] = q_L * q_H * (miu_L - miu_H)**2
#     th_otsu = np.argmax(sigma_b)
#     return th_otsu
