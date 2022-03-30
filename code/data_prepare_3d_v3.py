# data_prepare.py
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from preprocess import my_PreProc, adjust_gamma
import nibabel as nib
import torch


def nii_to_np(niifile):
    # 开始读取nii文件
    img_np = nib.load(niifile).get_data()  # 读取nii
    img_np = np.array(img_np)
    return img_np


def binary_mask2bbox(binary_mask):
    assert isinstance(binary_mask, np.ndarray) or isinstance(binary_mask, torch.Tensor)
    if binary_mask.ndim == 5:
        assert binary_mask.shape[1] == 1
        if isinstance(binary_mask, torch.Tensor):
            binary_mask = torch.squeeze(binary_mask, 1)
        else:
            binary_mask = np.squeeze(binary_mask, 1)
    if binary_mask.ndim == 4:
        if isinstance(binary_mask, torch.Tensor):
            return torch.stack([binary_mask2bbox(binary_mask[i] for i in range(len(binary_mask)))])
        else:
            return np.stack([binary_mask2bbox(binary_mask[i] for i in range(len(binary_mask)))])
    assert binary_mask.ndim == 3
    if isinstance(binary_mask, torch.Tensor):
        device = binary_mask.device
        binary_mask = binary_mask.detach().cpu().numpy()
    else:
        device = None
    d_mask = (binary_mask.sum((0, 1)) > 0).astype(np.float32)
    d_start = int(np.argmax(d_mask))
    d_end = int(d_start + d_mask.sum() - 1)
    h_mask = (binary_mask.sum((1, 2)) > 0).astype(np.float32)
    h_start = int(np.argmax(h_mask))
    h_end = int(h_start + h_mask.sum() - 1)
    w_mask = (binary_mask.sum((0, 2)) > 0).astype(np.float32)
    w_start = int(np.argmax(w_mask))
    w_end = int(w_start + w_mask.sum() - 1)
    bbox = np.array([d_start, d_end, h_start, h_end, w_start, w_end], np.int32)
    if device is not None:
        bbox = torch.from_numpy(bbox).to(device)
    return bbox


def get_imgs_and_masks_path(root_path, dataset, datatpye='training'):
    """Return all the couples (img, mask)"""
    path = os.path.join(root_path, dataset, datatpye)

    path_img = os.path.join(path, 'images')
    path_label1 = os.path.join(path, 'masks/bladder')
    path_label2 = os.path.join(path, 'masks/colon')
    path_label3 = os.path.join(path, 'masks/intestine')
    path_label4 = os.path.join(path, 'masks/rectum')

    img_name = os.listdir(path_img)
    imgs_path = [os.path.join(path_img, i) for i in img_name]
    masks_path1 = [os.path.join(path_label1, i) for i in img_name]
    masks_path2 = [os.path.join(path_label2, i) for i in img_name]
    masks_path3 = [os.path.join(path_label3, i) for i in img_name]
    masks_path4 = [os.path.join(path_label4, i) for i in img_name]
    return list(zip(imgs_path, masks_path1, masks_path2, masks_path3, masks_path4))


def get_imgs_and_masks_patch(img_data, patch_size, slice_patch_size, train=True, patch_num=1):
    """Return all the couples (img, mask)"""
    data = []
    if train == True:
        for img, mask in img_data:
            # position
            box_pos = binary_mask2bbox(mask)
            # print(box_pos)
            start_s = box_pos[0] - slice_patch_size + 1
            end_s = box_pos[1] - 1
            start_h = box_pos[2] - patch_size + 1
            end_h = box_pos[3] - 1
            start_w = box_pos[4] - patch_size + 1
            end_w = box_pos[5] - 1
            # print(start_s, end_s, start_h, end_h, start_w, end_w)

            start_h = max(start_h, int(patch_size / 2))
            end_h = min(end_h, mask.shape[0] - (patch_size - int(patch_size / 2)))
            start_w = max(start_w, int(patch_size / 2))
            end_w = min(end_w, mask.shape[1] - (patch_size - int(patch_size / 2)))
            start_s = max(start_s, int(slice_patch_size / 2))
            end_s = min(end_s, mask.shape[2] - (slice_patch_size - int(slice_patch_size / 2)))
            print(start_s, end_s, start_h, end_h, start_w, end_w)

            seed_h = np.random.randint(start_h, end_h + 1, patch_num)
            seed_w = np.random.randint(start_w, end_w + 1, patch_num)
            seed_s = np.random.randint(start_s, end_s + 1, patch_num)

            # img and mask
            data_ = [[img[seed_h[i] - int(patch_size / 2):seed_h[i] + patch_size - int(patch_size / 2),
                      seed_w[i] - int(patch_size / 2):seed_w[i] + patch_size - int(patch_size / 2),
                      seed_s[i] - int(slice_patch_size / 2):seed_s[i] + slice_patch_size - int(slice_patch_size / 2)],
                      mask[seed_h[i] - int(patch_size / 2):seed_h[i] + patch_size - int(patch_size / 2),
                      seed_w[i] - int(patch_size / 2):seed_w[i] + patch_size - int(patch_size / 2),
                      seed_s[i] - int(slice_patch_size / 2):seed_s[i] + slice_patch_size - int(slice_patch_size / 2)]]
                     for i in range(patch_num)]
            data += data_
            random.shuffle(data)
    else:
        for img, mask in img_data:
            # h and w and s (position list)
            h = list(range(0, img.shape[0], patch_size))
            w = list(range(0, img.shape[1], patch_size))
            s = list(range(0, img.shape[2], slice_patch_size))

            # if over the edge, get the patch beside the edge
            if h[-1] + patch_size > img.shape[0]:
                h[-1] = img.shape[0] - patch_size
            if w[-1] + patch_size > img.shape[1]:
                w[-1] = img.shape[1] - patch_size
            if s[-1] + slice_patch_size > img.shape[2]:
                s[-1] = img.shape[2] - slice_patch_size

            # img and mask and [postion,num]
            for i in range(len(h)):
                for j in range(len(w)):
                    for k in range(len(s)):
                        data.append([img[h[i]:h[i] + patch_size, w[j]:w[j] + patch_size, s[k]:s[k] + slice_patch_size],
                                     mask[h[i]:h[i] + patch_size, w[j]:w[j] + patch_size, s[k]:s[k] + slice_patch_size],
                                     [h[i], w[j], s[k]]])
    return data


def get_patch_for_one_img(img, mask, patch_size, slice_patch_size, train=True, patch_num=20):
    data = []
    if train == True:
        # print("img shape", np.shape(img))
        # print("mask shape", np.shape([mask[0], mask[1], mask[2], mask[3]]))
        # position
        box_pos1 = binary_mask2bbox(mask[0])
        box_pos2 = binary_mask2bbox(mask[1])
        box_pos3 = binary_mask2bbox(mask[2])
        box_pos4 = binary_mask2bbox(mask[3])
        start_s = min(box_pos1[0], box_pos2[0], box_pos3[0], box_pos4[0]) - slice_patch_size + 1
        end_s = max(box_pos1[1], box_pos2[1], box_pos3[1], box_pos4[1]) - 1
        start_h = min(box_pos1[2], box_pos2[2], box_pos3[2], box_pos4[2]) - patch_size + 1
        end_h = max(box_pos1[3], box_pos2[3], box_pos3[3], box_pos4[3]) - 1
        start_w = min(box_pos1[4], box_pos2[4], box_pos3[4], box_pos4[4]) - patch_size + 1
        end_w = max(box_pos1[5], box_pos2[5], box_pos3[5], box_pos4[5]) - 1
        # print(start_s, end_s + 1 + slice_patch_size, start_h, end_h + 1 + patch_size, start_w, end_w + 1 + patch_size)

        start_h = max(start_h, 0)
        end_h = min(end_h, mask[0].shape[0] - patch_size - 1)
        start_w = max(start_w, 0)
        end_w = min(end_w, mask[0].shape[1] - patch_size - 1)
        start_s = max(start_s, 0)
        end_s = min(end_s, mask[0].shape[2] - slice_patch_size - 1)
        # print("shape", mask[0].shape[0], mask[0].shape[1], mask[0].shape[2])
        # print(start_s, end_s + 1 + slice_patch_size, start_h, end_h + 1 + patch_size, start_w, end_w + 1 + patch_size)

        # seed是起始点
        seed_h = np.random.randint(start_h, end_h + 2, patch_num)
        seed_w = np.random.randint(start_w, end_w + 2, patch_num)
        seed_s = np.random.randint(start_s, end_s + 2, patch_num)

        # print(np.shape(img), np.shape(mask))
        img = np.array(img)
        mask = np.array(mask)
        # img and mask
        data_ = [[img[:, seed_h[i]:seed_h[i] + patch_size,
                  seed_w[i]:seed_w[i] + patch_size,
                  seed_s[i]:seed_s[i] + slice_patch_size],
                  mask[:, seed_h[i]:seed_h[i] + patch_size,
                  seed_w[i]:seed_w[i] + patch_size,
                  seed_s[i]:seed_s[i] + slice_patch_size]]
                 for i in range(patch_num)]
        data += data_
        random.shuffle(data)
    return data


def get_patch_for_test(img, mask, patch_size, slice_patch_size):
    data = []
    h = list(range(0, img[0].shape[0], patch_size))
    w = list(range(0, img[0].shape[1], patch_size))
    s = list(range(0, img[0].shape[2], slice_patch_size))

    # if over the edge, get the patch beside the edge
    if h[-1] + patch_size > img[0].shape[0]:
        h[-1] = img[0].shape[0] - patch_size
    if w[-1] + patch_size > img[0].shape[1]:
        w[-1] = img[0].shape[1] - patch_size
    if s[-1] + slice_patch_size > img[0].shape[2]:
        s[-1] = img[0].shape[2] - slice_patch_size

    # img and mask and [postion,num]
    img = np.array(img)
    mask = np.array(mask)
    for i in range(len(h)):
        for j in range(len(w)):
            for k in range(len(s)):
                data.append([img[:, h[i]:h[i] + patch_size, w[j]:w[j] + patch_size, s[k]:s[k] + slice_patch_size],
                             mask[:, h[i]:h[i] + patch_size, w[j]:w[j] + patch_size, s[k]:s[k] + slice_patch_size],
                             [h[i], w[j], s[k]]])
    return data


def batch(iterable, batch_size, patch_size, slice_patch_size, train=True):  # iterable 改为路径列表
    """Yields lists by batch"""
    j = 0
    b = []
    for i, data_list in enumerate(iterable):  # 返回i为序号，data_list为iterable具体内容
        img_path = data_list[0]
        # print(img_path)
        mask_path1 = data_list[1]
        mask_path2 = data_list[2]
        mask_path3 = data_list[3]
        mask_path4 = data_list[4]
        img = [my_PreProc(nii_to_np(img_path))]
        mask = [nii_to_np(mask_path1), nii_to_np(mask_path2), nii_to_np(mask_path3), nii_to_np(mask_path4)]
        mask = np.round(mask / np.max(mask))
        data = get_patch_for_one_img(img, mask, patch_size, slice_patch_size, train)
        # data = get_imgs_and_masks_patch(list(zip(img, mask)), patch_size, slice_patch_size, train)
        for patch_idx, t in enumerate(data):
            j += 1
            b.append(t)
            if j % batch_size == 0:
                yield b
                b = []
    if len(b) > 0:
        yield b
