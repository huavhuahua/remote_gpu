# main.py -- train for whole image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import argparse

from data_prepare_3d_v3 import get_imgs_and_masks_path, batch
from data_prepare_3d_v3 import get_patch_for_one_img
from vnet import VNet
import SimpleITK as sitk
import nibabel as nib
from preprocess import my_PreProc
from dice_loss import BinaryDiceLoss

root_path = r'../data'
root_result = r'../result_muti_whole_image'
root_model = os.path.join(root_result, 'model')
os.makedirs(root_model, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--arch', '-a', metavar='ARCH', default='VNet')
parser.add_argument('--root', default=root_path, help='path to dataset (images list file)')
parser.add_argument('--dataset', default='CT_whole_resize', help='CT, DRIVE or CHASEDB')
parser.add_argument('--x_patch_size', type=int, default=512, help='x patch size')
parser.add_argument('--y_patch_size', type=int, default=256, help='y patch size')
parser.add_argument('--slice_patch_size', type=int, default=64, help='slice patch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate for training')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
parser.add_argument('--optim', type=str, default='SGD', help='optim for training, Adam / SGD (default)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay for SGD / Adam')
parser.add_argument('--gpu', type=bool, default=True, help='use GPU or not')
parser.add_argument('--model_path', type=str, default=root_model, help='model file to save')
parser.add_argument('--resume_path', type=str, default=None, help='model file to resume to train')
# args = parser.parse_args()
args = parser.parse_known_args()[0]
print(args)

# model
Net = eval(args.arch)  # VNet
model = Net(elu=True, nll=False, numclass=4)
if args.gpu:
    model = model.cuda()

# criterion
criterion = nn.BCELoss().cuda()
criterion2 = BinaryDiceLoss().cuda()
# criterion = nn.CrossEntropyLoss().cuda()

# use pre-train
if args.resume_path is not None:
    pretrained_model = torch.load(args.resume_path)
    model.load_state_dict(pretrained_model)

# optim
if args.optim == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
else:
    raise NotImplementedError('Other optimizer is not implemented')


# lr_reduce
def lr_opt(lr, epoch):
    e_max = 60
    if epoch <= e_max:
        lr_new = lr * (0.1 ** (float(epoch) / 15))
    else:
        lr_new = lr * (0.1 ** (float(e_max) / 15))
    return lr_new


def nii_to_np(niifile):
    # 开始读取nii文件
    img_np = nib.load(niifile).get_data()  # 读取nii
    img_np = np.array(img_np)
    return img_np


# train
def train_net(model, criterion, criterion2, optimizer, epoch, lr_cur):
    loss_epoch = []

    # data
    data_list = get_imgs_and_masks_path(args.root, args.dataset, datatpye='training')
    dataloader = batch(data_list, args.batch_size, args.x_patch_size, args.y_patch_size, args.slice_patch_size)

    # switch to train mode
    model.train()

    for i, b in enumerate(dataloader):
        imgs = np.array([i[0] for i in b]).astype(np.float32)
        true_masks = np.array([i[1] for i in b]).astype(np.float32)

        imgs = torch.from_numpy(imgs)  # .unsqueeze(1)
        true_masks = torch.from_numpy(true_masks)  # .unsqueeze(1)

        if args.gpu:
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()

        masks_pred = model(imgs)

        mask_pred_flat = masks_pred.view(-1)
        true_mask_flat = true_masks.view(-1)

        loss = criterion(mask_pred_flat, true_mask_flat)
        # print(loss)
        loss_epoch.append(loss.item())

        print(
            'Epoch:{}, batch:{}/{}, Lr:{}, Loss:{}'.format(epoch, i, len(data_list) // args.batch_size * 1, lr_cur,
                                                           loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch{} Finished, Lr:{}, Loss:{}'.format(epoch, lr_cur, np.mean(loss_epoch)))

    return loss_epoch


def val_net(model, dataset_list, x_patch_size, y_patch_size, slice_patch_size, val_patch_num=1):
    model.eval()
    loss_val = []
    for i, data_list in enumerate(dataset_list):
        img_path = data_list[0]
        mask_path1 = data_list[1]
        mask_path2 = data_list[2]
        mask_path3 = data_list[3]
        mask_path4 = data_list[4]
        img = [my_PreProc(nii_to_np(img_path))]
        mask = [nii_to_np(mask_path1), nii_to_np(mask_path2), nii_to_np(mask_path3), nii_to_np(mask_path4)]
        mask = np.round(mask / np.max(mask))
        data = get_patch_for_one_img(img, mask, x_patch_size, y_patch_size, slice_patch_size, train=True, patch_num=val_patch_num)
        for i, b in enumerate(data):
            img_patch = b[0].astype(np.float32)
            true_mask_patch = b[1].astype(np.float32)

            img_patch = torch.from_numpy(img_patch).unsqueeze(0)  # .unsqueeze(0)
            true_mask_patch = torch.from_numpy(true_mask_patch).unsqueeze(0)  # .unsqueeze(0)

            if args.gpu:
                img_patch = img_patch.cuda()
                true_mask_patch = true_mask_patch.cuda()

            masks_pred = model(img_patch)

            loss = criterion(masks_pred.contiguous().view(-1), true_mask_patch.contiguous().view(-1))
            loss_val.append(loss.item())
            print('Epoch:{}, VALIDATION batch:{}/{}, Loss:{}'.format(epoch, i, len(data_list), loss.item()))
    print('VALIDATION Finished, Loss:{}'.format(np.mean(loss_val)))
    return np.mean(loss_val)


def save_result(loss_train):
    # save train loss per iteration
    plt.figure(1)
    plt.plot(loss_train, 'r')
    plt.title('loss_train', fontsize='large', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.xlabel('Iteration', fontweight='bold')
    plt.savefig(os.path.join(root_result, 'loss_train.png'))
    plt.close()


def save_result_epochlast(loss_train, loss_val):
    # save train and validation loss per epoch
    plt.figure(1)
    plt.plot(loss_train, 'r')
    plt.plot(loss_val, 'b')
    plt.title('loss', fontsize='large', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.legend(['train', 'val'])
    plt.savefig(os.path.join(root_result, 'loss.png'))
    plt.close()


def np2nii(img_np):
    img = np.transpose(img_np, [2, 1, 0])
    out = sitk.GetImageFromArray(img)
    return out


if __name__ == '__main__':

    # validation dataset
    val_list = get_imgs_and_masks_path(args.root, args.dataset, datatpye='val')

    loss_train = []
    loss_train_epoch_mean = []
    loss_val_epoch_mean = []
    for epoch in range(args.start_epoch, args.epochs):
        lr_cur = lr_opt(args.lr, epoch)  # speed change
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_cur

        # train for one epoch
        loss_train_epoch = train_net(model, criterion, criterion2, optimizer, epoch, lr_cur)
        loss_train += loss_train_epoch
        loss_train_epoch_mean.append(np.mean(loss_train_epoch))

        # evaluate on validation set
        loss_val_epoch_mean.append(val_net(model, val_list, args.x_patch_size, args.y_paych_size, args.slice_patch_size))

        # save checkpoint and loss
        save_result(loss_train)
        save_result_epochlast(loss_train_epoch_mean, loss_val_epoch_mean)
        torch.save(model.state_dict(), os.path.join(args.model_path, 'epoch_' + str(epoch) + '_model.pkl'))

    print("finish")
