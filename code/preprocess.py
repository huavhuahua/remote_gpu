# preprocess.py
import numpy as np
import cv2

#My pre processing (use for both training and testing!)
def my_PreProc(data):  # data-(h,w,c)
    # #black-white conversion
    # train_imgs = rgb2gray(data)
    # #my preprocessing:
    # train_imgs = clahe_equalized(train_imgs)
    # train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = np.float64(data)
    train_imgs = (train_imgs-np.min(train_imgs)) / (np.max(train_imgs)-np.min(train_imgs))
    return train_imgs


#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

def rgb2gray(rgb):
    bn_imgs = rgb[:,:,0]*0.299 + rgb[:,:,1]*0.587 + rgb[:,:,2]*0.114
    return bn_imgs

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def clahe_equalized(imgs):
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = clahe.apply(np.array(imgs, dtype = np.uint8)) #CLAHE
    return imgs_equalized

def adjust_gamma(imgs, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = cv2.LUT(np.array(imgs, dtype = np.uint8), table)
    return new_imgs