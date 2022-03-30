import numpy as np
import SimpleITK as sitk
from glob import glob
import os


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


def resize_for_dir(src_path, newSize):
    for root, dirs, files in os.walk(src_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(file_path)
            img = sitk.ReadImage(file_path)
            img_resize = resize_image_itk(img, newSize)
            save_root = root.replace("CT_ero_256", "CT_whole_resize")
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            sitk.WriteImage(img_resize, save_root + '/' + filename)


src_path = '../data/CT_ero_256'
resize_for_dir(src_path, (512, 256, 64))
print('finish')
