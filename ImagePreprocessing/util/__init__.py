import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.util import montage
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from PIL import Image

def get_vol_montage(img, vmin, vmax, is_show_montage=False):
    montage_np = montage(np.expand_dims(img, axis=-1), multichannel=True).squeeze()
    if is_show_montage:
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 16))
        ax1.imshow(montage_np, cmap='bone', vmin=vmin, vmax=vmax)
    return montage_np


def image_to_PIL(img, vmin, vmax):
    img[img > vmax] = vmax
    img[img < vmin] = vmin
    PIL_img = Image.fromarray(((img - vmin) * (1 / (vmax - vmin)) * 255).astype('uint8'))
    return PIL_img

def get_vol_info(vol_range_dict, slice_center_dict, dataset_name, m, patient_id):
    vol_range = 'full'
    slice_center = '1/2'
    if m in slice_center_dict[dataset_name].keys():
        for vol_range_, patient_list in vol_range_dict[dataset_name][m].items():
            if patient_id in patient_list:
                vol_range = vol_range_
                break
    if m in slice_center_dict[dataset_name].keys():
        for slice_center_, patient_list in slice_center_dict[dataset_name][m].items():
            if patient_id in patient_list:
                slice_center = slice_center_
                break
    return [vol_range, slice_center]


def load_vol_from_path(vol_paths):
    vol_np_list = []
    for vol_path in vol_paths:
        vol_nifty = nib.load(vol_path)
        vol_np = np.array(vol_nifty.dataobj).transpose(1, 0, 2)
        vol_np_list.append(vol_np)
    return vol_np_list

def img_resize(img, shape):
    return resize(img, shape, anti_aliasing=True, preserve_range=True)

def img_standardize(img):
    scaler = StandardScaler()
    return scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

def save_img(img, img_path, format='jpg'):
    if format == 'nii':
        img_nifty = nib.Nifti1Image(img, affine=np.eye(4))
        nib.save(img_nifty, img_path + '.nii')

    elif format == 'jpg':
        img_PIL = Image.fromarray(img)
        img_PIL.save(img_path + '.jpg')
