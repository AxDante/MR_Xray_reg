from __future__ import print_function

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import csv

def load_image(path):
    if(path[-3:] == 'dng'):
        import rawpy
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
    elif(path[-3:]=='bmp' or path[-3:]=='jpg' or path[-3:]=='png'):
        return cv2.imread(path)[:,:,::-1]
    else:
        img = (255*plt.imread(path)[:,:,:3]).astype('uint8')
    return img

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def tensor2np(img_tensor, np_type = np.float):
    return img_tensor[0].cpu().float().numpy().astype(np_type)

def np2tensor(img):
    return torch.Tensor(img[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def normalize_img(img, undo=False):
    if not undo:
        return img / (255.0/2.0) - 1.0
    else:
        return (img + 1)* 255.0 / 2.0


def save_diff_map(diff, imageA, imageB):
    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(np.float32(imageA), (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(np.float32(imageB), (x, y), (x + w, y + h), (0, 0, 255), 2)

    # show the output images
    #cv2.imshow("Original", imageA)
    #cv2.imshow("Modified", imageB)
    cv2.imshow("Diff", diff)
    cv2.imshow("Thresh", thresh)
    #cv2.waitKey(0)

def save_to_file(file_path, data):
    with open(file_path, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            writer.writerow(row)