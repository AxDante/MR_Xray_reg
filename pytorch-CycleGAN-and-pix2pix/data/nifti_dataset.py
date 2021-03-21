import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

import numpy as np
import nibabel as nib
import torch


class NiftiDataset(BaseDataset):

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        print('self.dir_A ', self.dir_A)
        print('self.dir_B ', self.dir_B)


        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        print('self.A_paths ', self.A_paths)
        print('self.B_paths ', self.B_paths)

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        print('self.A_size ', self.A_size)
        print('self.B_size ', self.B_size)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]


        print('HEREHEREHERE')
        print(A_path)
        A_nifti = nib.load(A_path)
        #print(A_nifti)
        print('an', A_nifti.shape)
        A_numpy = np.array(A_nifti.dataobj)
        #print(A_numpy)
        #print(A_numpy.shape)
        print('an2', A_numpy.shape)

        A = torch.from_numpy(A_numpy)
        #print(A)
        print('an3', A.shape)
        print(B_path)

        B_nifti = nib.load(B_path)
        #print(B_nifti)
        B_numpy = np.array(B_nifti.dataobj)
        #print(B_numpy)
        B = torch.from_numpy(B_numpy)
        print(B.shape)
        # else:
        #     A_img = Image.open(A_path).convert('RGB')
        #     B_img = Image.open(B_path).convert('RGB')
        #     # apply image transformation
        #     A = self.transform_A(A_img)
        #     B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
