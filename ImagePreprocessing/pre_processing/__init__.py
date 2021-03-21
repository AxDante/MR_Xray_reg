"""
  Email:  knight16729438@gmail.com
  License: MIT

"""
import os
import util
import math
import random
import monai
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

# Single volume
class PP_Vol:
    def __init__(self, id, m, path, vrange, cent):
        self.id = id
        self.m = m
        self.path = path
        self.vrange = vrange
        self.cent = cent

# Class for all preprocessing volumes
class PP_Vols:
    def __init__(self, datasets_name, pp_main_path,
                 format='jpg',
                 reg_size={'ct': [320, 320, 3], 'mr': [256, 256, 3]},
                 roi_thickness=3,
                 threshold_range={'mr': [0, 400], 'ct': [-1000, 1400]},
                 slice_std={'ct': 15, 'mr': 6},
                 n_imgs={'mr': {'train': 2000, 'test': 50}, 'ct': {'train': 2000, 'test': 50}},
                 pp_folder_name={'mr': {'train': 'train_mr', 'test': 'test_mr'},
                                 'ct': {'train': 'train_ct', 'test': 'test_ct'}},
                 pp_info='',
                 test_dict={'ct':[], 'mr':[]},
                 is_use_date=True,
                 is_save_preview=True):

        self.vols = []
        self.datasets_name = datasets_name
        self.threshold_range = threshold_range
        self.slice_std = slice_std
        self.reg_size = reg_size
        self.format = format
        self.n_imgs = n_imgs
        self.test_dict = test_dict
        self.is_save_preview = is_save_preview
        self.roi_thickness = roi_thickness

        assert roi_thickness == 1 or roi_thickness == 3

        # Generate folders to save the preprocessed dataset
        date_str = ''
        if pp_info != '': pp_info = '_' + pp_info
        if is_use_date: date_str = 'date{:02d}{:02d}'.format(date.today().month, date.today().day)
        pp_dir_name = '{}_thickness_{}_format_{}_{}{}'.format('_'.join(self.datasets_name),
                                                              self.roi_thickness,
                                                              self.format,
                                                              date_str,
                                                              pp_info)

        self.main_path = os.path.join(pp_main_path, pp_dir_name)
        self.path = {'mr': {'train': os.path.join(self.main_path, pp_folder_name['mr']['train']),
                            'test': os.path.join(self.main_path, pp_folder_name['mr']['test'])},
                     'ct': {'train': os.path.join(self.main_path, pp_folder_name['ct']['train']),
                            'test': os.path.join(self.main_path, pp_folder_name['ct']['test'])}}
        self.checkpoint_path = os.path.join(self.main_path, 'checkpoints')
        self.preview_path = os.path.join(self.main_path, 'preview')

        print('Preprocessed images will be saved to: {}'.format(self.main_path))
        os.makedirs(self.main_path, exist_ok=True)
        for m in ['mr', 'ct']:
            for mode in ['train', 'test']:
                os.makedirs(self.path[m][mode], exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        if self.is_save_preview: os.makedirs(self.preview_path, exist_ok=True)

    # Add a PP_Vol
    def add_vol(self, PP_Vol):
        self.vols.append(PP_Vol)

        # Volume montage preview
        if self.is_save_preview:
            print(PP_Vol.id + '_' + PP_Vol.m)
            vol_np = self.load_vol(PP_Vol.id, PP_Vol.m)
            print('volume shape: '.format(vol_np.shape))
            vol_montage = util.get_vol_montage(vol_np.transpose(2, 0, 1), self.threshold_range[m][0], threshold_range[m][1])
            vol_montage_PIL = util.image_to_PIL(vol_montage, self.threshold_range[m][0], self.threshold_range[m][1])

            # Save image montage preview
            vol_montage_PIL.save(os.path.join(self.preview_path, PP_Vol.id + '_' + PP_Vol.m + '_preview.jpg'))

    # Get number of volumes of chosen image modality
    def get_vol_num(self, m):
        return len(util.get_path_list(self, m))

    # Load the volume as numpy array
    def load_vol(self, id, m):
        for pp_vol in self.vols:
            if pp_vol.id == id and pp_vol.m == m:
                vol_nifty = nib.load(pp_vol.path)
                vol_np = np.array(vol_nifty.dataobj).transpose(1, 0, 2)
                return vol_np
        raise Exception('The volume ID ({}) and modality ({}) cannot be found.'.format(id, m))

    # Display a slice of single volume
    def display_vol(self, id, m, slice_num=-1):
        vol_np = self.load_vol(id, m)
        if slice_num == -1:
            slice_num = vol_np.shape[2] // 2
        plt.imshow(vol_np[:, :, slice_num], cmap='bone')

    # Return a list of image paths of chosen image modality
    def get_vols(self, m, train_split):
        vol_list = []
        test_vol_list = []
        test_id = []
        for dn in self.datasets_name:
            if m in self.test_dict[dn]:
                test_id = [dn + '_' + id for id in self.test_dict[dn][m]]

        # Identify the test volumes
        for pp_vol in self.vols:
            if pp_vol.m == m:
                if pp_vol.id not in test_id:
                    vol_list.append(pp_vol)
                else:
                    test_vol_list.append(pp_vol)

        # Split other volumes into training and testing sets base on train split ratio
        vols = {'train': vol_list[0:math.floor(len(vol_list) * train_split)],
                'test': vol_list[math.floor(len(vol_list) * train_split):]}

        # Add the test volumes
        vols['test'] = vols['test'] + test_vol_list

        return vols

    # Generate pre-processed images
    def generate_pp_imgs(self, shuffle=True, train_split=0.8, verbose=True, is_preview_image=False):

        modality_list = ['mr', 'ct']

        # Open split info file
        with open(os.path.join(self.main_path, 'split_info.txt'), 'w') as f:

            for m in modality_list:

                vols = self.get_vols(m, train_split)

                # Check if train and test sets are not empty
                assert len(vols['train']) >= 1
                assert len(vols['test']) >= 1

                f.write('-----\n')
                f.write('{} train ID:\n'.format(m))
                f.writelines('{}\t'.format(vol.id) for vol in vols['train'])
                f.write('\n{} test ID:\n'.format(m))
                f.writelines('{}\t'.format(vol.id) for vol in vols['test'])
                f.write('\n')

                vol_path_dict = {'train': [vol.path for vol in vols['train']],
                                 'test': [vol.path for vol in vols['test']]}
                # print( [vol.path for vol in vols['train']])
                # #
                # print([vol.path for vol in vols['test']])
                # print([vol.id for vol in vols['test']])

                vol_nps_dict = {'train': util.load_vol_from_path(vol_path_dict['train']),
                                'test': util.load_vol_from_path(vol_path_dict['test'])}

                for mode in ['train', 'test']:
                    f.write('--{}--\n'.format(mode))
                    if verbose: print('Generating pre-processing images : {}'.format(m))

                    # Generate training images
                    tr_img_count = 0

                    while tr_img_count < self.n_imgs[m][mode]:

                        if not shuffle or mode == 'test':
                            vol_id = tr_img_count % len(vol_nps_dict[mode])
                        else:
                            vol_id = random.randint(0, len(vol_nps_dict[mode]) - 1)


                        vol = vols[mode][vol_id]
                        vol_np = vol_nps_dict[mode][vol_id]

                        f.writelines('{}.{}: {}\n'.format(str(tr_img_count).zfill(4), self.format, vol.id) )

                        ######################

                        # Randomly crop a region in the volume

                        # Determine the region boundary in the z axis
                        if m == 'ct':
                            # volume center (where femur bones locate) usually locate around at the 1/2 portion on the sagittal plane
                            if vol.cent == '1/2':
                                div = 1 / 2

                            # volume center locate around at the 2/3 portion on the sagittal plane in NMDID dataset
                            elif vol.cent == '2/3':
                                div = 2 / 3

                            reg_z_s_max = vol_np.shape[2] - self.reg_size[m][2]

                            # select starting center through normailzation
                            reg_z_s = np.random.normal(reg_z_s_max * div, reg_z_s_max // self.slice_std[m], 1)[
                                0].astype(int)
                            if reg_z_s > reg_z_s_max: reg_z_s = reg_z_s_max
                            if reg_z_s < 0: reg_z_s = 0

                        elif m == 'mr':
                            # reg_z_s = np.random.randint(0, vol_np.shape[2]-self.reg_size[m][2])

                            if vol.cent == '1/2':
                                div = 1 / 2
                            elif vol.cent == '2/3':
                                div = 2 / 3
                            reg_z_s_max = vol_np.shape[2] - self.reg_size[m][2]
                            reg_z_s = np.random.normal(reg_z_s_max * div, reg_z_s_max // self.slice_std[m], 1)[
                                0].astype(int)
                            if reg_z_s > reg_z_s_max: reg_z_s = reg_z_s_max
                            if reg_z_s < 0: reg_z_s = 0

                        reg_z_e = reg_z_s + self.reg_size[m][2]

                        # Determine the region boundary in the x and y axis
                        if m == 'ct' and vol.vrange == 'full':
                            reg_y_s = vol_np.shape[0] // 4
                            reg_y_e = 3 * (vol_np.shape[0] // 4)
                            reg_x_s = vol_np.shape[1] // 2
                            reg_x_e = vol_np.shape[1]

                        elif m == 'ct' and vol.vrange == 'half':
                            reg_y_s = vol_np.shape[0] // 6
                            reg_y_e = 5 * (vol_np.shape[0] // 6)
                            reg_x_s = vol_np.shape[1] // 3
                            reg_x_e = vol_np.shape[1]

                        elif m == 'ct' and vol.vrange == 'hip':
                            reg_y_s = vol_np.shape[0] // 6
                            reg_y_e = 5 * (vol_np.shape[0] // 6)
                            reg_x_s = vol_np.shape[1] // 6
                            reg_x_e = 5 * (vol_np.shape[1] // 6)

                        elif m == 'mr':
                            reg_y_s = 0
                            reg_y_e = vol_np.shape[0]
                            reg_x_s = 0
                            reg_x_e = vol_np.shape[1]

                        vol_np_crop = vol_np[reg_x_s:reg_x_e, reg_y_s:reg_y_e, reg_z_s:reg_z_e]
                        vol_np_crop = util.img_resize(vol_np_crop, self.reg_size[m])

                        ######################
                        # Image normalization & augmentation

                        modality_standardize = {'mr': True, 'ct': False}

                        if modality_standardize[m]:
                            vol_np_crop = util.img_standardize(vol_np_crop)
                            min_scale = -3
                            max_scale = 3
                        else:
                            min_scale = self.threshold_range[m][0]
                            max_scale = self.threshold_range[m][1]

                        # Data augmentation transform functions
                        data_aug_transforms = monai.transforms.Compose([
                            monai.transforms.ScaleIntensityRange(a_min=min_scale, a_max=max_scale, b_min=0, b_max=255),
                            monai.transforms.ThresholdIntensity(255.0, above=False, cval=255.0),
                            monai.transforms.ThresholdIntensity(0, above=True, cval=0),
                            monai.transforms.RandZoom(min_zoom=0.9, max_zoom=1.1, prob=1.0, keep_size=True),
                            monai.transforms.AddChannel(),
                            monai.transforms.RandSpatialCrop(roi_size=[256, 256, self.roi_thickness],
                                                             random_size=False),
                            monai.transforms.RandRotate(range_z=(-0.1, 0.1), prob=1.0, keep_size=True),
                            monai.transforms.SqueezeDim(dim=0),
                            monai.transforms.CastToType(np.uint8),
                        ])

                        # Perform data augmentation
                        vol_np_aug = data_aug_transforms(vol_np_crop)

                        # Tile the third channed if roi_thickness is set to 1
                        if vol_np_aug.shape[2] == 1:
                            vol_np_aug = np.tile(vol_np_aug[:, :], [1, 1, 3])

                        # Save pre-processed images to folder
                        img_name = os.path.join(self.path[m][mode], str(tr_img_count).zfill(4))
                        util.save_img(vol_np_aug, img_name, self.format)

                        tr_img_count += 1

        f.close()

# # Single volume
# class PP_Vol:
#     def __init__(self, id, m, path, vrange, cent):
#         self.id = id
#         self.m = m
#         self.path = path
#         self.vrange = vrange
#         self.cent = cent
#
# # Class for all preprocessing volumes
# class PP_Vols:
#     def __init__(self, datasets_name, pp_main_path,
#                  format='jpg',
#                  reg_size={'ct': [320, 320, 3], 'mr': [256, 256, 3]},
#                  roi_thickness=3,
#                  threshold_range={'mr': [0, 400], 'ct': [-1000, 1400]},
#                  slice_std={'ct': 15, 'mr': 6},
#                  n_imgs={'mr': {'train': 2000, 'test': 50}, 'ct': {'train': 2000, 'test': 50}},
#                  pp_folder_name={'mr': {'train': 'train_mr', 'test': 'test_mr'},
#                                  'ct': {'train': 'train_ct', 'test': 'test_ct'}},
#                  pp_info='',
#                  is_use_date=True,
#                  is_save_preview=True):
#
#         self.vols = []
#         self.datasets_name = datasets_name
#         self.threshold_range = threshold_range
#         self.slice_std = slice_std
#         self.reg_size = reg_size
#         self.format = format
#         self.n_imgs = n_imgs
#         self.is_save_preview = is_save_preview
#         self.roi_thickness = roi_thickness
#
#         assert roi_thickness == 1 or roi_thickness == 3
#
#         # Generate folders to save the preprocessed dataset
#         date_str = ''
#         if pp_info != '': pp_info = '_' + pp_info
#         if is_use_date: date_str = 'date{:02d}{:02d}'.format(date.today().month, date.today().day)
#         pp_dir_name = '{}_thickness_{}_format_{}_{}{}'.format('_'.join(self.datasets_name),
#                                                               self.roi_thickness,
#                                                               self.format,
#                                                               date_str,
#                                                               pp_info)
#
#         self.main_path = os.path.join(pp_main_path, pp_dir_name)
#         self.path = {'mr': {'train': os.path.join(self.main_path, pp_folder_name['mr']['train']),
#                             'test': os.path.join(self.main_path, pp_folder_name['mr']['test'])},
#                      'ct': {'train': os.path.join(self.main_path, pp_folder_name['ct']['train']),
#                             'test': os.path.join(self.main_path, pp_folder_name['ct']['test'])}}
#         self.checkpoint_path = os.path.join(self.main_path, 'checkpoints')
#         self.preview_path = os.path.join(self.main_path, 'preview')
#
#         print('Preprocessed images will be saved to: {}'.format(self.main_path))
#         os.makedirs(self.main_path, exist_ok=True)
#         for m in ['mr', 'ct']:
#             for mode in ['train', 'test']:
#                 os.makedirs(self.path[m][mode], exist_ok=True)
#         os.makedirs(self.checkpoint_path, exist_ok=True)
#         if self.is_save_preview: os.makedirs(self.preview_path, exist_ok=True)
#
#     def add_vol(self, PP_Vol):
#         self.vols.append(PP_Vol)
#         if self.is_save_preview:
#             print(PP_Vol.id + '_' + PP_Vol.m)
#             vol_np = self.load_vol(PP_Vol.id, PP_Vol.m)
#             print(vol_np.shape)
#             vol_montage = util.get_vol_montage(vol_np.transpose(2, 0, 1), self.threshold_range[m][0], threshold_range[m][1])
#             vol_montage_PIL = util.image_to_PIL(vol_montage, self.threshold_range[m][0], self.threshold_range[m][1])
#             vol_montage_PIL.save(os.path.join(self.preview_path, PP_Vol.id + '_' + PP_Vol.m + '_preview.jpg'))
#
#     # Get number of volumes of chosen image modality
#     def get_vol_num(self, m):
#         return len(util.get_path_list(self, m))
#
#     # Load the volume as numpy array
#     def load_vol(self, id, m):
#         for pp_vol in self.vols:
#             if pp_vol.id == id and pp_vol.m == m:
#                 vol_nifty = nib.load(pp_vol.path)
#                 vol_np = np.array(vol_nifty.dataobj).transpose(1, 0, 2)
#                 return vol_np
#         raise Exception('The volume ID ({}) and modality ({}) cannot be found.'.format(id, m))
#
#     # Display a slice of single volume
#     def display_vol(self, id, m, slice_num=-1):
#         vol_np = self.load_vol(id, m)
#         if slice_num == -1:
#             slice_num = vol_np.shape[2] // 2
#         plt.imshow(vol_np[:, :, slice_num], cmap='bone')
#
#     # Return a list of image paths of chosen image modality
#     def get_vol_list(self, m):
#         pp_vol_list = []
#         for pp_vol in self.vols:
#             if pp_vol.m == m:
#                 pp_vol_list.append(pp_vol)
#         return pp_vol_list
#
#     # Generate pre-processed images
#     def generate_pp_imgs(self, shuffle=True, train_split=0.8, verbose=True, is_preview_image=False):
#
#         modality_list = ['mr', 'ct']
#
#         with open(os.path.join(self.main_path, 'split_info.txt'), 'w') as f:
#
#             for m in modality_list:
#
#                 vol_list = self.get_vol_list(m)
#                 vols = {'train': vol_list[0:math.floor(len(vol_list) * train_split)],
#                         'test': vol_list[math.floor(len(vol_list) * train_split):]}
#
#                 # Check if train and test sets are not empty
#                 assert len(vols['train']) >= 1
#                 assert len(vols['test']) >= 1
#
#                 f.write('{} train ID:\n'.format(m))
#                 f.writelines('{}\t'.format(vol.id) for vol in vols['train'])
#                 f.write('\n{} test ID:\n'.format(m))
#                 f.writelines('{}\t'.format(vol.id) for vol in vols['test'])
#                 f.write('\n')
#
#                 vol_paths = [vol.path for vol in vols['train']]
#
#                 vol_nps = util.load_vol_from_path(vol_paths)
#
#                 for mode in ['train', 'test']:
#                     if verbose: print('Generating pre-processing images : {}'.format(m))
#
#                     # Generate training images
#                     tr_img_count = 0
#
#                     while tr_img_count < self.n_imgs[m][mode]:
#
#                         if not shuffle or mode == 'test':
#                             vol_id = tr_img_count // len(vol_nps)
#                         else:
#                             vol_id = random.randint(0, len(vol_paths) - 1)
#                         vol = vols['train'][vol_id]
#                         vol_np = vol_nps[vol_id]
#
#                         ######################
#
#                         # Randomly crop a region in the volume
#
#                         # Determine the region boundary in the z axis
#                         if m == 'ct':
#                             # volume center (where femur bones locate) usually locate around at the 1/2 portion on the sagittal plane
#                             if vol.cent == '1/2':
#                                 div = 1 / 2
#
#                             # volume center locate around at the 2/3 portion on the sagittal plane in NMDID dataset
#                             elif vol.cent == '2/3':
#                                 div = 2 / 3
#
#                             reg_z_s_max = vol_np.shape[2] - self.reg_size[m][2]
#
#                             # select starting center through normailzation
#                             reg_z_s = np.random.normal(reg_z_s_max * div, reg_z_s_max // self.slice_std[m], 1)[
#                                 0].astype(int)
#                             if reg_z_s > reg_z_s_max: reg_z_s = reg_z_s_max
#                             if reg_z_s < 0: reg_z_s = 0
#
#                         elif m == 'mr':
#                             # reg_z_s = np.random.randint(0, vol_np.shape[2]-self.reg_size[m][2])
#
#                             if vol.cent == '1/2':
#                                 div = 1 / 2
#                             elif vol.cent == '2/3':
#                                 div = 2 / 3
#                             reg_z_s_max = vol_np.shape[2] - self.reg_size[m][2]
#                             reg_z_s = np.random.normal(reg_z_s_max * div, reg_z_s_max // self.slice_std[m], 1)[
#                                 0].astype(int)
#                             if reg_z_s > reg_z_s_max: reg_z_s = reg_z_s_max
#                             if reg_z_s < 0: reg_z_s = 0
#
#                         reg_z_e = reg_z_s + self.reg_size[m][2]
#
#                         # Determine the region boundary in the x and y axis
#                         if m == 'ct' and vol.vrange == 'full':
#                             reg_y_s = vol_np.shape[0] // 4
#                             reg_y_e = 3 * (vol_np.shape[0] // 4)
#                             reg_x_s = vol_np.shape[1] // 2
#                             reg_x_e = vol_np.shape[1]
#
#                         elif m == 'ct' and vol.vrange == 'half':
#                             reg_y_s = vol_np.shape[0] // 6
#                             reg_y_e = 5 * (vol_np.shape[0] // 6)
#                             reg_x_s = vol_np.shape[1] // 3
#                             reg_x_e = vol_np.shape[1]
#
#                         elif m == 'ct' and vol.vrange == 'hip':
#                             reg_y_s = vol_np.shape[0] // 6
#                             reg_y_e = 5 * (vol_np.shape[0] // 6)
#                             reg_x_s = vol_np.shape[1] // 6
#                             reg_x_e = 5 * (vol_np.shape[1] // 6)
#
#                         elif m == 'mr':
#                             reg_y_s = 0
#                             reg_y_e = vol_np.shape[0]
#                             reg_x_s = 0
#                             reg_x_e = vol_np.shape[1]
#
#                         vol_np_crop = vol_np[reg_x_s:reg_x_e, reg_y_s:reg_y_e, reg_z_s:reg_z_e]
#                         vol_np_crop = util.img_resize(vol_np_crop, self.reg_size[m])
#
#
#
#
#                         # print('#########')
#                         # print(np.max(vol_np_crop))
#                         # print(np.min(vol_np_crop))
#                         # print(vol_np_crop[:,:,0].shape)
#
#                         # plt.imshow(vol_np_crop[:,:,0], cmap='bone')
#
#                         ######################
#                         # Image normalization & augmentation
#
#                         # if m == 'mr':
#
#                         vol_np_crop = util.img_standardize(vol_np_crop)
#
#                         # print(np.max(vol_np_crop))
#                         # print(np.min(vol_np_crop))
#                         # print(vol_np_crop.shape)
#
#                         # Data augmentation transform functions
#                         data_aug_transforms = monai.transforms.Compose([
#                             # monai.transforms.RandScaleIntensity(0.1, prob=0.1),
#                             monai.transforms.ScaleIntensityRange(a_min=-3, a_max=3, b_min=0, b_max=255),
#                             # monai.transforms.ScaleIntensityRange(a_min=threshold_range[m][0], a_max=threshold_range[m][1], b_min=0, b_max=255),
#                             monai.transforms.ThresholdIntensity(255.0, above=False, cval=255.0),
#                             monai.transforms.ThresholdIntensity(0, above=True, cval=0),
#                             # monai.transforms.ScaleIntensity(minv=0.0, maxv=255.0),
#                             monai.transforms.RandZoom(min_zoom=0.9, max_zoom=1.1, prob=1.0, keep_size=True),
#                             monai.transforms.AddChannel(),
#                             monai.transforms.RandSpatialCrop(roi_size=[256, 256, self.roi_thickness],
#                                                              random_size=False),
#                             monai.transforms.RandRotate(range_z=(-0.1, 0.1), prob=1.0, keep_size=True),
#                             monai.transforms.SqueezeDim(dim=0),
#                             monai.transforms.CastToType(np.uint8),
#                         ])
#
#                         # Perform data augmentation
#                         vol_np_aug = data_aug_transforms(vol_np_crop)
#
#                         # Tile the third channed if roi_thickness is set to 1
#                         if vol_np_aug.shape[2] == 1:
#                             vol_np_aug = np.tile(vol_np_aug[:, :], [1, 1, 3])
#
#                         # Save pre-processed images to folder
#                         img_name = os.path.join(self.path[m][mode], str(tr_img_count).zfill(4))
#                         util.save_img(vol_np_aug, img_name, self.format)
#
#                         tr_img_count += 1
#
#         f.close()
