import os
import glob
import util
from pre_processing import PP_Vol, PP_Vols

if __name__ == '__main__':
    # preprocessing settings
    is_create_pp = True
    is_preview_image = False
    is_mr_normalization = True
    roi_thickness = 3
    reg_size = {'ct':[320,320,3], 'mr':[290,290,3]}
    n_imgs = {'mr':{'train':30, 'test': 10}, 'ct':{'train':30, 'test': 10}}
    #n_imgs = {'mr':{'train':30, 'test': 5}, 'ct':{'train':30, 'test': 5}}
    slice_std = {'ct': 12, 'mr': 4}
    threshold_range = {'mr': [0, 400], 'ct': [-1000, 1400]}
    pp_format = 'jpg'  # 'jpg' or 'nii'
    pp_folder_name = {'mr': {'train': 'trainA', 'test': 'testA'}, 'ct': {'train': 'trainB', 'test': 'testB'}}
    dataset_name_list = ['AVN', 'NMDID']
    parent_dir = os.path.abspath(__file__ + "/../../../")
    dataset_path_dict = {'AVN': parent_dir + '/dataset/AVN/', 'NMDID': parent_dir + '/dataset/NMDID/'}
    modality_filename_dict = {'ct': 'CT.nii', 'mr': 'MR_T1.nii', 'xray':'Xray.mhd'}

    patient_dict = {}    # Dictionary of patient volumes ('mr' and 'ct')
    vol_range_dict = {}    # Dictionary of volume range ('full'(full body image), 'half'(lower half body image), 'hip')
    slice_center_dict = {}   # Dictionary of slice center (femur bone location) along the sagittal axis ('1/2', '2/3')
    test_dict = {}  # Dictionary of test volumes (specify volumes to be used as test sets, could be left empty)

    result_path = parent_dir + '/result/'

    if 'AVN' in dataset_name_list:

      patient_dict['AVN'] = {'ct': ['1', '3', '12', '13', '20', '23'],
                              'mr': ['1', '3', '4',  '6', '7', '9', '10', '11', '12', '13', '15', '19', '20', '22', '23', '25', '26', '28', '30']}

      #'ct_mr': ['1', '3', '12', '13', '20','23']
      vol_range_dict['AVN'] = {'ct': {'full':['1', '20'], 'half':['3', '12', '23'], 'hip':['13']}}
      slice_center_dict['AVN'] = {'ct': {'1/2':['1', '3', '12', '13', '20', '23']}}
      test_dict['AVN'] = {'ct': ['23'], 'mr': ['23']}

    if 'NMDID' in dataset_name_list:

      patient_dict['NMDID'] = {'ct': ['100131', '100625', '101289', '101607', '101822', '102436', '102846', '105710', '106665', '107024', '108185']}
      vol_range_dict['NMDID'] = {'ct': {'full':['100131', '100625', '101289', '101607', '101822', '102436', '102846', '105710', '106665', '107024', '108185']}}
      slice_center_dict['NMDID'] = {'ct': {'2/3':['100131', '100625', '101289', '101607', '101822', '102436', '105710'], '1/2':[ '102846', '106665', '107024', '108185']}}
      test_dict['NMDID'] = {}

    # pp_vols = PP_Vols(dataset_name_list, result_path,
    #                   reg_size={'ct':[320,320,3], 'mr':[290,290,3]},
    #                   threshold_range=threshold_range,
    #                   format='jpg',
    #                   pp_folder_name=pp_folder_name,
    #                   n_imgs={'mr':{'train':3000, 'test': 50}, 'ct':{'train':3000, 'test': 50}},
    #                   slice_std = {'ct': 12, 'mr': 4},
    #                   roi_thickness=1,
    #                   is_save_preview=False)

    pp_vols = PP_Vols(dataset_name_list, result_path,
                      reg_size=reg_size,
                      threshold_range=threshold_range,
                      format=pp_format,
                      pp_folder_name=pp_folder_name,
                      n_imgs=n_imgs,
                      slice_std=slice_std,
                      roi_thickness=roi_thickness,
                      test_dict=test_dict,
                      is_save_preview=False)

    for dataset_name, dataset_info in patient_dict.items():

      # Get dataset path
      dataset_path = dataset_path_dict[dataset_name]
      for m, patient_id_list in dataset_info.items():
        for patient_id in patient_id_list:
          cur_path = os.path.join(dataset_path, patient_id)
          img_path = glob.glob(os.path.join(cur_path, modality_filename_dict[m]))[0]
          #print(dataset_name + '_' + patient_id)
          vol_range, slice_center = util.get_vol_info(vol_range_dict, slice_center_dict, dataset_name, m, patient_id)
          pp_vol = PP_Vol(dataset_name + '_' + patient_id, m, img_path, vol_range, slice_center)
          pp_vols.add_vol(pp_vol)

    pp_vols.generate_pp_imgs(is_preview_image=False)