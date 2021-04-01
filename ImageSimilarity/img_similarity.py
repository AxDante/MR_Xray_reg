################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################

import argparse
import os
import util
import cv2
import csv
import metrics
import glob
#from skimage.metrics import structural_similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-d0','--dir0', type=str, default='./imgs/test01/ex_dir0')
    # parser.add_argument('-d1','--dir1', type=str, default='./imgs/test01/ex_dir1')
    # parser.add_argument('-dataroot','--dir1', type=str, default='./imgs/test01/ex_dir1')
    # parser.add_argument('-o','--dir_out', type=str, default='./result/test01/')
    parser.add_argument('-v','--version', type=str, default='0.1')
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

    parser.add_argument('--result_dir', type=str, default='../test_results')
    parser.add_argument('--name', type=str, default='ex01')


    opt = parser.parse_args()

    field_names = ['Patient_ID', 'MSE', 'AE', 'PSNR', 'SSIM', 'LPIPS']

    run_path = opt.result_dir + '/' + opt.name + '/' + opt.name
    result_file_path =  run_path+ '/sim_result.csv'
    img_dir = run_path + '/test_latest/images/'

    with open(result_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()

        real_A_list = glob.glob(img_dir + '*_real_A.png')
        real_B_list = glob.glob(img_dir + '*_real_B.png')

        #files = os.listdir(opt.dir0)
        #files = os.listdir(opt.dir0)

        for real_file in real_A_list:

            fake_file = real_file.replace('real_A', 'rec_A')

            if(os.path.exists(fake_file)):

                f_dict = {fn: '-' for fn in field_names}
                f_dict['Patient_ID'] = real_file

                # Load images
                img0 = util.load_image(real_file)
                img1 = util.load_image(fake_file)

                img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

                img0_tensor = util.np2tensor(util.normalize_img(img0))
                img1_tensor = util.np2tensor(util.normalize_img(img1))

                if(opt.use_gpu):
                    img0_tensor = img0_tensor.cuda()
                    img1_tensor = img1_tensor.cuda()

                # Compute distance
                metric = metrics.MSE()
                f_dict['MSE'] = metric(img0_tensor, img1_tensor).item()
                metric = metrics.PSNR()
                f_dict['PSNR'] = metric(img0_tensor, img1_tensor).item()
                metric = metrics.SSIM()
                f_dict['SSIM'] = metric(img0_tensor, img1_tensor).item()
                metric = metrics.LPIPS(opt.use_gpu)
                f_dict['LPIPS'] = metric(img0_tensor, img1_tensor).item()
                metric = metrics.AE()
                f_dict['AE'] = metric(img0_tensor, img1_tensor).item()

                print(file)
                print('MSE : {:.3f}'.format(f_dict['MSE']))
                print('AE : {:.3f}'.format(f_dict['AE']))
                print('PSNR : {:.3f}'.format(f_dict['PSNR']))
                print('SSIM : {:.3f}'.format(f_dict['SSIM']))
                print('LPIPS : {:.3f}'.format(f_dict['LPIPS']))
                print('-'*8)
                writer.writerow(f_dict)
