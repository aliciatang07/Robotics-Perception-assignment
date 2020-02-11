import os
import sys
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import kitti_dataHandler


def get_depth_map(disparity_map, stereo_calib):

    # Replace all instances of 0 disparity with 0.1 to avoid error 
    disparity_map[disparity_map == 0] = 0.1
    disparity_map[disparity_map == -1] = 0.1
 

    # Initialize the depth map to match the size of the disparity map
    depth_map = np.ones(disparity_map.shape, np.single)

    # Calculate the depths 
    depth_map[:] = stereo_calib.f  * stereo_calib.baseline / disparity_map[:]
   
    return depth_map


def main():

    ################
    # Options
    ################
    # train
    if(sys.argv[1] == "train"):
        disp_dir = 'data/train/disparity'
        output_dir = 'data/train/est_depth'
        calib_dir = 'data/train/calib'
        sample_list = ['000001', '000002', '000003', '000004', '000005','000006',
        '000007','000008','000009','000010']
        print("#####TRAINING MODE")

    if(sys.argv[1] == "test"):
    # test    
        disp_dir = 'data/test/disparity'
        output_dir = 'data/test/est_depth'
        calib_dir = 'data/test/calib'
        sample_list = ['000011', '000012', '000013', '000014', '000015']
        print("#####TESTING MODE")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sample_name in (sample_list):
        # output_file = open(os.path.join(output_dir,sample_name + ".txt"), "a")
        # output_file.truncate(0)

        # Read disparity map
        disparity_map = cv.imread(os.path.join(disp_dir, sample_name + ".png"),cv.IMREAD_ANYDEPTH)
        disparity_map = disparity_map/256
        # Read calibration info
        frame_calib = kitti_dataHandler.read_frame_calib(os.path.join(calib_dir, sample_name + ".txt" ))
        stereo_calib = kitti_dataHandler.get_stereo_calibration(frame_calib.p2, frame_calib.p3)
        # Calculate depth (z = f*B/disp)
        depth_map = get_depth_map(disparity_map, stereo_calib)

        # Discard pixels past 80m >80 <0.1
        
        depth_map[depth_map > 80] = 0
        depth_map[depth_map < 0.1] = 0

        # print(depth_map)

        # plt.figure()
        plt.imshow(depth_map, cmap='gray')
        # save depth map 
        cv.imwrite(os.path.join(output_dir,sample_name + ".png"), depth_map) 




if __name__ == '__main__':
    main()