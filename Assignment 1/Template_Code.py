import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import csv
import math
import os
import sys

class FrameCalib:
    """Frame Calibration

    Fields:
        p0-p3: (3, 4) Camera P matrices for cameras 0-3. Combines extrinsic and intrinsic parameters.
        r0_rect: (3, 3) Rectification matrix
        velo_to_cam: (3, 4) Transformation matrix from velodyne to cam coordinates
        Point_Camera = p_cam * r0_rect * Tr_velo_to_cam * Point_Velodyne
        """

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.velo_to_cam = []


def read_frame_calib(calib_file_path):
    """Reads the calibration file for a sample

    Args:
        calib_file_path: calibration file path

    Returns:
        frame_calib: FrameCalib frame calibration
    """

    data_file = open(calib_file_path, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    frame_calib = FrameCalib()
    frame_calib.p0 = p_all[0]
    frame_calib.p1 = p_all[1]
    frame_calib.p2 = p_all[2]
    frame_calib.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calib.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calib.velo_to_cam = np.reshape(tr_v2c, (3, 4))

    return frame_calib


class StereoCalib:
    """Stereo Calibration

    Fields:
        baseline: distance between the two camera centers
        f: focal length
        k: (3, 3) intrinsic calibration matrix
        p: (3, 4) camera projection matrix
        center_u: camera origin u coordinate
        center_v: camera origin v coordinate
        """

    def __init__(self):
        self.baseline = 0.0
        self.f = 0.0
        self.k = []
        self.center_u = 0.0
        self.center_v = 0.0


def krt_from_p(p, fsign=1):
    """Factorize the projection matrix P as P=K*[R;t]
    and enforce the sign of the focal length to be fsign.


    Keyword Arguments:
    ------------------
    p : 3x4 list
        Camera Matrix.

    fsign : int
            Sign of the focal length.


    Returns:
    --------
    k : 3x3 list
        Intrinsic calibration matrix.

    r : 3x3 list
        Extrinsic rotation matrix.

    t : 1x3 list
        Extrinsic translation.
    """
    s = p[0:3, 3]
    q = np.linalg.inv(p[0:3, 0:3])
    u, b = np.linalg.qr(q)
    sgn = np.sign(b[2, 2])
    b = b * sgn
    s = s * sgn

    # If the focal length has wrong sign, change it
    # and change rotation matrix accordingly.
    if fsign * b[0, 0] < 0:
        e = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    if fsign * b[2, 2] < 0:
        e = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    # If u is not a rotation matrix, fix it by flipping the sign.
    if np.linalg.det(u) < 0:
        u = -u
        s = -s

    r = np.matrix.transpose(u)
    t = np.matmul(b, s)
    k = np.linalg.inv(b)
    k = k / k[2, 2]

    # Sanity checks to ensure factorization is correct
    if np.linalg.det(r) < 0:
        print('Warning: R is not a rotation matrix.')

    if k[2, 2] < 0:
        print('Warning: K has a wrong sign.')

    return k, r, t


def get_stereo_calibration(left_cam_mat, right_cam_mat):
    """Extract parameters required to transform disparity image to 3D point
    cloud.

    Keyword Arguments:
    ------------------
    left_cam_mat : 3x4 list
                   Left Camera Matrix.

    right_cam_mat : 3x4 list
                   Right Camera Matrix.


    Returns:
    --------
    stereo_calibration_info : Instance of StereoCalibrationData class
                              Placeholder for stereo calibration parameters.
    """

    stereo_calib = StereoCalib()
    k_left, r_left, t_left = krt_from_p(left_cam_mat)
    _, _, t_right = krt_from_p(right_cam_mat)

    stereo_calib.baseline = abs(t_left[0] - t_right[0])
    stereo_calib.f = k_left[0, 0]
    stereo_calib.k = k_left
    stereo_calib.center_u = k_left[0, 2]
    stereo_calib.center_v = k_left[1, 2]

    return stereo_calib


def get_disparity_list(pixel_u_list, pixel_v_list, pixel_right_u_list):

    disparity_list = []
    for i in range(len(pixel_u_list)):
        disparity_list.append(abs(pixel_u_list[i]-pixel_right_u_list[i]))
                
    return disparity_list

def get_depth_list(disparity_list, stereo_calib):

    # Replace all instances of 0 disparity with a small minimum value (to avoid div by 0 or negatives)
    disparity_list = [0.1 if x == 0 else x for x in disparity_list]
 

    # Initialize the depth map to match the size of the disparity map
    depth_list = np.zeros(len(disparity_list))  

    # Calculate the depths 
    depth_list[:] = stereo_calib.f  * stereo_calib.baseline / disparity_list[:]
   
    return depth_list

def cal_correct_rate(depth_list, depth, pixel_u_list, pixel_v_list, sample_name):

    gd_depth_list = []

    correct_count = 0
    overall_count = len(pixel_u_list)

    for i in range(overall_count):
        try:
            gd_depth_list.append(depth[math.floor(pixel_v_list[i])][math.floor(pixel_u_list[i])])
        except Exception as e:
            print(e)

    # compare generated depth list with groud truth depth list 

    for i in range(overall_count):
        if(abs(gd_depth_list[i]-depth_list[i])<=5):
            correct_count += 1

    print("{} : overall Count {}  Correct count:{} correct rate:{} ".format(sample_name, overall_count, correct_count, correct_count/overall_count))

    return correct_count/overall_count



if __name__ == '__main__':

    if(len(sys.argv) <= 1):
        print("Please choose test or training mode to run the program")
        sys.exit()

    if(sys.argv[1] == "test"):
        print("Running test mode: Result will be in P3_result.txt")

    # Input
        left_image_dir = 'test/left'
        right_image_dir = 'test/right'
        calib_dir = 'test/calib'
        sample_list = ['000011', '000012', '000013', '000014','000015']

        ## Output
        output_file = open("P3_result.txt", "a")
        output_file.truncate(0)


    if(sys.argv[1] == "training"):

        print("Running training mode: Result will be in Train_result.txt")

        left_image_dir = 'training/left'
        right_image_dir = 'training/right'
        calib_dir = 'training/calib'
        sample_list = ['000001', '000002', '000003', '000004','000005','000006','000007',
        '000008','000009','000010']

        ## Output
        output_file = open("Train_result.txt", "a")
        output_file.truncate(0)
        depth_dir = "training/gt_depth_map"

    
    ## Main

    total_count_rate = 0
    for sample_name in sample_list:

        if(sys.argv[1] == "training"):
            left_depth = cv.imread(os.path.join(depth_dir,  sample_name+".png" ), cv.IMREAD_ANYDEPTH )
            depth = left_depth/256

        #load images
        img_left = cv.imread(os.path.join(left_image_dir, sample_name + ".png" ))
        img_right = cv.imread(os.path.join(right_image_dir, sample_name + ".png" ))
        
        # create a feature detector
        # choose SURF detector 
        
        hessianThreshold = 400 
        surf = cv.xfeatures2d.SURF_create(hessianThreshold)
        raw_kp_left, raw_des_left = surf.detectAndCompute(img_left,None)
        raw_kp_right, raw_des_right = surf.detectAndCompute(img_right,None)

        # orb = cv.ORB_create(nfeatures = 1000)
        # raw_kp_left, raw_des_left = orb.detectAndCompute(img_left,None)
        # raw_kp_right, raw_des_right = orb.detectAndCompute(img_right,None)

    
        # find the keypoints: sort accoridng to keypoint response, the greater the better 

        kp_left,des_left =zip(*sorted(zip(raw_kp_left, raw_des_left), key=lambda pair: pair[0].response,reverse = True))
        kp_right,des_right =zip(*sorted(zip(raw_kp_right, raw_des_right), key=lambda pair: pair[0].response,reverse = True))

        #find the keypoints
        kp_left = kp_left[:1000]
        kp_right = kp_right[:1000]
        
        # compute the descriptors
        des_left = np.array(des_left)[:1000]
        des_right = np.array(des_right)[:1000]

        # display left images 
        keypoint_image = cv.drawKeypoints(img_left, kp_left, None)
        plt.imshow(keypoint_image)
        plt.show()

        # create a matcher

        bf = cv.BFMatcher_create()
        # match descriptors
        matches = bf.knnMatch(des_left,des_right, k=2)  

        # use ratio test to remove bad match 
        good_matches = []
        ratio = 0.6
        for m,n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
        
        print("{} ORIGINAL SURF MATCH with hessianThreshold{} PLUS RATIO TEST {} HAS {} matching".format(sample_name, hessianThreshold, ratio, len(good_matches)))

        #PART2 : remove non-horizontal matching 

        horizontal_matches = []
        points_left = []
        points_right = []
        before_pixel_u_list = []
        before_pixel_v_list = []
        before_pixel_u_right_list = []

        for match in good_matches: 

            img_left_idx = match.queryIdx
            img_right_idx = match.trainIdx

            x_left,y_left = kp_left[img_left_idx].pt
            x_right,y_right = kp_right[img_right_idx].pt

            if(math.floor(y_left) == math.floor(y_right)):
                horizontal_matches.append(match)

            points_left.append(kp_left[img_left_idx].pt)
            points_right.append(kp_right[img_right_idx].pt)
            before_pixel_u_list.append(x_left)
            before_pixel_v_list.append(y_left)
            before_pixel_u_right_list.append(x_right)



        print("{} HORIZONTAL MATCH HAS {} matching".format(sample_name,len(horizontal_matches)))
        # ====================Draw graph for horizontal match !!!
        horizontal_img = cv.drawMatches(img_left,kp_left,img_right,kp_right,horizontal_matches, None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow( horizontal_img )
        plt.show()

        # ====================Draw graph for raw match !!!
        raw_match_img = cv.drawMatches(img_left,kp_left,img_right,kp_right,good_matches, None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow( raw_match_img )
        plt.show()


        #PART3 : oitlier rejection 
        #use fundamental matrix with RANSAC algorithm
        img_points_left = np.array(points_left, dtype=np.int32)
        img_points_right = np.array(points_right, dtype=np.int32)


        ransacReprojThreshold = 4
        confidence = 0.99
        F, mask = cv.findFundamentalMat(img_points_left, img_points_right, method= cv.FM_8POINT+cv.FM_RANSAC, ransacReprojThreshold=ransacReprojThreshold, confidence=confidence)

        filtered_matches = []
        for i in range(len(good_matches)):
            if mask[i] == 1:
                filtered_matches.append(good_matches[i])
        
        print("{} FILTERED MATCH HAS {} matching with ransacReprojThreshold:{} and confidence:{} ".format(sample_name,len(filtered_matches), ransacReprojThreshold, confidence))
        # ====================Draw graph for horizontal match !!!


        # read calibration
        frame_calib = read_frame_calib(os.path.join(calib_dir, sample_name + ".txt" ))
        # what's the left_cam_mat and right_cam_mat 
        stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # compute disparity and depth

        pixel_u_list = []
        pixel_v_list = []
        pixel_right_u_list = []
        pixel_right_v_list = []

        for match in filtered_matches:

            img_left_idx = match.queryIdx
            img_right_idx = match.trainIdx

            x_left,y_left = kp_left[img_left_idx].pt
            x_right,y_right = kp_right[img_right_idx].pt

            pixel_u_list.append(x_left)
            pixel_v_list.append(y_left)
            pixel_right_u_list.append(x_right)
            pixel_right_v_list.append(y_right)

        # print("First in the list L{} R{}".format(pixel_u_list[0], pixel_right_u_list[0]))

        # second feature mapping 

        # after filter 
        disparity_list = get_disparity_list(pixel_u_list, pixel_v_list, pixel_right_u_list)

        depth_list = get_depth_list(disparity_list,stereo_calib)

        if(sys.argv[1] == "training"):

            correct_rate = cal_correct_rate(depth_list, depth, pixel_u_list, pixel_v_list, sample_name)

            total_count_rate += correct_rate        

            
        # save output
        for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
            line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
            output_file.write(line + '\n')

        
        final_match_img = cv.drawMatches(img_left,kp_left,img_right,kp_right,filtered_matches, None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(final_match_img)
        plt.show()

    output_file.close()
    if(sys.argv[1] == "training"):
        print("avg correct rate {}".format(total_count_rate/len(sample_list)))



