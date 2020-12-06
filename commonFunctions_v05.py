import os
import csv
import numpy as np
import cv2
import sklearn # for utils.shuffle()


from scipy import ndimage
from sklearn.model_selection import train_test_split
from random import shuffle


# list of common functions
# from commonFunctions_vxx import get_log_path
# from commonFunctions_vxx import get_lines_logfile 
# from commonFunctions_vxx import get_info_from_lines
# from commonFunctions_vxx import get_info_from_logfile
# from commonFunctions_v03 import flip_horizontally
# from commonFunctions_vxx import

# History
# v01 : Start
# v02 : add nb_images to read parameter
# v03 : from scipy import ndimage, due to cv2.imread will get images in BGR format, while drive.py uses RGB. In the video above one way you could keep the same image formatting is to do "image = ndimage.imread(current_path)"
# v04 : use left + right images to augment data + measurements extrapolations
# v05 : add Generator Function + modify all other functions whenever necessary to use generator function ...

driving_log_file = 'driving_log.csv'

# Select right sample data folder whether in GPU mode or not
# check if ./data/driving_log.csv exists otherwise select 
# simulationData/001_1stTrackSampleDrivingData/
def get_log_path() :
    if os.path.exists("./data/" + driving_log_file) :
        return("./data/")
    else : 
        return("./simulationData/001_1stTrackSampleDrivingData/")


def get_lines_logfile(nb_lines=None) :
    l = []
    with open (get_log_path() + driving_log_file ) as csv_file :
        reader = csv.reader(csv_file)
        for line in reader :
            l.append(line)
    return l[:nb_lines]

# if possible get path in input parameter to avoid calling this repeatedly ; log_path = get_log_path()
def get_info_from_lines(l,leftright_steer_corr,log_path,nb_images=None) :
    imgs = []
    meas = []
    # log_path = get_log_path()
    #print('Function get_info_from_lines() : Load images ... Please wait ....')
    for line in l[1:nb_images] :
        #image = cv2.imread(log_path + line[0].strip())
        for i in range(3) :         # center image, left , right images
            image = ndimage.imread(log_path + line[i].strip())
            imgs.append(image)
        measurement = float(line[3]) # center image
        meas.append(measurement)
        measurement += leftright_steer_corr # left image
        meas.append(measurement)
        measurement -= leftright_steer_corr # right image
        meas.append(measurement)
    #print('Function get_info_from_lines() : Images loaded')
    return imgs,meas

def get_info_from_logfile(leftright_steer_correction,nb_images=None,test_size=0.2) :
    lines = get_lines_logfile()
    log_folder = get_log_path()
    return get_info_from_lines(lines,leftright_steer_correction,log_folder,nb_images)

def flip_horizontally(img,meas):
    aug_img, aug_meas = [],[]
    for i,m in zip(img,meas) :
        aug_img.append(cv2.flip(i,1))
        aug_meas.append(m*(-1.0))
    return aug_img,aug_meas

def generator(samples, leftright_steer_corr, batch_size=32):
    # notes : taking lines from 'driving_log.csv' in input ...
    num_samples = len(samples)
    log_folder = get_log_path()
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):        
            batch_samples = samples[offset:min(offset+batch_size,num_samples)]  # added correction here (min())

            # we have this already : def get_info_from_lines(l,leftright_steer_corr,nb_images=None)
            images = []
            angles = []
            '''
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            '''            

            images,angles = get_info_from_lines(batch_samples,leftright_steer_corr,log_folder,nb_images=None)
            #print(f'Generator : images list size : {len(images)}, each image shape : {images[0].shape}, y_train size {len(angles)}')
            
            # data augmentation flip horizontally image + inverse measurements
            augm_images, augm_angles = flip_horizontally(images,angles)
            images.extend(augm_images)
            angles.extend(augm_angles)
            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #print(f'Generator : X_train {X_train.shape}, y_train {y_train.shape}')
            yield sklearn.utils.shuffle(X_train, y_train)
