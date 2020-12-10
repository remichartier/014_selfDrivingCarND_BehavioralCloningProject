import os
import csv
import cv2
import numpy as np # for np.array() np.append()
from datetime import datetime # print timestamps


# for loss history visualization image
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy import ndimage

# list of common functions
# from commonFunctions_vxx import get_log_path
# from commonFunctions_vxx import get_lines_logfile 
# from commonFunctions_vxx import get_info_from_lines
# from commonFunctions_vxx import get_info_from_logfile
# from commonFunctions_vxx import flip_horizontally
# from commonFunctions_vxx import visualize_loss_history
# from commonFunctions_vxx import RGB2YUV
# from commonFunctions_vxx import print_info
# from commonFunctions_vxx import

# History
# v01 : Start
# v02 : add nb_images to read parameter
# v03 : from scipy import ndimage, due to cv2.imread will get images in BGR format, while drive.py uses RGB. In the video above one way you could keep the same image formatting is to do "image = ndimage.imread(current_path)"
# v04 : use left + right images to augment data + measurements extrapolations
# v05 : add Generator Function + modify all other functions whenever necessary to use generator function ...
# v06 : Re-start from v04 aas fit_generator and need to add generator obsolete.
#       Latest Keras.Model.fit integrates a generator in itself.
#        ie v06 : visualize loss history
# v07 : For nvidia model, convert RGB to YUV
# v08 : add print_info() to print debug/progress info
# v09 : Try to avoid list to numpy conversion, taking few minutes, start with numpy image array straight from start
#       But failed. Need to use and adapt to Generator
# v10 : adapt to generator_02.py


driving_log_file = 'driving_log.csv'

# Select right sample data folder whether in GPU mode or not
# check if ./data/driving_log.csv exists otherwise select 
# simulationData/001_1stTrackSampleDrivingData/
def get_log_path() :
    if os.path.exists("./data/" + driving_log_file) :
        return("./data/")
    else : 
        return("./simulationData/001_1stTrackSampleDrivingData/")


def get_lines_logfile() :
    l = []
    with open (get_log_path() + driving_log_file ) as csv_file :
        reader = csv.reader(csv_file)
        for line in reader :
            l.append(line)
    return l

 
def get_info_from_lines(l,leftright_steer_corr,nb_images=None) :
    imgs = []
    meas = []
    
    log_path = get_log_path()    
    
    for line in l[:nb_images] :
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
    return imgs,meas

def get_info_from_logfile(leftright_steer_correction,nb_images=None) :
    lines = get_lines_logfile()
    return get_info_from_lines(lines,leftright_steer_correction,nb_images)

def flip_horizontally(img,meas):
    aug_img, aug_meas = [],[]
    for i,m in zip(img,meas) :
        aug_img.append(cv2.flip(i,1))
        aug_meas.append(m*(-1.0))
    return aug_img,aug_meas


def visualize_loss_history(history) :
    ### plot the training and validation loss for each epoch
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    # plt.show()
    plt.savefig('lossHistory.png')

def RGB2YUV(im):
    yuv = []
    for i in im :
        yuv.append(cv2.cvtColor(i, cv2.COLOR_RGB2YUV))
    return yuv

def print_info(info):
    now = datetime.now()
    infotime = now.strftime("%H:%M:%S")
    # can not use f-string due to GPU python version v3.5.2
    print('{}. Time : {}'.format(info,infotime))
    
                        