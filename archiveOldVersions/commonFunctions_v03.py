import os
import csv
import cv2

from scipy import ndimage

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


def get_info_from_lines(l,nb_images=None) :
    imgs = []
    meas = []
    log_path = get_log_path()
    print('Function get_info_from_lines() : Load images ... Please wait ....')
    for line in l[1:nb_images] :
        #image = cv2.imread(log_path + line[0].strip())
        image = ndimage.imread(log_path + line[0].strip())
        imgs.append(image)
        measurement = float(line[3])
        meas.append(measurement)
    print('Function get_info_from_lines() : Images loaded')
    return imgs,meas

def get_info_from_logfile(nb_images=None) :
    lines = get_lines_logfile()
    return get_info_from_lines(lines,nb_images)

def flip_horizontally(img,meas):
    aug_img, aug_meas = [],[]
    for i,m in zip(img,meas) :
        aug_img.append(cv2.flip(i,1))
        aug_meas.append(m*(-1.0))
    return aug_img,aug_meas
