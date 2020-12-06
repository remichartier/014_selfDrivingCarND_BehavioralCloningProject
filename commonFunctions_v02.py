import os
import csv
import cv2
import numpy as np



# list of common functions
# from commonFunctions_vxx import get_log_path
# from commonFunctions_vxx import get_lines_logfile 
# from commonFunctions_vxx import get_info_from_lines
# from commonFunctions_vxx import get_info_from_logfile
# from commonFunctions_vxx import

# History
# v01 : start
# v02 : try use only np.arrays, to skip conversion list/np.arrays


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


def get_info_from_lines(l) :
    imgs = np.array([])
    meas = np.array([])
    log_path = get_log_path()
    print('Function get_info_from_lines() : Load images ... Please wait ....')
    for line in l[1:] :
        image = cv2.imread(log_path + line[0].strip())
        imgs = np.append(imgs,image)
        measurement = float(line[3])
        meas = np.append(meas,measurement)
    print('Function get_info_from_lines() : Images loaded')
    print(f'imgs : {imgs.shape}, meas : {meas.shape}')
    return imgs,meas

def get_info_from_logfile() :
    lines = get_lines_logfile()
    return get_info_from_lines(lines)