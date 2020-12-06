import os
import csv
import cv2


# list of common functions
# from commonFunctions_vxx import get_log_path()
# from commonFunctions_vxx import get_lines_logfile() 
# from commonFunctions_vxx import get_info_from_lines()
# from commonFunctions_vxx import get_info_from_logfile()
# from commonFunctions_vxx import

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
    imgs = []
    meas = []
    log_path = get_log_path()
    print('Function get_info_from_lines() : Load images ... Please wait ....')
    for line in l[1:] :
        image = cv2.imread(log_path + line[0])
        imgs.append(image)
        measurement = float(line[3])
        meas.append(measurement)
    im = imgs[0]
    # print(f'image type : {type(im)}')
    print('Function get_info_from_lines() : Images loaded')
    return imgs,meas

def get_info_from_logfile() :
    lines = get_lines_logfile()
    return get_info_from_lines(lines)