import os
import csv


# list of common functions
# from commonFunctions_vxx import get_log_file()
# from commonFunctions_vxx import get_lines_logfile() 
# from commonFunctions_vxx import get_info_from_lines()
# from commonFunctions_vxx import get_info_from_logfile()
# from commonFunctions_vxx import

driving_log_file = 'driving_log.csv'

# Select right sample data folder whether in GPU mode or not
# check if ./data/driving_log.csv exists otherwise select 
# simulationData/001_1stTrackSampleDrivingData/
def get_log_file() :
    if os.path.exists("./data/" + driving_log_file) :
        return("./data/" + driving_log_file)
    else : 
        return("./simulationData/001_1stTrackSampleDrivingData/" + driving_log_file)


def get_lines_logfile() :
    l = []
    with open get_log_file() as csv_file :
        reader = csv.reader(csv_file)
        for line in reader :
            l.append(line)
    return l


def get_info_from_lines(l) :
    imgs = []
    meas = []
    for line in l :
        image = cv2.imread(line[0])
        imgs.append(image)
        measurement = float(line[3])
        meas.append(measurement)
    return imgs,meas

def get_info_from_logfile() :
    lines = get_lines_logfile()
    return get_info_from_lines(lines)