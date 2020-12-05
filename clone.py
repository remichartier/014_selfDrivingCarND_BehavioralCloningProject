import numpy as np
import os

driving_log_file = 'driving_log.csv'

# Select right sample data folder whether in GPU mode or not
# check if ./data/driving_log.csv exists otherwise select 
# simulationData/001_1stTrackSampleDrivingData/
def get_log_file() :
    if os.path.exists("./data/driving_log.csv") :
        return "./data/driving_log.csv"
    else : 
        return "./simulationData/001_1stTrackSampleDrivingData/driving_log.csv"
    