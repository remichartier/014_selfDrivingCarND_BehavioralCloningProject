#!/usr/bin/env python

# History
# v01 : adaptation from model_v14.py to output images for writeup_report.md

import os
import csv
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import random


from math import ceil
from sklearn.model_selection import train_test_split

from commonFunctions_v14 import get_lines_logfile,get_info_from_lines,STEER_CORRECTION_FACTOR
from commonFunctions_v14 import generator,flip_horizontally
from commonFunctions_v14 import batch_len
from commonFunctions_v14 import print_info
from commonFunctions_v14 import visualize_loss_history
from commonFunctions_v14 import get_log_pathSampleData

path_last_hard_turn_data = "./simulationData/002_hardLeftTurn20201208_0220/"
path_003_OwnRecordingOneLapAntiClockwise = "./simulationData/003_OwnRecordingOneLapAntiClockwise/"
path_004_ownRecordOneLapClockwise = "./simulationData/004_ownRecordOneLapClockwise/"
path_005_ownRecordOneLapAntiClockwise = "./simulationData/005_ownRecordOneLapAntiClockwise/"
path_006_OwnRecord2LapsRecoverSidesAntiClockwise = "./simulationData/006_OwnRecord2LapsRecoverSidesAntiClockwise/"
path_007_trainHardTurnSeveralTimes = "./simulationData/007_trainHardTurnSeveralTimes/"

def saveImagesSideBySide(img_list,path,filename) :
    nb_img = len(img_list)
    f, ax = plt.subplots(nrows=1, ncols=nb_img, figsize=(9, 9))
    f.tight_layout()
    for i in range(nb_img) :
        # choose cmap option if 3 color channels or if 1 (grayscale)
        img_cmap = 'gray' if (img_list[i].shape[2] == 1) else None 
        im = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB)
        ax[i].imshow(im.squeeze(),img_cmap)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(path + '/' + filename)
    # cv2.cvtColor(cube, cv2.COLOR_BGR2RGB)

# Reading CSV file FROM SAMPLE DATA, extracting lines.
samples = get_lines_logfile(get_log_pathSampleData())
# Reading CSV file from Last Hard Turn, extracting lines
# add them to samples lines.
#samples.extend(get_lines_logfile(path_last_hard_turn_data))
# Reading CSV file from 003_OwnRecordingOneLapAntiClockwise, extracting lines
# add them to samples lines.
#samples.extend(get_lines_logfile(path_003_OwnRecordingOneLapAntiClockwise))
# Reading CSV file from 004_ownRecordOneLapClockwise, extracting lines
# add them to samples lines.
#samples.extend(get_lines_logfile(path_004_ownRecordOneLapClockwise))
#samples.extend(get_lines_logfile(path_005_ownRecordOneLapAntiClockwise))
#samples.extend(get_lines_logfile(path_006_OwnRecord2LapsRecoverSidesAntiClockwise))
#samples.extend(get_lines_logfile(path_007_trainHardTurnSeveralTimes))

# choose record line randomly.
n = random.randint(1,len(samples)-1) 


images, angles = get_info_from_lines(samples[n:n+1],STEER_CORRECTION_FACTOR,nb_images=None)
# data augmentation flip horizontally image + inverse measurements
augm_images, augm_measurements = flip_horizontally(images,angles)
print('Nb images : {}, Nb angles : {}'.format(len(images),len(angles)))
print('Angles Center, Left, Right images : {}'.format(angles))
print('flipped Angles Center, Left, Right images : {}'.format(augm_measurements))
print('STEER_CORRECTION_FACTOR : {}'.format(STEER_CORRECTION_FACTOR))

saveImagesSideBySide(images,'./writeupReportMaterials','randomImageCenterLeftright.png')
saveImagesSideBySide(augm_images,'./writeupReportMaterials','randomImageCenterLeftrightFLIPPED.png')

# model.add(Cropping2D(cropping=((70,25),(0,0))))

# save picture lossHistory.png
# visualize_loss_history(history_object)
print('Image shape : {}'.format(images[0].shape))
height = images[0].shape[0]
width = images[0].shape[1]
# need to show cropped images
# crop_img = img[y:y+h, x:x+w]
def crop (img_lst,h,w):
    output = []
    for im in img_lst:
        output.append(im[70:h-25,0:w])
    return output

cropped = crop(images,height,width)
saveImagesSideBySide(cropped,'./writeupReportMaterials','randomImageCenterLeftrightCROPPED.png')

