# **Behavioral Cloning** 

## Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image8]: ./writeupReportMaterials/Nvidia_cnn-architecture-624x890.png "NVidia Model Visualization"
[image9]: ./writeupReportMaterials/epochsTrainingExemple.png "Epochs Training"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_v14.py containing the script to create and train the model
* drive_01.py for driving the car in autonomous mode
* commonFunctions_v13.py for gathering all the functions needed by model_v14.py including generator function. 
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 to show model trained allows car to drive autonomously on 1st track of the simulator 

#### 2. Submission includes functional code
Using the Udacity provided simulator with my drive_01.py file and the model.h5 file, the car can be driven autonomously around the track by executing 
```
python drive_01.py model.h5
```
- It can drive laps of track 01 without going out of the road, video.mp4 shows a little more that one full lap with car staying on the road.

#### 3. Submission code is usable and readable

The model_v14.py and commonFunctions_v13.py files contain the code for training and saving the convolution neural network. 

More specifically, file model_v14.py shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Model_v14.py with help of commonFunctions_v13.py, use a python generator to generate data for training rather than storing the training data in memory.
- Done in commonFunctions_V13.py, function generator() lines 147-174, called from model_v14.py lines 76-77 to define train and validation generators used in model.fit_generator() lines 136-140. 
- I experimented first loading all images in one list, and then converting this list in an Numpy array. Taking Sample Data, it was about 8000 center images, mutiplied by 3 to include left and right images --> ~24000 images, multiply by 2 when adding flipping images --> 48000 images. And it was taking around 3 to 4 minutes to convert this list of ~48000 images to Numpy arrays for X_train and y_train. That's how I realized using a generator would either save this converstion time or spread it into the training process so that would would become transparent in term of processing time. Not yet even considering the benefit of using a generator to reduce memory size needed to held ~48000 images in memory.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I followed the Nvidia model as mentioned in the project courses, as it was said to be working for this kind of project. This Nvidia model is described in this paper : [https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars).

The model is implemented in model_v14.py lines 88 to 114.

I took the general architecture of the model and adapted to our purpose here. Following this achitecture : 

![alt text][image8]

This model consists of a 5 successive convolution neural network layers, 3 first CNN layers with 5x5 filter size, 2 last CNN layers with 3x3 filter sizes, each CNN layer having different depths : 24, 36, 48, 64, 64 (model_v14.py lines 94-99).

The CNN layers are followed by a flatten layer, and then 3 fully-connected layers reduce number of neurons from 100 to 50 to 10 and then to 1 neuron representing the steering control value of the car (model_v14.py lines 104-114). 

The model includes several RELU layers to introduce nonlinearity (code lines 103,107,110,113) within the fully-connected layers.

The data is normalized in the model using a Keras lambda layer (code line 89), reducing values of pixels ranging [0;255] to range [-0.5;0.5] in order to have data mean zero centered which helps training converge faster to optimzed model and parameters.

In terms of pre-processing, I also followed recommendations both from the Project course introduction as well as from the Nvidia model used, and those steps consist of : 

- While training and testing the model :
  - Taking each image coming via cv2.imread() in BGR format 
  - Converting to YUV format which is the input format for the Nvidia model (using function BGR2YUV() line 167 generator() function in commonFunctions_v13.py
  - Cropping the top and bottom of images to exclude the hood of the car and the horizon which will help focus the training the road elements and discard anything above the road level. This cropping processing is fully integrated in the Keras model pipeline (model_v14.py line 90)
- While simulating the model with the simulator :
  - Taking each image coming from the simulator in RGB format.
  - converting to YUV format to be in the same format as the training was done for the Nvidia model, this conversion RGB to YUV is done in drive_01.py line 72.
  

#### 2. Attempts to reduce overfitting in the model

The model contains several dropout layers in order to reduce overfitting (model_v14.py lines 102,106,109,112).

I am spliting data after shuffling all data input, between training data and validation data (train/validation split of 0.2, model_v14.py line 69) in order to measure over/under fitting of the data via model.fit_generator() line 136.

The other method used to prevent overfitting is to run several epochs, save each epoch model, and select the model in which the epoch  loss and validation loss are trending down and not up.

- Saving each model is done via using callback in the model.fit_generator(), cf model_v14.py lines 134 and 140.
- One such exemple of training with many epochs is shown below. At the end, I selected model parameters built after epoch #09 as an additional way to reduce overfitting.

![alt text][image9]

My current submission for this project only trains and validates the model based on the sample data provided for this project and is only for a proof of concept.

- However, in the course of testing the project, I developped also a system to train and validate on several other additional data inputs (images + steering values) as extra means to reduce overfitting, cf model_v14.py lines 45-66.
- I used those additional data thouroughfully to train and test different scenarios, however I was not seing improvements due to another issue I had not identified at that time. After finding this issue and fixing it, I just relied on the basic sample data provided with the project to validate the model and code to prove this concept is working. That is why the additional data I collected is not used for this submission, but could be if I had more time to play with it. It could be used as well to prevent overfitting on the basic sample data.

But even with sample data provided with the project, the model was trained and validated, and chosen to not overfit by selecting parameters trained on epoch #09. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track (cf the provided video.mp4).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model_v14.py line 118).

#### 4. Appropriate training data

As mentioned above, I collected myself additional training data in an attempt to keep the vehicle driving on the road. 

However, after fixing some pre-processing miss in drive_v01.py, I realized that just using the sample data provided with the project was enough for the model I implemented to allow the car to drive autonomously on the track without running out of the road.

But if I had to train more in order to reduce overfitting, I could use this extra collected data. I used a combination of : 
- Center lane driving both anti-clockwise and clockwise.
- Collected data from recovering from the left and right sides of the road to the center. 
- Collected data by running smoothly on turns/curves to improve car behavior whild driving toward curves. 

All this additional data to reduce overfitting was used at some points to train and validate the model while I was having issues to keep on the road.
- code in model_v14.py lines 45-66.
- However I commented it out after fixing pre-processing issues (conversions RGB to YUV and BGR to YUV) as just using the Sample Data was enough and I had struggled too much to find a good model keeping the car in the road that I stopped there and did not have time to take additional risks to test those overfitting solutions further before submitting the project and moving on to progress on the nanodegree.

I also used 2 strategies suggested in the project course to reduce overfitting by augmenting the data already at hand : 

1. The driving Simulator provides center image, left and right image, and steering for the center image.
  - I added both left and right image to the training, with a steering correction of +/- 0.2 vs center image steering value. (code in commonFunctions_v13.py lines 92-103, and 158).
2 I flipped all images (center/left/right) and inversed corresponging steering values associate with each image, as a way to augment data to reduce overfitting. (code in commonFunctions_v13.py lines 110-115 and model_v14.py lines 160-162)

#### 5. Creation of training dataset and training process documented 
How the model was trained, what the characteristics of the dataset are. Info how the dataset was generated , exemples of images from the dataset must be included.

- saving model.
- learning transfer model_V14.py line 129.
- mistakes done.
- batch size.
- 
