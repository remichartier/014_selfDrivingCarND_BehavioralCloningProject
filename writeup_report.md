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
[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 to show model trained allows car to drive autonomously on 1st track of the simulator 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive_01.py model.h5
```

#### 3. Submission code is usable and readable

The model_v14.py and commonFunctions_v13.py files contain the code for training and saving the convolution neural network. More specifically, file model_v14.py shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I followed the Nvidia model as mentioned in the project courses, as it was said to be working for this kind of project. This Nvidia model is described in this paper : [https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars).

The model is implemented in model_v14.py lines 88 to 114.

I took the general architecture of the model and adapted to our purpose here. Following this achitecture : 

![alt text][image8]

This model consists of a 5 successive convolution neural network layers, 3 first CNN layers with 5x5 filter size, 2 last CNN layers with 3x3 filter sizes, each CNN layer having different depths : 24, 36, 48, 64, 64 (model_v14.py lines 94-99).

The CNN layers are followed by a flatten layer, and then 3 fully-connected layers reduce number of neurons from 100 to 50 to 10 and then to 1 neuron representing the steering control value of the car (model_v14.py lines 104-114). 

The model includes several RELU layers to introduce nonlinearity (code lines 103,107,110,113) within the fully-connected layers, and the data is normalized in the model using a Keras lambda layer (code line 89).

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

The other method used to prevent overfitting is to run several epochs, save each epoch model, and select the model for the epoch in which the loss and validation loss are trending down and not up.

- Saving each model is done via using callback in the model.fit_generator(), cf model_v14.py lines 134 and 140.
- One such exemple of training with many epochs is shown below. At the end, I selected model parameters built after epoch #09 to reduce overfitting.

![alt text][image9]

My current submission for this project only train and validate the model based on the sample data provided for this project and is only for a proof of concept.

However, in the course of testing the project, I developped also a system to train and validate on several other additional data inputs (images + steering values), cf model_v14.py lines 45-66.

I used those thouroughfully to train and test different scenarios, however I was not seing improvements due to another issue I had not identified at that time. After finding this issue and fixing it, I just relied on the basic sample data provided with the project to validate the model and code to prove this concept is working, that is why the other data I collected is not used for this submission, but could be if I had more time to play with it. It could be used as well to prevent overfitting on the basic sample data.

But even with sample data provided with the project, the model was trained and validated, and chosen to not overfit by selecting parameters trained on epoch #09. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track (cf the provided video.mp4).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model_v14.py line 118).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
