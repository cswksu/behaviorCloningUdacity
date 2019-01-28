# Behavior Cloning Project
## Solution to behavior cloning Udacity SDCND project

The goals / steps of this project are the following:
•	Use the simulator to collect data of good driving behavior
•	Build, a convolution neural network in Keras that predicts steering angles from images
•	Train and validate the model with a training and validation set
•	Test that the model successfully drives around track one without leaving the road
•	Summarize the results with a written report
## Rubric Points
### Required Files
*Are all required files submitted?*

This repo contains all necessary files.

### Quality of Code
*Is the code functional?*

The model.py code creates a model.h5 file which can operate the simulation, as seen in the video file submission.

*Is the code usable and readable?*

The model.py file is thoroughly commented. The drive.py file contains a comment where it diverges from the given file, namely in un-normalizing the prediction data. Generators were not used, as GPU and system memory were not a constraint at this time.

### Model Architecture and Training Strategy
*Has an appropriate model architecture been employed for the task?*

This project implements (with slight modifications) the architecture utilized by NVIDIA that was discussed in the lectures. It can be found here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/

![NVIDIA architecture](https://github.com/cswksu/behaviorCloningUdacity/blob/master/images/nvidiaArch.png)
 
The final model had the following parameters and features:
*	Lambda layer to normalize pixel data around 0 with range -0.5 to 0.5
*	Crop 70 top pixels and 20 bottom pixels from image
*	2D convolutional layer with depth of 24 and 5x5 kernel size and RELU activation function
*	Dropout layer with 75% retention, 25% dropout
*	2D convolutional layer with depth of 36 and 5x5 kernel size and RELU activation function 
*	Dropout layer with 75% retention, 25% dropout
*	2D convolutional layer with depth of 48 and 5x5 kernel size and RELU activation function 
*	Dropout layer with 75% retention, 25% dropout
*	2D convolutional layer with depth of 64 and 5x5 kernel size and RELU activation function 
*	Dropout layer with 75% retention, 25% dropout
*	2D convolutional layer with depth of 64 and 3x5 kernel size and RELU activation function 
*	Dropout layer with 75% retention, 25% dropout
*	2D convolutional layer with depth of 64 and 3x5 kernel size and RELU activation function 
*	Dropout layer with 75% retention, 25% dropout
*	Flatten Layer
*	100 node fully connected layer
*	50 node fully connected layer
*	1 node fully connected output later
Data was normalized in the Lambda layer, and RELU functions introduce non-linearity.

*Has an attempt been made to reduce overfitting of the model?*

Dropout layers were inserted after all convolutional layers in an attempt to reduce dropout. The following chart shows MSE over time for the validation and training set. 

![MSE plot](https://github.com/cswksu/behaviorCloningUdacity/blob/master/images/MSE.png)

30% of data was randomly selected to be held back for validation, and data was shuffled automatically by Keras. As can be seen, validation loss hits a plateau early, and training loss continues to drop, indicating that even with dropout, some overfitting still occurs. Reducing epochs to somewhere between 5-10 may help. Also, further tuning the correction parameter for the left and right images may help.

*Have the model parameters been tuned appropriately?*

An Adam optimizer was used.

*Is the training data chosen appropriately?*

Training data was collected by recording a lap of the car going around Track 1 on the lane centerline once. Instead of recording more data, the following data augmentation strategies were used:
*	Data from the center camera was flipped, with the negative steering angle added to the y_data.
*	Data from the left and right cameras were added, with an offset of 2 degrees of steering angle added to the y_data
These strategies reduced the need to film the car running the track in the opposite direction, or recording the recovery from off-angle runs.

### Architecture and Training Documentation
*Is the solution design documented?*

This README attempts to document the model architecture and solution approach.

*Is the model architecture documented?*

The previous section details the architecture of the array.

*Is the creation of the training dataset and training process documented?*

This document captures the augmentation and data capture procedure.

### Simulation
*Is the car able to navigate correctly on test data?*

The vehicle successfully navigates the track in the middle of the lane. There is some “indecisiveness” in the steering angle, but the error seems to be centered about zero, and vehicle stays in its lane.

## Discussion and Further Improvement
A deep CNN was utilized to teach a simulated vehicle to navigate Track 1 without leaving the track boundaries. 
The model could be further improved by creating a second PID controller to smooth the input to the steering wheel. While the vehicle operates in a safe manner, it is not a comfortable manner. Collecting data from track two would be helpful in further generalizing the model.
Efficiency of training could be improved by utilizing a generator function to reduce memory load.
