#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/random_img_visual.png "Random Image Visualization"
[image2]: ./examples/signcount.png "Number of images per class in training data"
[image3]: ./examples/curveLeft.png "Curve Left"
[image4]: ./examples/Do-Not-Enter.jpg "Do Not Enter"
[image5]: ./examples/end-speed.png "End of Speed Limit (80km)"
[image6]: ./examples/stop.jpg "Stop"
[image7]: ./examples/wild-animal.png "Wild Animal Crossing"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/miradesne/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in code cell *#2* of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3 (3 is the rgb color channel)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in code cell *#4* of the IPython notebook.  

Here is an exploratory visualization of the data set. 

This is a random image from the training data set. The top is in color, and the bottom is in grayscale. The image contrast is improved after it becomes gray scale. 

![alt text][image1]

It is a bar chart showing number of images per class in the training data set. We can see that they are not evenly distributed. Some classes have way more images than the others.

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in code cell *#6* of the IPython notebook.

As a first step, I decided to convert the images to grayscale because from the visualization, grayscale increases constrast of the images and decreases number of features that are not neccesarily relavent to the recognition process. It can also make the training faster because we reduce the input data dimension from 3 to 1.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image1]

After that, I normalized the image data because a smaller scaled data set usually produce better learning result. Also training converges faster with normalization.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data comes with training, validation, and test sets so I only needed to load them into the script.

The training set had 34799 number of images. The validation set and test set had 4410 and 12630 number of images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in code cell *#7* of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64	 				|
| Flatten				| outputs 1600									|
| Fully connected		| outputs 120									|
| RELU					|												|
| Dropout				| keep 75% of the nodes							|
| Fully connected		| outputs 84									|
| RELU					|												|
| Dropout				| keep 75% of the nodes							|
| Fully connected		| outputs 43									|



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training pipeline is located in code cell *#9* of the ipython notebook. 

To train the model, I first calculate the cross entropy after applying softmax function to my output by using `tf.nn.softmax_cross_entropy_with_logits()`. Then I defined the loss as the mean of the cross entrophy by using `tf.reduce_mean()`. I chose to use Adam optimizer because it uses moving averages of the parameters, and converges faster than the simple gradient descent optimizer.
I used a learning rate of 0.001, a batch size of 128 and 30 epochs.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for defining accuracy evaluation function is located in code cell *#10*
The code for calculating the accuracy of the model is located in code cell *#11*, *#12* and *#13* of the Ipython notebook.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 95.6%
* test set accuracy of 93.4%

My journey of reaching the final model:
* What was the first architecture that was tried and why was it chosen?

I first started from the LeNet in the lab with 6 features in the 1st convolution, and 16 features in the 2nd convolution. It's a nice set up because it's used in image classification of a different data set that has the same dimension. I thought it would probably work well in this data set too. 

* What were some problems with the initial architecture?
1. The number of features are too small in the convolution layers. We are trying to classify images with more complex set of features, and more variaty of features in the traffic signs, instead of just small curves, straight lines and circles as they have in digits.

2. The model was overfitting with the training data so the validation accuracy was very low.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?

1. I tuned the number of features in the convolution layers to 32 and 64. I tried bigger numbers and they didn't perform well either.

2. I set up drop out layers of a rate of 25% in each of the fully connected layer to treat overfitting.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The convolution layers work well with this problem because traffic signs have specific sets of features (numbers, shapes, etc). The convolution layers can extract these features, and learn how they are related to the meaning of the signs.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 They all have relatively high accuracy (more than 90%). The model converges very fast too, usually within 20 to 30 epochs. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] 

The 1st and 2nd images should be easy to classify because the contrast is large and the shapes of the sign are simple.

The 3rd image(end of speed limit 80) can be extremely hard to classify because it has the number of "80" which can be confused with the "80 speed limit" sign. Also in the training data set the number of examples is so low that the model develops bias towards other classes.

The 4th image(stop sign) can be hard to classify because it includes four shapes and combined with a meaning. The shapes can also be mistaken from the numeric shapes like the 30/50/80 speed limits. Also, the number of examples in the training set is relatively small.

The 5th image(wild animal crossing) can be hard to classify because the resolution of the images is super low and we need to identify a certain shape that just looks like a squiggle in 32x32 dimension.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

The code for making predictions on my final model is located in code cell *#16* of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Curve Left      		| Curve Left   									| 
| No Entry     			| No Entry 										|
| End of Speed Limit(80)| Children Crossing 							|
| Stop  	      		| Stop      					 				|
| Wild Animal Crossing  | Wild Animal Crossing  						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is lower than the accuracy of the test set. However the sample number is very small, so it cannot determine that the accuracy would be much lower. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions with top 5 softmax probabilities is located in code cell *#18* of the Ipython notebook.

For the first image, the model is absolutely sure that this is a curve left sign, and the image does contain a curve left sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Curve Left   									| 
| .0     				| No passing 									|
| .0 					| Ahead only									|
| .0	      			| Right-of-way at the next intersection			|
| .0				    | Slippery Road      							|

For the second image, the model is absolutely sure that this is a no-entry sign, and the image does contain a no-entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No Entry   									| 
| .0     				| Roundabout mandatory							|
| .0 					| Turn left ahead								|
| .0	      			| Keep right                        			|
| .0				    | Stop                							|

For the third image, the model doesn't really know what it is doing and is just trying to guess that this is a child crossing sign, but the image contains an end of speed limit(80) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .49         			| Children crossing								| 
| .32     				| Roundabout mandatory							|
| .18 					| Turn left ahead								|
| .01	      			| Ahead only                        			|
| .0				    | Keep left            							|

For the fourth image, the model is guessing that this is a stop sign, and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .59         			| Stop 											| 
| .23     				| End of all speed and passing limits 			|
| .08 					| End of speed limit (80km/h)					|
| .03	      			| Turn left Ahead                       		|
| .03				    | Speed limit (30km/h)         					|

For the fifth image, the model is absolutely sure that this is a wild animal crossing sign, and the image does contain a wild animal crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Wild animals crossing							| 
| .0     				| Road work 									|
| .0 					| Slippery road									|
| .0	      			| Double curve                       			|
| .0				    | Beware of ice/snow             				|
