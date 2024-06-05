All experiments in this study are conducted using the popular open-source machine learning library PyTorch. 
The Intel Iris Xe Graphics is used as the GPU. 
The hardware components feature an 11th Gen Intel® Core™ i5-1135G7 CPU with a base clock speed of 2.40GHz, running on the Windows 10 operating system.

Data
------------------------------------------------------------------------------------
There exist 1685 leaf images in both datasets and 5 disease and severity classes.

-----------------------------
Leaf Disease Classification:
-----------------------------

training set:
	brown: 247 images,
	cercospora: 98 images,
	healthy: 190 images,
	miner: 273 images,
	rust: 371 images
	
validation set:
	brown: 48 images,
	cercospora: 31 images,
	healthy: 35 images,
	miner: 53 images,
	rust: 86 images

testing set:
	brown: 53 images,
	cercospora: 18 images,
	healthy: 47 images,
	miner: 61 images,
	rust: 74 images

--------------------------
Leaf Severity Estimation:
--------------------------

training set:
	healthy: 190 images,
	high: 69 images,
	low: 225 images,
	very_high: 34 images,
	very_low: 661 images
	
validation set:
	healthy: 35 images,
	high: 14 images,
	low: 54 images,
	very_high: 12 images,
	very_low: 138 images

testing set:
	healthy: 47 images,
	high: 18 images,
	low: 53 images,
	very_high: 10 images,
	very_low: 125 images


Code
------------------------------------------------------------------------------------
Some operations in DeepEMPR.py 

-> Computing the HDMR constant, univariate, bivariate and trivariate components.\n
-> Computing the EMPR constant, univariate, bivariate and trivariate components.
-> Computing optimum support vectors.
-> Creating a new channel using some EMPR components and adding it to the original image.
-> Enhancing the contrast of newly generated channel based on HDMR.
-> Data augmentation (Horizontal Flip, Vertical Flip, Rotation, Adjusting contrast, brightness and saturation).
-> Constructing the Deep CNN models (AlexNet, VGG16, ResNet50).
-> Modify the deep architectures according to 4-channel image.
-> Training the deep model with transfer learning approach.
-> Testing the CNN models and evaluating them on accuracy,precision,recall and f1-score metrics. 
-> Sketching the ROC-AUC curve.
