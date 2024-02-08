# Deep-Learning-in-Python
This repository contains the final project I did together with two classmates for the Deep Leanring in Python course at the University of Amsterdam. 

# File Overview
- _DLiP - Image Degradation (Poster).pdf_ - Poster summarizing our project
- _Main.ipynb_ - Our final jupyter notebook including background information, the training pipelines, and our results

# Abstract
To perform reliably in the real world, computer vision models need to generalize well to images regardless of their state of degradation. For example, self driving cars have to reliably detect obstacles no matter the light conditions or the speed of the vehicle. To investigate how well convolutional neural networks (CNN) generalize to augmented images, we re-trained (i.e., transfer learning) seven Imagenet based Resnet-50 to classify different augmentations of the same images of animals. Specifically, we trained the CNNs on normal images (CNN-normal) , images with motion blur (CNN-mblur), images with gaussian blur (CNN-gblur), images with gaussian noise (CNN-gnoise), underexposed images (CNN-uexp), overexposed images (CNN-oexp), and all augmentations (CNN-comb) respectively. Importantly we found that the CNN-normal did not generalize well to degraded images while CNNs trained on degraded images generalized well to normal images. As expected, CNN-comb performed well on all types of images. Together these results underline the importance of explicitly including image augmentations into the training pipeline of CNNs. 

# Image Augmentations
![augmentations.png](https://github.com/lukekorthals/Deep-Learning-in-Python/blob/master/results/augmentations.png)
