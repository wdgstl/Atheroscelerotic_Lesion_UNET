# **AtheroQuantNet v1.0**
Atherosclerotic Lesion Segmentation & Quantification UNET

v.10 - Feb 7, 2024

## Author

William Giles, UVA CS '25

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Usage](#installation-&-usage)
- [Model Overview](#model-overview)
  - [Architecture](#architecture)
  - [Training](#training)
- [Datasets](#datasets)
- [Results](#results)
- [Further Work](#further-work)
- [Citations](#citations)
- [Contact](#contact)

## Introduction

The pathological cause of deadly heart diseases such as coronary artery disease, ischemic stroke, and peripheral artery disease, atherosclerosis, is a chronic inflammatory disease of the arterial wall that forms fatty plaques. These lesions occur when lipid-containing plaques that build up and line the artery walls cause ruptures, disrupting blood flow and causing adverse cardiovascular events. Atherosclerotic research is commonly conducted using gene engineered mice, such as Apoe and Ldlr knockouts (Apoe-/-, Ldlr-/-). These mice develop all phases of atherosclerotic lesions that humans do, making them a great way to gain more insight into how this deadly pathological cause manifests itself in humans. The method for histological analysis of atherosclerosis consists of the measurement of plaque burden on cross sections of the aortic sinus and root stained with oil red O staining and hematoxylin. 

Because the current methods for measuring atherosclerotic lesion sizes on numerous histological sections are extremely tedious, the goal of this research project is to develop and train a 2D-Unet to automate and optimize this process.

## Getting Started

### Prerequisites

Requires up to date installations of Python3 and pip3.

All additional requirements needed for running this model are in requirements.txt.

To install them, simply run the command: pip install -r requirements.txt 

### Installation & Usage

1. Open terminal and clone this repository using git clone https://github.com/wdgstl/Atheroscelerotic_Lesion_UNET.git

2. Install all requirements from requirements.txt using command: pip install -r requirements.txt 

3. Upload image to segment to the same folder (in .tif format)

4. Run the command: python3 measure_lesion.py 

5. Follow prompts and enter the image filepath 

6. Histogram Image, Segmentation Image Mask, Image with Measurements, and a csv with measurements will be saved in the results file of the directory. 

## Model Overview

### Architecture

![image](https://github.com/wdgstl/Atheroscelerotic_Lesion_UNET/assets/117789564/fcd2cd88-1cc7-402b-991b-ba787347de63)

Convolutional neural networks are a type of deep learning model that have the ability to process and analyze images, making them ideal candidates for improving many biomedical image classification, recognition, and segmentation processes that are required in the medical field. These models are supervised learning models, meaning that they can be trained with thousands of images that are labeled with a corresponding mask, or annotated image. U-net is a type of convolutional neural network architecture that is ideal for segmenting 2D biomedical images. It consists of two primary components: the contracting path, and the expanding path. The contracting path is down convolutional and consists of a series of repeated convolutional and pooling operations. As the images are processed through this path, the feature maps get spatially smaller, increasing the “what” and decreasing the “where.” Next, the feature maps are built up into the original image size in the expanding path, where the up-sample representations at each step are concatenated with the corresponding feature maps in the contraction pathway. Ultimately, training a 2D U-Net model with hundreds to thousands of images will increase its accuracy in segmenting images until it is ready to be safely deployed in the medical field. As mentioned, these models have had tremendous success in segmenting biomedical images. In the Shi lab, a 2D-Unet supervised model has been created and trained to automatically segment abdominal subcutaneous fat and visceral fat on CT and magnetic resonance images. The question is whether a 2D-Unet supervised model can be used to catalyze advances in researching deadly heart diseases.

### Training

The first component in data collection is developing the histology images of the oil red O-stained cross sections of mouse aortas from the Apoe and Ldlr knockouts. First, the microscopic slides of these mice must be developed and stained. In order to stain the aortic samples, the aortic root and adjacent 1/3 of the heart are embedded in Tissue-Tek compound and cross-sectioned in 10-µm thickness. Sections are stained with oil red O and hematoxylin and counterstained with fast green. Stained sections are imaged under a Zeiss primo star microscope through AxioVision 4.8 program. Then, the Zeiss Zen program is used to image these samples, and the images are saved as .czi images that contain manually segmented atherosclerotic lesions along with measured sizes for each lesion.

After the manually segmented images are obtained, several steps of image processing must be taken in order to ensure the images are compatible with the UNET architecture and that the model can be trained properly. One of the initial problems that was encountered with the image dataset was that there were several folders and subfolders that contained both the raw and manually segmented images. In order to train the 2D U-Net, the images need to be contained in only two folders: one for the histology images, and the other for the segmented images. To solve this problem, I created several bash scripts that pulled out all of the images from the histology folders and subfolders and placed them into one histology folder. I repeated this for the segmented images to achieve a total of two image folders. The next issue I faced was that the two folders did not contain the same number of files, which means that not all histology images had a corresponding manual segmentation, which would be a real problem for model training since each input image needs exactly one corresponding label in a supervised deep learning model. To fix this, I created a python program that traversed through both folders and returned the difference between the two sets, or in other words the files that needed to be removed in order for each histology image to map to its segmentation. 

The next stage of image preprocessing involved converting .czi segmentation images to .tif segmentation masks, since .czi file extensions are not compatible with the UNET architecture. This stage involved using ImageJ to convert the manual segmentations of the raw image into binary masks, where white represents areas that are segmented as the lesion, and black is the background. This is an important step when dealing with training a 2D U-Net because it puts the mask in terms of 1s and 0s, which allows it to perform the correct matrix calculations. Next, because measuring the lesions was part of the histology image preparation section, the measurement values for each image are also saved. Overall, the end goal of preprocessing is to have the images with the correct file extension so that they are compatible with the UNET, be placed into two distinct folders (histology and segmentation), and that each segmentation has a corresponding histology image. 

There were a total of 2533 images used in this project. They were broken up into 1533 images that were used to train the UNET, 510 images that were used to validate the UNET, and 510 images that were used to test and measure the accuracy of the UNET. 

## Datasets

This model used private datasets collected from Shi Lab. (2,533 histogram images and 2,533 segmentation images). 

## Results 

The model takes in a histogram image and outputs a predicted segmentation mask along with the predicted areas of each segmented lesion, as shown below:

_**Raw Histogram **_

![hist](https://github.com/wdgstl/Atheroscelerotic_Lesion_UNET/assets/117789564/0b667b86-8197-41cb-99d7-65367f14c54f)


_**Segmentation Mask (Model output a.)**_ 

![mask](https://github.com/wdgstl/Atheroscelerotic_Lesion_UNET/assets/117789564/6f5c946a-bd54-4561-a0ae-a0bbeb31a081)


_**Segmentation with Measurements (Model output b.)**_ 

![meas](https://github.com/wdgstl/Atheroscelerotic_Lesion_UNET/assets/117789564/1a9483e2-b0bb-4406-a312-23a50ac13693)



The model metrics, displayed below, demonstrate the segmentation accuracy of v1.0 of the AtheroQuantNet 2D-UNET.

F1: 0.82380

Jaccard: 0.71482

Recall: 0.80116

Precision: 0.86614

## Further Work

Future work will be dedicated to enhancing model accuracy and usability. Some benchmarks include: deploying model to an app and experimenting with different model architectures (DeepLabV3).

## Citations

https://github.com/nikhilroxtomar/Brain-Tumor-Segmentation-in-TensorFlow-2.0/blob/main/UNET/README.md
Consulted for overall unet architecture & training/testing pipelines.

https://arxiv.org/abs/1505.04597
Consulted for 2D-UNET architecture. 

## Contact

For more information, contact William Giles @ wdgstl@gmail.com

