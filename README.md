# Atherosclerotic Lesion Segmentation & Quantification UNET

## Author

William Giles, UVA CS '25

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Model Overview](#model-overview)
  - [Architecture](#architecture)
  - [Training](#training)
  - [Performance](#performance)
- [Datasets](#datasets)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citations](#citations)
- [Contact](#contact)

## Introduction

The pathological cause of deadly heart diseases such as coronary artery disease, ischemic stroke, and peripheral artery disease, atherosclerosis, is a chronic inflammatory disease of the arterial wall that forms fatty plaques. These lesions occur when lipid-containing plaques that build up and line the artery walls cause ruptures, disrupting blood flow and causing adverse cardiovascular events. Atherosclerotic research is commonly conducted using gene engineered mice, such as Apoe and Ldlr knockouts (Apoe-/-, Ldlr-/-). These mice develop all phases of atherosclerotic lesions that humans do, making them a great way to gain more insight into how this deadly pathological cause manifests itself in humans. The method for histological analysis of atherosclerosis consists of the measurement of plaque burden on cross sections of the aortic sinus and root stained with oil red O staining and hematoxylin. 

Because the current methods for measuring atherosclerotic lesion sizes on numerous histological sections are extremely tedious, the goal of this research project is to develop and train a 2D-Unet to automate and optimize this process.

## Getting Started

### Prerequisites

List any libraries, frameworks, or tools that need to be installed before using your project.

### Installation

1. Open terminal and clone this repository using git clone https://github.com/wdgstl/Atheroscelerotic_Lesion_UNET.git

2. Install all requirements from requirements.txt using command:

3. Upload image to segment to the same folder

4. Run the command: python3 measure_lesion.py

5. Follow prompts and enter the image filepath

6. Segmentation Image, Image with Measurements, and a text file with measurements will be output into a folder called (image_name-predicted)

## Usage

Explain how to use your model with code snippets and examples. Detail any scripts or commands to run the model, input data formats, and how to interpret the output.

## Model Overview

### Architecture

Describe the architecture of your model. Include diagrams if possible.

### Training

Detail the training process, including any pre-processing steps, hardware used (e.g., GPU), training times, and challenges faced.

### Performance

Discuss the performance metrics you used to evaluate your model and the results it achieved.

## Datasets

The first component in data collection is developing the histology images of the oil red O-stained cross sections of mouse aortas from the Apoe and Ldlr knockouts. First, the microscopic slides of these mice must be developed and stained. In order to stain the aortic samples, the aortic root and adjacent 1/3 of the heart are embedded in Tissue-Tek compound and cross-sectioned in 10-µm thickness. Sections are stained with oil red O and hematoxylin and counterstained with fast green. Stained sections are imaged under a Zeiss primo star microscope through AxioVision 4.8 program. Then, the Zeiss Zen program is used to image these samples, and the images are saved as .czi images that contain manually segmented atherosclerotic lesions along with measured sizes for each lesion.

After the manually segmented images are obtained, several steps of image processing must be taken in order to ensure the images are compatible with the UNET architecture and that the model can be trained properly. One of the initial problems that was encountered with the image dataset was that there were several folders and subfolders that contained both the raw and manually segmented images. In order to train the 2D U-Net, the images need to be contained in only two folders: one for the histology images, and the other for the segmented images. To solve this problem, I created several bash scripts that pulled out all of the images from the histology folders and subfolders and placed them into one histology folder. I repeated this for the segmented images to achieve a total of two image folders. The next issue I faced was that the two folders did not contain the same number of files, which means that not all histology images had a corresponding manual segmentation, which would be a real problem for model training since each input image needs exactly one corresponding label in a supervised deep learning model. To fix this, I created a python program that traversed through both folders and returned the difference between the two sets, or in other words the files that needed to be removed in order for each histology image to map to its segmentation. 

The next stage of image preprocessing involved converting .czi segmentation images to .tif segmentation masks, since .czi file extensions are not compatible with the UNET architecture. This stage involved using ImageJ to convert the manual segmentations of the raw image into binary masks, where white represents areas that are segmented as the lesion, and black is the background. This is an important step when dealing with training a 2D U-Net because it puts the mask in terms of 1s and 0s, which allows it to perform the correct matrix calculations. Next, because measuring the lesions was part of the histology image preparation section, the measurement values for each image are also saved. Overall, the end goal of preprocessing is to have the images with the correct file extension so that they are compatible with the UNET, be placed into two distinct folders (histology and segmentation), and that each segmentation has a corresponding histology image. 

There were a total of 2533 images used in this project. They were broken up into 1533 images that were used to train the UNET, 510 images that were used to validate the UNET, and 510 images that were used to test and measure the accuracy of the UNET. 

## Results

Showcase the results of your model with images that compare the input and segmented output. Quantitative metrics can also be included here.

## Contributing

If you're open to contributions, provide guidelines for how others can contribute to your project.

## License

Specify the license under which your project is released.

## Citations

If your work builds upon or uses datasets or code from other projects, cite these sources.

## Contact

Include contact information for the project maintainers or contributors for users with further questions or feedback.

