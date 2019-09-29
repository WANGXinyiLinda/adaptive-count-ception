# Cell Counting by Adaptive Fully Convolutional Redundant Counting

This the repository for our HKUST COMP4901J (Stanford CS231) final project. 

Cell counting is an important task in biological domain. However, it could be time consuming and error-prone to count cells manually. Some deep fully convolutional neural networks have been proposed recently to automatically count cells from the microscopic images. But the current methods can only be trained to count one type of cells. If we want to count another kind of cells, a proper amount of new data is needed to train the whole network with millions of parameters which is very time consuming. To enable fast domain transfer between different kind of cells, we propose to pre-train the network on a large dataset of simple synthetic microscopic cell images and then freeze the domain agnostic parameters and only train the domain specific parameter on the new domain. In details, our network structure is obtained by adding several residual adapters to the count-ception network, which is the current state of the art method for cell counting task. Experiment results show that our proposed method significantly outperforms the original training from scratch method on two small datasets of real microscopic cells images.  

For more details on the model and expriments, see our final [report](4901J_final_report.pdf) and final presentation [slides](4901J_final_presentation.pdf).

## Data

This repository includes the three data sets we used: VGG cells, MBM cells, and a Hela cell dataset collected from Professor Hong Xue's biochemistry lab in HKUST. The former two datasets are standard public dataset which can be obtained on internet. More detailed dataset statistics can be found in section 3 our final report.

## Usage

This python 3 implementation is mostly based on Keras 2.2.4. You will also need skimage and scipy to handle the images.

For how to train the model, see the jupyter notebooks.

## Results

A sample test result from the Hela cells dataset. From left to right: the original cell image, the processed ground truth we used, and our prediction. The predicted cell nember is 84.0 while the ground truth is 89. 75 Hela cell images has been used for training.

![](img\cell-img.png)
![](img\ground-truth(89).png)
![](img\adapt-prediction(84.0).png)

## Acknowledgment

This implementation is based on the Keras implementation of count-ception at https://github.com/fizzoo/countception-recreation.