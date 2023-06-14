# Neural Style Transfer

I got trouble to finish the project at school (Due to checker issues (Tell you if youre code is doing good or not))

So I decided to do works on my owns without the checker. And it works acctually pretty well.

This code was based on the [TensorFlow blog post](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398) that help me to get a good understanding of what happen under the hood.

## Abstract
This use the VGG19 model pre-trained on the imagenet dataset. It will take a style image and a content image as input, and return the "sum" of thoses.

## Version
It work on:
- *python3.10*
- TensorFlow 2.12
- Numpy 1.23.5
- MatPlotLib dependancies related to TF and NP

## How to use
Just hit the `run_style_tranfer` function with the path of the 2 images (first style then content) and that pretty it.

You can tune hyperparameter such as the weight given for each image (The ones calculated on the loss function). To make simple, the bigger the weigth is, the most of the image will be use in the output image.