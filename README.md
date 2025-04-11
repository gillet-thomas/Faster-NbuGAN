# Faster NbuGAN Code Master Thesis: Generating High-Resolution Adversarial Images Using GANs

## Abstract  
Convolutional Neural Networks (CNNs) play a crucial role in various computer vision tasks,
but they are susceptible to adversarial attacks. However, such attacks can be positive for at
least two reasons. Firstly, they reveal vulnerabilities in CNNs, prompting efforts to enhance
their robustness. Secondly, adversarial images can also be employed to preserve privacy-sensitive
information from CNN-based threat models aiming to extract such data from images. For such
applications, the construction of high-resolution (HR) adversarial images is mandatory in practice.

This thesis analyses Generative Adversarial Network (GAN) architectures and their utilization in generating HR adversarial images. It details the modification of an Adversarial GAN
architecture capable of generating high-resolution adversarial images of any clean image size
using targeted black-box attacks on Convolutional Neural Networks.

Experiments were conducted using 100 high-resolution clean images against the target CNN,
the VGG16 classifier. The introduced framework, Faster NbuGAN, configured with an epsilon
value of 8 and the Lanczos interpolation technique, enables a fast generation of high-resolution
adversarial images requiring 111 epochs on average. It also achieved a 100% success rate in the
generation of HR adversarial images, with an average confidence value of 50.73% in the desired
target categories.
With the successful achievement of this project’s objectives, our team is currently summarizing the content of this thesis for publication in an international conference.

[Read the full thesis here](https://thomasgillet.com/master_thesis.pdf).

## Code Overview  
This repository contains the official code used to produce the results outlined in the thesis.

**Note**: The part of the code related to the "Noise Blowing Up" strategy has been removed for copyright reasons.
Users will need to re-implement this strategy by replacing the commented code block starting with **"Noise blowing up strategy"** in the code.

The code was developed and run using **TensorFlow 2.15**.
