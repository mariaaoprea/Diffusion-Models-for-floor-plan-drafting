# Diffusion-Models-for-floor-plan-drafting
Code for my bachelor thesis: "Using Diffusion Models to improve the process of floor plan drafting”
## Project Description

This repository contains the code for my Bachelor Thesis at the _University Osnabrück, Germany_, titled "Using Diffusion Models to Improve the Process of Floor Plan Drafting". The goal of this project was to explore the application of diffusion models in the field of floor plan drafting and evaluate their capabilities in improving florr plan the drafting process.

## 📖 Table of Contents
  - [❓ Why?](#-why)
  - [✴️ Model](#model)
  - [✨ Features](#-features)
  - [💻 Usage](#-usage)
  - [💾 Structure](#-structure)
  - [📎 License](#-license)
  <p align="right">(<a href="#top">back to top</a>)</p>

## ❓ Why?
Artificial Neural Networks (ANNs) are only loosely inspired by the human brain while Spiking Neural Networks (SNNs) incorporate various concepts of it.
Spike Time Dependent Plasticity (STDP) is one of the most commonly used biologically inspired unsupervised learning rules for SNNs.<br/>
In order to obtain a better understanding of SNNs we compared their performance in image classification to Fully-Connected ANNs using the MNIST dataset. <br/> 
<img src="Images/MNISTDatasetSample.JPG" alt="MNIST Example Images" align="middle" width="500" /> <br/> 
For this to work, we had to transform the data for the SNN into rate-encoded spike trains.
As a major part of our work, we provide a comprehensible implementation of an STDP-based SNN.
<p align="right">(<a href="#top">back to top</a>)</p>

## ✴️ Model <a name="model"></a>
The model can be downloaded from: https://huggingface.co/maria26/Floor_Plan_LoRA <br/>
<p align="right">(<a href="#top">back to top</a>)</p>

## ✨ Features
With the files we provided you can either train your own Spiking-Neural-Network or do inference on existing pretrained weights. For training you can either use the dataset we uploaded in the MNIST folder and subfolders or you can simply use the MNIST dataset provided by tensorflow. Therefore in the [SNN.py](SNN.py) file you can find examples for both, how to convert your own image data into spike trains and how to transform an existing tensorflow dataset into spike trains.<br/>
<p align="right">(<a href="#top">back to top</a>)</p>

## 💻 Usage
To use our code, you first have to install the requiered libraries from the requirements.txt.
 ```
  pip install -r requirements.txt
  ```
After this, you can train your own SNN.
 ```
  python3 main.py -mode training -use_tf_dataset
  ```
You can also use this script to test your own trained network and weights.
 ```
  python3 main.py -mode inference -weights_path folder/weights.csv -labels_path folder/labels.csv -image_inference_path folder/picture.png
  ```
To get a list of all possible hyperparameters use
 ```
  python3 main.py -h
```
<p align="right">(<a href="#top">back to top</a>)</p>

## 💾 Structure
<!-- Project Structure -->

    .
    ├───dataset
    │   └───train
    │   │   ├───0001.png
    │   │   ├───...
    │   │   ├───0280.png
    │   │   └───metadata.jsonl
    ├───Evaluation
    │   ├───Interface
    │   │   ├───stress_test_results.csv
    │   │   ├───stress_test.py
    │   ├───LPIPS and SSIM
    │   │   └───images
    │   │   │   ├───L1
    │   │   │   |   ├───BFMB_1.png
    │   │   │   |   ├───...
    │   │   │   |   └───SFOSM_10.png
    │   │   │   ├───L1_6
    │   │   │   |   ├───BFMB_1.png
    │   │   │   |   ├───...
    │   │   │   |   └───SFOSM_10.png
    │   │   │   ├───L1_8
    │   │   │   |   ├───BFMB_1.png
    │   │   │   |   ├───...
    │   │   │   |   └───SFOSM_10.png
    │   │   │   ├───MSE
    │   │   │   |   ├───BFMB_1.png
    │   │   │   |   ├───...
    │   │   │   |   └───SFOSM_10.png
    │   │   │   ├───Reference
    │   │   │   |   ├───BFMB_1.png
    │   │   │   |   ├───...
    │   │   │   |   └───SFOSM_10.png
    │   │   │   ├───SD
    │   │   │   |   ├───BFMB_1.png
    │   │   │   |   ├───...
    │   │   │   |   └───SFOSM_10.png
    │   │   │   └───SNR
    │   │   │   |   ├───BFMB_1.png
    │   │   │   |   ├───...
    │   │   │   |   └───SFOSM_10.png
    │   │   ├───calculate_lpips_ssim.py
    │   │   ├───inference.py
    │   │   ├───L1_r6_results.csv
    │   │   ├───L1_r8_results.csv
    │   │   ├───L1_results.csv
    │   │   ├───stress_test_results.csv
    │   │   ├───MSE_results.csv
    │   │   ├───SD_results.csv
    │   │   └───SNR_results.csv
    │   ├───Robustness
    │   │   └───images
    │   │   │   ├───1_1_1.png
    │   │   │   ├───...
    │   │   │   └───8_5_4.png
    │   │   └───image_generation.py
    │   └───Training Loss
    │   │   ├───Combined_loss.png
    │   │   ├───Combined_ranks.png
    │   │   ├───Loss_L1_r6.csv
    │   │   ├───Loss_L1_r8.csv
    │   │   ├───Loss_L1r4_MSE_SNR.csv
    │   │   ├───plot_different_losses.py
    │   │   ├───plot_different_ranks.py
    ├───Interface
    │   ├───node_modules
    │   ├───static
    │   └───templates
    └───Training
        ├───arguments.py
        ├───lora_training.py
        ├───preprocessing.py
        └───run_script.py
<p align="right">(<a href="#top">back to top</a>)</p>



## 📎 License
Copyright 2024 Maria Oprea

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
<p align="right">(<a href="#top">back to top</a>)</p>
