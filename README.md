# Diffusion-Models-for-floor-plan-drafting
Code for my Bachelor Thesis: "Using Diffusion Models to improve the process of floor plan drafting”
## Project Description

This repository contains the code developed for my Bachelor Thesis at the *University of Osnabrück, Germany*, titled "Using Diffusion Models to Improve the Process of Floor Plan Drafting." The project focuses on exploring the application of diffusion models in the field of floor plan drafting and evaluating their potential to enhance the drafting process.

## 📖 Table of Contents
  - [❓ Why?](#-why)
  - [✴️ Model](#model)
  - [✨ Features](#-features)
  - [💻 Usage](#-usage)
  - [💾 Structure](#-structure)
  - [📎 License](#-license)
  <p align="right">(<a href="#top">back to top</a>)</p>

## ❓ Why?
Stable diffusion has shown great potential for generating realistic images. However, SD cannot generate coherent architectural floor plans because it is not specialized for certain domains. This project focused on fine-tuning SD-v1.5 with LoRA to obtain a specialized tool that lets users generate architectural floor plans that follow specific constraints.
<br/>
<p align="right">(<a href="#top">back to top</a>)</p>

## ✴️ Model <a name="model"></a>
The weights for the LoRA module with the best performance (L1 loss, 250 epochs, rank 4) can be downloaded from: https://huggingface.co/maria26/Floor_Plan_LoRA <br/> 
<br/> 
and then loaded on top of SD-v1.5: https://huggingface.co/runwayml/stable-diffusion-v1-5<br/> 
<p align="right">(<a href="#top">back to top</a>)</p>

## ✨ Features
**Training:** You can train your own LoRA on another labeled dataset of floor plans <br/>
**Experiment:** You can train a LoRA module on the provided dataset and try out different hyperparameters <br/>
**User Interface:** You can interact with your Model with one of the 2 UIs. One lets you input any prompt in text form, the other one has fixed, selectable options to customize your floor plan
<br/>
<p align="right">(<a href="#top">back to top</a>)</p>

## 💻 Usage
To use the code, you first have to install the required libraries from the requirements.txt.
 ```
  pip install -r requirements.txt
  ```
After this, you can create your own LoRA on your dataset.
 ```
  ????python3 main.py -mode training -use_tf_dataset
  ```
You can also use the web-interfaces.
 ```
  ???
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
    │   │   ├───0001.png                      #dataset images
    │   │   ├───...
    │   │   ├───0280.png
    │   │   └───metadata.jsonl                #image descriptions
    ├───Evaluation
    │   ├───Interface
    │   │   ├───stress_test_results.csv       # stress test results
    │   │   └───stress_test.py                #stress test script
    │   ├───LPIPS and SSIM
    │   │   └───images                        #images generated for LPIPS and SSIM
    │   │   │   ├───L1                        #each model has a separate folder
    │   │   │   |   ├───BFMBM_1.png           #10 images each were generated
    │   │   │   |   ├───...                   #encoded with initials of the quantifiers
    │   │   │   |   └───SFOSM_10.png
    │   │   │   ├───L1_6
    │   │   │   |   └───...                   #same image names, different outputs
    │   │   │   ├───L1_8
    │   │   │   |   └───...
    │   │   │   ├───MSE
    │   │   │   |   └───...
    │   │   │   ├───Reference
    │   │   │   |   └───...
    │   │   │   ├───SD
    │   │   │   |   └───...
    │   │   │   └───SNR
    |   |   |       └───...
    │   │   ├───calculate_lpips_ssim.py      #script to compute LPIPS and SSIM scores
    │   │   ├───inference.py                 #script to generate the images above     
    │   │   ├───L1_r6_results.csv            #results of the different models
    │   │   ├───L1_r8_results.csv
    │   │   ├───L1_results.csv
    │   │   ├───mean_values.csv              #table with mean values of all models
    │   │   ├───MSE_results.csv
    │   │   ├───SD_results.csv
    │   │   └───SNR_results.csv
    │   ├───Robustness
    │   │   ├───images                       #images generated for robustness test
    │   │   │   ├───1_1_1.png                #8 categories
    │   │   │   ├───...                      #5 prompts per category
    │   │   │   └───8_5_4.png                #4 images per prompt  
    │   │   └───image_generation.py          #script to generate the images
    │   └───Training Loss
    │       ├───Combined_loss.png            #plot showing L1, SNR and MSE
    │       ├───Combined_ranks.png           #plot showing L1 with different ranks
    │       ├───Loss_L1_r6.csv               #training results of different models
    │       ├───Loss_L1_r8.csv
    │       ├───Loss_L1r4_MSE_SNR.csv
    │       ├───plot_different_losses.py     #script to plot losses
    │       └───plot_different_ranks.py      #script to plot results with diff. ranks
    ├───Interface
    │   ├───node_modules
    │   ├───static
    │   │   ├───input.css
    │   │   ├───output.css
    │   │   └───styles.css
    │   └───templates
    │   │   ├───index-selection_input.html  #selection input interface
    │   │   └───index-text_input.html       #text input interface
    │   ├───__init__.py
    │   ├───app.py
    │   ├───interface.jpynb
    │   ├───package_lock.json
    │   ├───package.json
    │   └───tailwind.config.js
    └───Training
        ├───arguments.py                   #parameters
        ├───lora_training.py               #training script
        ├───preprocessing.py               #dataset preprocessing
        └───run_script.py                  #run file
<p align="right">(<a href="#top">back to top</a>)</p>



## 📎 License
Copyright 2024 Maria Oprea

Licensed under the MIT License;
you may not use this file except in compliance with the License.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.<br/>
<p align="right">(<a href="#top">back to top</a>)</p>
