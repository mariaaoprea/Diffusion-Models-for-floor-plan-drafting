# Diffusion-Models-for-floor-plan-drafting
Code for my Bachelor Thesis: _"Using Diffusion Models to improve the process of floor plan drafting”_
## Project Description

This repository contains the code developed for my Bachelor’s Thesis, conducted at the *DFKI Osnabrück (German Research Center for Artificial Intelligence)* as part of my Bachelor’s degree at the *University of Osnabrück, Germany*. The project focused on exploring the application of diffusion models in floor plan drafting and assessing their potential to improve the drafting process.

## 📖 Table of Contents
  - [❓ Why?](#-why)
  - [✴️ Model](#model)
  - [✨ Features](#-features)
  - [💻 Usage](#-usage)
  - [💾 Structure](#-structure)
  - [📎 License](#-license)
  <p align="right">(<a href="#top">back to top</a>)</p>

## ❓ Why?
SStable Diffusion has demonstrated impressive capabilities in generating realistic images. However, it struggles to produce coherent architectural floor plans, as it is not tailored for this specific domain. This project aims to address that limitation by fine-tuning SD-v1.5 with LoRA, resulting in a specialized tool that enables users to generate architectural floor plans adhering to specific constraints.<br/> 

<p align="right">(<a href="#top">back to top</a>)</p>

## ✴️ Model <a name="model"></a>
The weights for the LoRA module with the best performance (L1 loss, 250 epochs, rank 4) can be downloaded from: https://huggingface.co/maria26/Floor_Plan_LoRA <br/> 
<br/> 
and then loaded on top of SD-v1.5: https://huggingface.co/runwayml/stable-diffusion-v1-5<br/> 
<p align="right">(<a href="#top">back to top</a>)</p>

## ✨ Features
**Training:** Train your own LoRA model on a different labeled dataset of floor plans. <br/>
**Experimentation:** Fine-tune a LoRA module using the provided dataset and explore various hyperparameter settings. <br/>
**User Interface:** Interact with your model through one of two available UIs: a text-based interface for freeform prompts or a selection-based interface with preset options for customizing your floor plan.
<br/>
<p align="right">(<a href="#top">back to top</a>)</p>

## 💻 Usage
To use the code, you first have to install the required libraries from the requirements.txt.
 ```
  pip install -r requirements.txt
  ```
To see all possible parameters, look at `arguments.py` and set the desired values in `run_script.py`. <br/>
After this, you can train your own LoRA module on the provided floor plan dataset or your own one by running the code below. Make sure to give the right path to the `train_data_dir` parameter.
 ```
  python3 run_script.py 
  ```
If you want to use my fine-tuned model, it is already linked to the interface.
<br/>
<br/>
You can also access the web interfaces by running the code below. Make sure that the path to the LoRA safe-tensors in `app.py` is right so that it loads correctly.😊
<br/>
 ```
  python3 app.py 
  ```
You can link `app.py` to your preferred interface (text or selection input). By default, it is set to the selection input, but you can adjust the corresponding line in the `home()` function as needed.
<p align="right">(<a href="#top">back to top</a>)</p>

## 💾 Structure
<!-- Project Structure -->

    .
    ├───Dataset
    │   └───train
    │       ├───0001.png                      #dataset images
    │       ├───...
    │       ├───0280.png
    │       └───metadata.jsonl                #image descriptions
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
    │       └───plot_different_ranks.py      #script to plot results with different ranks
    ├───Interface
    │   ├───node_modules
    │   ├───static
    │   │   ├───input.css
    │   │   ├───output.css
    │   │   └───styles.css
    │   └───templates
    │   │   ├───index-selection_input.html   #selection input interface
    │   │   └───index-text_input.html        #text input interface
    │   ├───__init__.py
    │   ├───app.py                           #make sure to add the right path to your model
    │   ├───interface.jpynb
    │   ├───package_lock.json
    │   ├───package.json
    │   └───tailwind.config.js
    ├───Model                                #EMPTY folder, you can ADD your trained LoRA model to this folder
    └───Training
        ├───arguments.py                     #parameters
        ├───lora_training.py                 #training script
        ├───preprocessing.py                 #dataset preprocessing
        └───run_script.py                    #run file
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
