# CaptchaTheBots

This is a project for the CS 6220 course of Big Data Systems and Analytics.

**This repository contains results of our explorations into protecting the CIFAR-10 dataset**

Contributors:

- Aditya Chaurasia
- Jineet Desai
- Khushi Talesra
- Dipam Shah
- Ryan Ding

## Problem
Recent advancements in Large Language Models (LLMs) and other Machine Learning (ML) technologies have brought about both positive and negative impacts. While the positive effects are evident, the associated risks, particularly concerning image manipulation and content misuse, are less apparent.

The increasing capabilities of image recognition and generation, coupled with sophisticated ML algorithms, raise the stakes for hosting photos online. Protecting photos is imperative for various reasons, including preventing unauthorized use, preserving intellectual property, mitigating data scraping risks, and safeguarding personal and professional privacy.

## Our Goals
Our project aims to address the risks associated with hosting photos online by introducing modifications (attacks) on images that would thwart pre-trained bots, simulated by our models. We propose employing an ensemble of attacks based on data collected from trained models on attacked images.

### Static Approach
Initially, we adopt a static approach, applying randomly chosen attacks on the test set to evaluate their effectiveness in countering pre-trained bots.

### Dynamic Approach
Subsequently, we explore a more dynamic approach by ranking different perturbations and attacks. This dynamic ranking system enhances image protection by selecting attacks based on their potency against models attempting to classify them. This approach allows for more adaptive and efficient protection against potential threats.

### Class-based Ranking
We classify the potency of different attacks on various classes, evaluating their success in protecting each class from classification attempts by models. This ranking system aids in selecting the most suitable attack for images belonging to specific classes.

### Extension to CAPTCHA Systems
Finally, we investigate possible extensions of our approach to existing CAPTCHA systems, leveraging our findings to enhance the security and robustness of these systems.

By combining static and dynamic approaches, along with class-based rankings, our project aims to provide a comprehensive solution to the challenges posed by advancing ML technologies in the realm of online photo protection.

## Open Source Packages
All of our models were derived from the [Keras API](https://keras.io/api/applications/). The models are written in python code totaling around 250 lines for each model.

Our implementations for the FGSM, Hopskip, and PGD attacks were retrieved from the following [aggregated repository](https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Attacks). The implementation is done in Python and contains roughly 200 lines per attack.


## Dataset 
We used the CIFAR10 dataset provided by [Keras.](https://keras.io/api/datasets/cifar10/) CIFAR10 contains 50,000 training images and 10,000 testing images divided into 10 classes (each featuring a specific object). Each image is of size 32x32.

## Performance Measurement Tools
Please see `group14.FinalReport.pdf` which contains our analysis and responses. The file can be found on canvas as a submission by Aditya Chaurasia. As per the Georgia Tech honor code, our final report is not available to the public.

## Usage
Follow the respective `readme` document, if available, for each subfolder. Most of the material is self-contained in notebooks and thus can be run sequentially out-of-the-box.

## Repository Structure
The `Adversarial Attacks` folder contains the notebooks used to generate the perturbed images using our three adversarial attacks (FGSM, Hopskip, PGD) on our test models.

The `AttackTransfer` folder contains the notebook that was used to explore transferability by applying perturbed images on non-directed models. Results are contained in the Excel file.

The `CaptchaTheBots-webapp` folder contains the repository for our demo web application. The web application is purely for visualization purposes with regards to a simulated CAPTCHA environment. Instructions are located in the `readme`` for the folder.

The `model training` folder contains the notebooks used to train our models (CNN, Resnet50, Resnet101, VGG, VGG_aug) on the CIFAR10 dataset.

The `Noise tests` folder contains the notebooks used to evaluate the initial simple noise functions (Salt & Pepper, Speckle, Gaussian, Poisson) on our models. These results were not included in our final report.

The `Ranking Attacks` folder contains the notebooks that were utilized to construct the initial system for comparing and ranking different adversarial attacks in a CAPTCHA construction. See the `readme`` for further elaboration.

`Attacks_easy_medium_hard.ipynb` was used to generate the Easy, Medium, and Hard scenarios that were shown in our final report.

`StaticAccuracy` was used to derive the base accuracies for the models as well as the accuracies for the simulations of the 9-Grid CAPTCHA system.



# CaptchaTheBots Website

This is the sample web application that we created to showcase adversarial attacks and noise functions on real-life CAPTCHA images.
We used the Kaggle dataset of scraped reCaptcha Images: https://www.kaggle.com/datasets/mikhailma/test-dataset 
We use Django MVC Architecture to create three web pages. 


How to run?

Step 1. Create a virtual environment

virtalenv venv
> source venv/bin/activate

Step 2. Install requirements
> pip install -r requirements.txt

Step 3. Start Server
> python manage.py runserver

Application should be up on localhost:8000
CAPTCHA Page should be visible on http://localhost:8000/captcha/

The base page http://localhost:8000/ may not work because we have hidden the Google reCAPTCHA API KEYS and SECRETS for security reasons
