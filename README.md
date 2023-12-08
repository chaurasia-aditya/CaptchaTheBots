# CaptchaTheBots

This is a project for the CS 6220 course of Big Data Systems and Analytics.

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
