# PyTorch-Projects

A repo for self-learning projects in pytorch
## Contents: 

Install the requirements: `pip install -r requirements.txt`

1. AlexNet from scratch

    - Just `python Alexnet.py`

2. Transformers
    - BERT imdb classification with PytorchLightning -> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16jIR2MtvNRRaON4xfIDQOsN5W0j817z3#scrollTo=e29ce689-a0b6-4570-83c3-92a103d97e05)

3. Hindi Character Recognition

    Getting the data:
    - Download the data from [here](https://www.kaggle.com/datasets/suvooo/hindi-character-recognition)
    - Unzip it. You need to split the data into 4 different directories, since we are training for Hindi digits & letters separately.
    ![image](https://user-images.githubusercontent.com/54038552/209815855-cd629bdd-5a9a-474e-8ad6-1d4df1954fdc.png)
    
    How to run ?
    - You can create your custom model in the `model.py` file. Make sure you initialize it at the bottom of the file to the `model` variable. Do not change the variable name. In case you do you will need to make sure to import it in the `train.py` file. Would be too much hassle, isn't it ?
    - Now to train the model with default params do, `python train.py`. You can also specify epochs and lr. Most important, is the `model_type`
    - To train do, `python train.py --epochs <num-epochs> --lr <learning-rate> --model_type <type-of-model>`
        
