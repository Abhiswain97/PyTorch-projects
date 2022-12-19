# PyTorch-Projects

a repo for self-learning projects in pytorch
## Contents: 

Install the requirements: `pip install -r requirements.txt`

1. AlexNet from scratch

    - Just `python Alexnet.py`

2. UNet from scratch

    - Unet
    - Attention UNet
    - UNet - 3D
    - Attention UNet - 3D
    
    How to run ? 
    - Download and unzip data-science-bowl-2018 data from kaggle
    - Unzip data-science-bowl-2018/stage1_train.zip into the data folder
    - Do `python LightningNuclei.py --attn=False/True` depending on if you want to use  Original UNet or you can refer the Jupyter notebook provided

3. Transformers
    - BERT imdb classification with PytorchLightning -> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16jIR2MtvNRRaON4xfIDQOsN5W0j817z3#scrollTo=e29ce689-a0b6-4570-83c3-92a103d97e05)

4. Hindi Character Recognition

    How to run ?
    - You can create your custom model in the `model.py` file. Make sure you initialize it at the bottom of the file to the `model` variable. Do not change the variable name. In case you do you will need to make sure to import it in the `train.py` file. Would be too much hassle, isn't it ?
    - Now to train the model with default params do, `python train.py`. You can also specify epochs and lr, using `python train.py --epochs <num-epochs> --lr <learning-rate>`

    The training output should be similar to this:
    ```
    (torch) C:\Users\abhi0\Desktop\PyTorch-projects\Hindi-Character-Recognition\src>python train.py --epochs 1 --lr 1e-5

    Training details:
    ------------------------
    Model: HNet
    Epochs: 1
    Optimizer: Adam
    Loss: CrossEntropyLoss
    Learning Rate: 1e-05
    Train-dataset samples: 62560
    Validation-dataset samples: 15640
    -------------------------

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1955/1955 [01:18<00:00, 24.76it/s] 
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 489/489 [00:15<00:00, 31.64it/s] 
    +-------+------------+-----------+----------+---------+
    | Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
    +-------+------------+-----------+----------+---------+
    |   1   |   2.347    |   41.56   |  1.533   |  60.198 |
    +-------+------------+-----------+----------+---------+
    ```       