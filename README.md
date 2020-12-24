# Byol-PyTorch-Lightning
Implementation of the Self-supervised system presented in "Bootstrap you own latent" from DeepMind. 

## Getting started

BYOL implementation for image classification to produce images representations useful for subtasks. The system is based on two networks: an online network composed by encoder, projector and predictor and a target network composed by encoder and projection only. Online network has the goal to predict target representations. 
Only online network encoder is used to produce representations.

We tested two version for the BYOL: a classical setup and a modified version which use multiple loss in the internal structure of the encoder. 
We use also two different type of representations: classical representation which use encoder's avgpool layer (we use a ResNet as encoder,so the second-last layer) and a modified representaton which use a concatenation of avgpool layer and internal layers of the encoder(in our work Layer2.0 and Layer3.0 of the ResNet encoder).

## Evaluation
To evaluate representations produced we perform both linear evaluation and fine-tuning. 

The byol_training folder in the repository include both classical and multiple loss training file.
Evaluation folder is divided in linear evaluation and fine-tuning folder and in each of them there are 4 files to perform:

   1. evaluation on classical representations extracted from classical byol architecture `...evaluation_base.py`        
   2. evaluation on combined representations extracted from classical byol architecture `...evaluation_base_variant.py`
   3. evaluation on classical representations extracted from multiple loss architecture `evaluation_multiple_loss.py`
   4. evaluation on combined representations extracted from multiple loss architecture `...evaluation_multiple_loss_variant.py`

## Prerequisites
The code is implemented in pytorch-lightning, a new deep learning framework based on torch.


## Running the tests
                                             
We perform our experiments on CIFAR10 dataset. 
To produce representations execute byol training file 

` python byol_training.py `

Then in the evaluation files specify the path to byol's weights saved previously in PATH_TO_SAVE_DATA and run like

` python linear_evaluation_classic_to_multiple.py `

to obtain the accuracy of the representations

# Acknowledgments
For the code I take inspiration from https://github.com/lucidrains/byol-pytorch

