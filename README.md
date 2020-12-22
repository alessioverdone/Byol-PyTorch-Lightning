# Byol-PyTorch-Lightning
Implementation of the Self-supervised system presented in "Bootstrap you own latent" from DeepMind. 

# BYOL implementation on pytorch 

BYOL implementation for image classification produce good images representations to be used in subsequnt subtasks. The system is composed by two networks: an online network composed by encoder, projector and predictor and a target network composed by encoder and projection only. The goal is to predict with online network target projections. At the end we use only online encoder to represent images. 
The implementation was built following pytorch-lightning guidelines and key sections are identifiable in the code.


We tested two version for the BYOL: a classical setup and a modified version which use multiple loss in the internal structure of the encoder. Moreover for the evaluation we use two different kind of representations for images: classical representation which use encoder avgpool layer (we use a ResNet as encoder,so it's the second-last layer) and a modified representaton which use a concatenation of avgpool layer and internal layers of the encoder, in our work Layer2.0 and Layer3.0 of the ResNet encoder. 

To evaluate representations produced we perform both linear evaluation and fine-tuning. 

The byol_training folder in the repository include both classical and multiple loss training file.
Evaluation folder is divided in linear evaluation and fine-tuning folder and in each of them there are 3 files to perform:

                                             1. evaluation on classical representations extracted from classical byol architecture        
                                             2. evaluation on classical representations extracted from multiple loss training 
                                             3. evaluation on combined representations extracted from classical byol architecture
                                             4. evaluation on combined representations extracted from multiple loss training
                                             
                                             
# Create representations
To produce representaton execute byol training filw which will save at the end all the byol class


''' python byol_training.py '''


To evaluate our system we perform evaluations on CIFAR10 dataset.
