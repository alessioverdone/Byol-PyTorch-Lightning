"""byol_training_multiple_loss.ipynb"""

import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
import torchvision
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split


import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from torch import nn
import torch.nn.functional as F

from kornia import augmentation as augs
from kornia import filters,color 
import copy
import random
from functools import wraps

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""# **BYOL Section**

## BYOL **model**
"""

#Get mean ans std of dataset
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)#, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, 
                 layer_name_list = [-2]):
        super().__init__()
        self.net = net
        self.layer_name_list = layer_name_list
        self.number_heads = len(self.layer_name_list)

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = []
        self.hook_registered = False

        self.mlp1 = MLP(1024, self.projection_size)
        self.mlp2 = MLP(2048, self.projection_size)

    def _find_layer(self,i):
        name_or_id = self.layer_name_list[i]
        if type(name_or_id) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(name_or_id, None)
        elif type(name_or_id) == int:
            children = [*self.net.children()]
            return children[name_or_id]
        return None

    def _hook(self, _, __, output):
        self.hidden.append(flatten(output))

    def _register_hook(self):
        for i in range(self.number_heads):
            layer = self._find_layer(i)
            assert layer is not None, f'hidden layer ({self.layer_name_list[i]}) not found'
            layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        # if self.layer == -1:
        #     return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = []
        assert hidden is not None, f'hidden layer {self.layer_name_list} never emitted an output'
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)# list of outputs 
        main_representation = representation[-1]#avgpool
        if self.number_heads > 1:
            small_repr_1 = representation[-2]
            projector_1 = self.mlp1(small_repr_1)
            small_repr_2 = representation[-3]
            projector_2 = self.mlp2(small_repr_2)


        projector = self._get_projector(main_representation)
        projection = projector(main_representation)
        return projection, projector_1, projector_2

# main class
class BYOL(nn.Module):
    def __init__(self, net, image_size=32,
                layer_name_list = [-2],
                 projection_size = 256,
                 projection_hidden_size = 4096,
                 augment_fn = None,
                 moving_average_decay = 0.99,
                 device_ = 'cuda',
                 number_of_classes = 10,
                 mean_data = torch.tensor([0.485, 0.456, 0.406]),
                 std_data = torch.tensor([0.229, 0.224, 0.225])):
        super().__init__()

        # default SimCLR augmentation

        DEFAULT_AUG = nn.Sequential(
            RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
            augs.RandomGrayscale(p=0.2),
            augs.RandomHorizontalFlip(),
            RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
            augs.RandomResizedCrop((image_size, image_size)),
            augs.Normalize(mean=mean_data, std=std_data)
        )

        self.augment = default(augment_fn, DEFAULT_AUG)
        self.device = device_

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer_name_list=layer_name_list).to(self.device)
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size).to(self.device)
        self.online_predictor1 = MLP(projection_size, projection_size, 512).to(self.device)
        self.online_predictor2 = MLP(projection_size, projection_size, 512).to(self.device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size).to(self.device))


    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder).to(self.device)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, x):
        image_one, image_two = self.augment(x), self.augment(x)

        online_proj_one, online_proj1_one, online_proj2_one = self.online_encoder(image_one)
        online_proj_two, online_proj1_two, online_proj2_two = self.online_encoder(image_two)

        online_pred2_one = self.online_predictor2(online_proj2_one)
        online_pred2_two = self.online_predictor2(online_proj2_two)

        online_pred1_one = self.online_predictor1(online_proj1_one)
        online_pred1_two = self.online_predictor1(online_proj1_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            target_proj_one, target_proj1_one, target_proj2_one = target_encoder(image_one)#.to(self.device)
            target_proj_two, target_proj1_two, target_proj2_two = target_encoder(image_two)#.to(self.device)

        #Principal loss
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        loss = loss_one + loss_two

        #Loss internal  1
        loss_proj1_one = loss_fn(online_pred1_one, target_proj1_two.detach())
        loss_proj1_two = loss_fn(online_pred1_two, target_proj1_one.detach())
        loss_proj1 = loss_proj1_one + loss_proj1_two

        #Loss internal  2
        loss_proj2_one = loss_fn(online_pred2_one, target_proj2_two.detach())
        loss_proj2_two = loss_fn(online_pred2_two, target_proj2_one.detach())
        loss_proj2 = loss_proj2_one + loss_proj2_two


        #Total loss
        final_loss = loss + loss_proj1 + loss_proj2
        return final_loss.mean()

        
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""## **Lt_model**"""

#LIghting wrapper
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, params, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)
        self.params = params

    def forward(self, images):
        return self.learner(images)

    def training_step(self, x, _):
        images, _ = x
        loss = self.forward(images)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, x, _):
        images, _ = x
        loss = self.forward(images)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, x, _):
        images, _ = x
        loss = self.forward(images)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params.LR)

    def on_before_zero_grad(self, _):
        self.learner.update_moving_average()

        
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""## **Data**"""

#Dataset
class Data_Module(pl.LightningDataModule):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.transform_train = transforms.Compose([
            transforms.Resize(self.params.IMAGE_SIZE),
            transforms.CenterCrop(self.params.IMAGE_SIZE),
            transforms.ToTensor(),
            ])
        
        self.transform_test = transforms.Compose([
            transforms.Resize(self.params.IMAGE_SIZE),
            transforms.CenterCrop(self.params.IMAGE_SIZE),
            transforms.ToTensor(),
            ])
        

    def prepare_data(self):
        #download
        CIFAR10(root=self.params.PATH_DATASET,  train=True, download=True)
        CIFAR10(root=self.params.PATH_DATASET, train=False, download=True)

    def setup(self, stage=None):
        #Train set 
        self.train_set = CIFAR10(self.params.PATH_DATASET, train=True, transform=self.transform_train)

        #Val and test set
        test_val_set = CIFAR10(self.params.PATH_DATASET, train=False, transform=self.transform_test)

        len_test_val_set = len(test_val_set)
        split = int(len_test_val_set/2)
        self.val_test, self.test_set = random_split(test_val_set, [split, split], generator=torch.Generator().manual_seed(42))
        assert len(self.val_test) == len(self.test_set)


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.params.BATCH_SIZE, shuffle=True)#, num_workers=self.params.NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_test, batch_size=self.params.BATCH_SIZE)#, num_workers=self.params.NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.params.BATCH_SIZE)#, num_workers=self.params.NUM_WORKERS)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
"""## **Callback**"""

class My_Callback(Callback):

    def __init__(self, params):
        self.params = params

    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_epoch_end(self,trainer, pl_module):
        ep = pl_module.current_epoch
        if ep  % 100 ==0: #save checkpoint at each 100 epochs
            name_checkpoint=self.params.PATH_TO_SAVE + '/weight_byol_multiple_loss' + str(ep)+ 'ep.ckpt'
            trainer.save_checkpoint(name_checkpoint)
            print('Weights saved!!')
            
    def on_train_end(self, trainer, pl_module):
        name_final_save=params.PATH_TO_SAVE + '/final_weights_multiple_loss.ckpt'
        trainer.save_checkpoint(name_final_save)
        print('Weights saved!!')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


"""## **Parameters**"""

class hparams():

    NAME_PROJECT = 'Byol_training_multiple_loss'
    BATCH_SIZE = 2048
    EPOCHS     = 1000
    LR         = 3e-4
    NUMBER_OF_CLASSES = 10
    NUM_GPUS   = 1
    IMAGE_SIZE = 32
    #NUM_WORKERS = multiprocessing.cpu_count()
    mean_dataset = torch.tensor([0.4914, 0.4822, 0.4465])#imagenet_values
    std_dataset = torch.tensor([0.2023, 0.1994, 0.2010])#imagenet_values

    PATH_TO_SAVE = '' #path where save system checkpoints
    PATH_DATASET = '' #CIFAR10 dataset location 

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""## **Main**"""

#Parameters
params = hparams()
#Net
resnet = models.resnet18(pretrained=False)
#Data
data_module = Data_Module(params)
data_module.prepare_data()
data_module.setup()

# params.mean_dataset, params.std_dataset = get_mean_and_std(data_module.train_set)# compute mean and std of current dataset

#Model
model = SelfSupervisedLearner(
    params,
    resnet,
    image_size = params.IMAGE_SIZE,
    layer_name_list = ['avgpool', 'layer3.0', 'layer2.0'], #list of layers where we set projection and prediction heads 
    projection_size = 256,
    projection_hidden_size = 4096,
    moving_average_decay = 0.99,
    mean_data = params.mean_dataset,
    std_data = params.std_dataset,
    number_of_classes = params.NUMBER_OF_CLASSES)

#Logger
# wandb.login()
# wandb.init()
# wandb_logger = WandbLogger(project=params.NAME_PROJECT, log_model=True)
# wandb_logger.watch(model, log='gradients', log_freq=100)

#Trainer     
trainer = pl.Trainer(
    #fast_dev_run=True,
    gpus = params.NUM_GPUS,
    #logger = wandb_logger,
    #resume_from_checkpoint= , #path to saved checkpoints, use this flag to resume training
    check_val_every_n_epoch=10,
    max_epochs = params.EPOCHS,
    accumulate_grad_batches = 1,
    callbacks = [My_Callback(params)])

if __name__ == '__main__':
    #Fit model
    trainer.fit(model, data_module)
    #Test model
    trainer.test(model)
    #wandb.finish()
    
    