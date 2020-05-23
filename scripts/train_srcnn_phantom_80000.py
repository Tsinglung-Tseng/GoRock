USE_GPU = 2

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')  
tf.config.set_visible_devices(physical_devices[USE_GPU], 'GPU')

use_device = physical_devices[USE_GPU]
tf.config.experimental.set_memory_growth(
    use_device, True
)

import matplotlib.pyplot as plt
import numpy as np
import json

import sys
sys.path.append("/mnt/users/qinglong/scaffold/dlsr")

from src.dataset import MNISTDataset, PhantomDataset
from src.dl_network.config import FrozenJSON
from src.dl_network.trainer import Trainer
from src.dl_network.models import SeqModel, ModelBuilder

from src.utils.bi_mapper import ConfigBiMapping
from src.utils.logger import LogAndPrint

dataset_config = {
    'class_name': 'PhantomDataset',
    'path': '/mnt/users/qinglong/data/phantom_80000.h5',
    'downsampling_ratio': 2,
    'test_portion': 0.1,
    'shuffle': 500,
    'batch_size': 32,
    'normalization': True
}

model_config = {
    "SeqModel": [
        {"UpSampling2D": {"size": (2,2)}}, 
        {"Conv2D": {
            "filters": 64, 
            "kernel_size": 9, 
            "activation": 'relu', 
            "input_shape": (128, 128,1), 
            "padding": 'same'
        }}, 
        {"Conv2D": {
            "filters": 32, 
            "kernel_size": 1, 
            "activation": 'relu', 
            "padding": 'same'
        }},
        {"Conv2D": {
            "filters": 1,
            "kernel_size": 5,
            "activation": 'relu',
            "padding": 'same'
        }}
    ],
    "comment": "SRCNN_learning_rate_0.00001"
}

trainner_config = {
    'epoch': 5,
    'loss_object': 'MSE',
    'optimizer': 'Adam_SRCNN_1eNeg5',
    'train_loss': 'Mean', #{'Mean': {'name': 'train_loss'}},
    'train_accuracy': 'MeanSquaredError',
    'test_loss': 'Mean', #{'Mean': {'name': 'train_loss'}},
    'test_accuracy': 'MeanSquaredError',
    'commit': 'SRCNN'
}

# pd = PhantomDataset(phantom_dataset_config)

tr = Trainer(
    dataset=PhantomDataset(dataset_config), 
    model=ModelBuilder(model_config)(),
    config=ConfigBiMapping.load(trainner_config),
    logger=LogAndPrint
)

tr.run()
