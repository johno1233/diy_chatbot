# This will use ingest.py to grab a directory of PDF's and parse it for use to train an LLM

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import sys
from ingest import nomnom

# create the file of tokens for training the model (stored as .txt)

nomnom()




