# This will use ingest.py to grab a directory of PDF's and parse it for use to train an LLM

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import sys
import ingest.py as noms

# get the PDF Directory, make sure it's valid, grab a list of the PDFs
files = noms.check_dir()
# Snag all the text from each pdf in List
textybois = noms.parse_dir(files)
# take the textybois and clean them up for LLMs
swiffer = noms.clean(textybois)

# now that we have, hopefully, clean textybois we can format it into a way that AI can chomp on it



