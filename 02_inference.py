import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

from torch.utils.data import DataLoader
from FlickrDataset import FlickrDataset
from CaptionTransformerDecoder import CaptionTransformerDecoder
from ProjectEmbeddingDimension import ProjectEmbeddingDimension
import wandb
import torch
import datetime
import torch
import wandb
import torch
import wandb
import torch
import os 
from tqdm import tqdm 
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
from dataset import *
from training import *

model = CaptionTransformerDecoder