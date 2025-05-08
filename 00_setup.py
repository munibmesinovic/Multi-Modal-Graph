
!pip install torchtuples
!pip install pycox

import pandas as pd
from itertools import islice
import random
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from functools import reduce
import datetime
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

import torch # For building the networks
from torch import nn
import torch.nn.functional as F
import torchtuples as tt # Some useful functions
from transformers import BertTokenizer, BertModel
from torch.cuda.amp import autocast

import torch.nn.functional as F
import torchtuples as tt # Some useful functions

from pycox.models import LogisticHazard
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.evaluation import EvalSurv

seed = 250
random.seed(seed)
np.random.seed(seed)

import seaborn as sn
sn.set_theme(style="white", palette="rocket_r")
