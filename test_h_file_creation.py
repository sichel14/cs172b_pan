import h5py
import torch
import sys
# from model import Net
import numpy as np
import random

file_path = "/Users/barry/Downloads/test_real_wv3.h5"

dataset = h5py.File(file_path, 'r')
print(type(dataset))
print(dataset.keys())
print(dataset["gt"].shape)

