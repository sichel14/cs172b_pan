import h5py
import torch
import sys
from model import Net
import numpy as np
sys.path.append(r"C:\Users\89709\Desktop\pansharpening\zeroshot\experiment\Toolbox")
import scipy.io as sio

satellite = 'wv3/'
name = 19
ckpt = 'weights/CSNET.pth'
model = Net()
weight = torch.load(ckpt)
model.load_state_dict(weight)

file_path = 'data/wv3/test_full.h5'
dataset = h5py.File(file_path, 'r')
ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

ms = torch.from_numpy(ms).float()
lms = torch.from_numpy(lms).float()
pan = torch.from_numpy(pan).float()

ms = torch.unsqueeze(ms.float(), dim=0)
lms = torch.unsqueeze(lms.float(), dim=0)
pan = torch.unsqueeze(pan.float(), dim=0)

print(lms.shape[1])
sr = model(pan, lms)

I_SR = torch.squeeze(sr*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
I_MS_LR = torch.squeeze(ms*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
I_MS = torch.squeeze(lms*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
I_PAN = torch.squeeze(pan*2047).cpu().detach().numpy()  # HxWxC
sio.savemat('result/' + str(name) + '.mat', {'I_SR': I_SR, 'I_MS_LR': I_MS_LR, 'I_MS': I_MS, 'I_PAN': I_PAN})
