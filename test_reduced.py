import h5py
import torch
import sys
from model import Net
import numpy as np
# sys.path.append(r"C:\Users\89709\Desktop\pansharpening\zeroshot\experiment\Toolbox")
import scipy.io as sio
device=torch.device('cuda:1')
# satellite = 'wv3/'
satellite = 'wv3'
# name = 19
# ckpt = '/home/wutong/proj/HJM_Pansharpening/Spectral_aware/Channel_Conv/weights/CSNET.pth'
model = Net().to(device)
# weight = torch.load(ckpt)
# model.load_state_dict(weight)
model.load_state_dict(torch.load('./weights/CSNET0.pth'))

# file_path = 'data/wv3/test_reduce.h5'
file_path = 'E:/HJM_Datasets/new_pan/test data/h5/WV3/reduce_examples/test_wv3_multiExm1.h5'

def load_test_data(file_path):
    dataset = h5py.File(file_path, 'r')
    ms = np.array(dataset['ms'], dtype=np.float32) / 2047.0
    lms = np.array(dataset['lms'], dtype=np.float32) / 2047.0
    pan = np.array(dataset['pan'], dtype=np.float32) / 2047.0
    gt = np.array(dataset['gt'], dtype=np.float32) / 2047




dataset = h5py.File(file_path, 'r')
print(dataset)
print(type(dataset))
# print("---keys---")
# print(dataset.keys())
# print("---key size---")
# print(len(dataset['pan'][:]))
# for i in range(len(dataset['pan'][:])):
#     print(dataset['pan'][i].shape)
#     print(dataset['ms'][i].shape)
#     print(dataset['lms'][i].shape)

# ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
# lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
# pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0
# gt = np.array(dataset['gt'][name], dtype=np.float32)

ms = np.array(dataset['ms'], dtype=np.float32) / 2047.0
lms = np.array(dataset['lms'], dtype=np.float32) / 2047.0
pan = np.array(dataset['pan'], dtype=np.float32) / 2047.0
gt = np.array(dataset['gt'], dtype=np.float32)

ms = torch.from_numpy(ms).float().to(device)
lms = torch.from_numpy(lms).float().to(device)
pan = torch.from_numpy(pan).float().to(device)
# gt = torch.from_numpy(gt)


model.eval()
with torch.no_grad():
    out = model(pan, lms)
    I_SR = torch.squeeze(out * 2047).cpu().detach().numpy()  # HxWxC
    I_MS_LR = torch.squeeze(ms * 2047).cpu().detach().numpy()  # HxWxC
    I_MS = torch.squeeze(lms * 2047).cpu().detach().numpy()  # HxWxC
    I_PAN = torch.squeeze(pan * 2047).cpu().detach().numpy()  # HxWxC
    I_GT = gt # HxWxC
    sio.savemat('./result/' + satellite + '.mat',
                {'I_SR': I_SR, 'I_MS_LR': I_MS_LR, 'I_MS': I_MS, 'I_PAN': I_PAN, 'I_GT': I_GT})



# out = model(pan, lms)
#
# I_SR = torch.squeeze(out*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
# I_MS_LR = torch.squeeze(ms*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
# I_MS = torch.squeeze(lms*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
# I_PAN = torch.squeeze(pan*2047).cpu().detach().numpy()  # HxWxC
# I_GT = gt.transpose(1, 2, 0)  # HxWxC
# # sio.savemat('result/' + str(name) + '.mat', {'I_SR': I_SR, 'I_MS_LR': I_MS_LR, 'I_MS': I_MS, 'I_PAN': I_PAN, 'I_GT': I_GT})
# sio.savemat('./result/' + satellite + '.mat', {'I_SR': I_SR, 'I_MS_LR': I_MS_LR, 'I_MS': I_MS, 'I_PAN': I_PAN, 'I_GT': I_GT})









