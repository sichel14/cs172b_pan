import h5py
import torch
import sys
# from model import Net
import numpy as np
import random
# sys.path.append(r"C:\Users\89709\Desktop\pansharpening\zeroshot\experiment\Toolbox")
# import scipy.io as sio

# satellite = 'wv3/'
name = 19
# ckpt = 'weights/CSNET.pth'
# model = Net()
# weight = torch.load(ckpt)
# model.load_state_dict(weight)

file_path = "/Users/barry/Downloads/valid_wv3.h5"
dataset = h5py.File(file_path, 'r')
print(dataset.keys())
list_of_index = [i for i in range(1080)]
random.shuffle(list_of_index)
t_data = {}
t_data["gt"] = np.stack([dataset["gt"][i] for i in list_of_index[:(108 * 7)]])
t_data["ms"] = np.stack([dataset["ms"][i] for i in list_of_index[:(108 * 7)]])
t_data["lms"] = np.stack([dataset["lms"][i] for i in list_of_index[:(108 * 7)]])
t_data["pan"] = np.stack([dataset["pan"][i] for i in list_of_index[:(108 * 7)]])

v_data = {}
v_data["gt"] = np.stack([dataset["gt"][i] for i in list_of_index[(108 * 7):(108 * 9)]])
v_data["ms"] = np.stack([dataset["ms"][i] for i in list_of_index[(108 * 7):(108 * 9)]])
v_data["lms"] = np.stack([dataset["lms"][i] for i in list_of_index[(108 * 7):(108 * 9)]])
v_data["pan"] = np.stack([dataset["pan"][i] for i in list_of_index[(108 * 7):(108 * 9)]])

te_data = {}
te_data["gt"] = np.stack([dataset["gt"][i] for i in list_of_index[(108 * 9):]])
te_data["ms"] = np.stack([dataset["ms"][i] for i in list_of_index[(108 * 9):]])
te_data["lms"] = np.stack([dataset["lms"][i] for i in list_of_index[(108 * 9):]])
te_data["pan"] = np.stack([dataset["pan"][i] for i in list_of_index[(108 * 9):]])

print(type(te_data["gt"]))
print("test_data: ", te_data["pan"].shape)
print(dataset["pan"].shape)


train_file = h5py.File("/Users/barry/Downloads/train_real_wv3.h5", "w")
valid_file = h5py.File("/Users/barry/Downloads/valid_real_wv3.h5", "w")
test_file = h5py.File("/Users/barry/Downloads/test_real_wv3.h5", "w")


for key, value in te_data.items():
    if isinstance(value, np.ndarray):
        test_file.create_dataset(key, data = value)

for key, value in t_data.items():
    if isinstance(value, np.ndarray):
        train_file.create_dataset(key, data = value)

for key, value in v_data.items():
    if isinstance(value, np.ndarray):
        valid_file.create_dataset(key, data = value)
        


# train_dataset = train_file.create_dataset("data", data = t_data)
# print(list_of_index)
# print(dataset["gt"])
# print(dataset["pan"])
# print(dataset["ms"])
print(dataset["lms"][0].shape)
ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

ms = torch.from_numpy(ms).float()
lms = torch.from_numpy(lms).float()
pan = torch.from_numpy(pan).float()

ms = torch.unsqueeze(ms.float(), dim=0)
lms = torch.unsqueeze(lms.float(), dim=0)
pan = torch.unsqueeze(pan.float(), dim=0)

print(lms.shape)
# sr = model(pan, lms)

# I_SR = torch.squeeze(sr*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
# I_MS_LR = torch.squeeze(ms*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
# I_MS = torch.squeeze(lms*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
# I_PAN = torch.squeeze(pan*2047).cpu().detach().numpy()  # HxWxC
# sio.savemat('result/' + str(name) + '.mat', {'I_SR': I_SR, 'I_MS_LR': I_MS_LR, 'I_MS': I_MS, 'I_PAN': I_PAN})