
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import xarray as xr
from pathlib import Path
import pickle
import numpy as np

class Z500_Dataset(Dataset):
    def __init__(self, data_dir, mode, lead_time, continuous=False, multistep=False, num_step=0):
        self.data = xr.open_mfdataset(f'{data_dir}/{mode}/*500hPa*.nc', combine='by_coords')['z'].values
        self.clim = np.mean(self.data, axis=0)
        self.mode = mode
        # 预报时效，若为continuous则为一个数组
        self.lead_time = lead_time
        # 是否训练continuous模型
        self.continuous = continuous
        # 是否训练多步预报模型
        self.multistep = multistep
        # 训练的步长
        self.num_step = num_step

        # data transformations
        with open(Path(data_dir)/f'scaler.pkl', 'rb') as f:
            item = pickle.load(f)
            self.mean = item['mean']
            self.std = item['std']
            f.close()
        
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((self.mean,), (self.std,))]
        )
        self.transforms_tensor = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        # 如果训练的是continuous的模型，则验证集需要把每一个步长都计算到
        if self.continuous:
            if self.mode != 'train':
                return len(self.lead_time) * (self.data.shape[0] - self.lead_time[-1])
            else:
                return self.data.shape[0] - self.lead_time[-1]
        # 如果训练的是multistep的模型，则需要预留多步的标签值
        elif self.multistep:
            return self.data[:-self.lead_time * self.num_step].shape[0]
        else:
            return self.data[:-self.lead_time].shape[0]

    def __getitem__(self, idx):
        # 如果训练continuous的模型，那么从lead_time中随机取一个值作为预报时效，与输入数据拼接作为模型输入
        if self.continuous:
            # 如果时训练模式，每一个样本的预报步长都是随机从lead_time中取的
            if self.mode == 'train':
                inputs = self.data[idx % self.__len__()]
                dt = int(np.random.choice(self.lead_time, 1, p=len(self.lead_time)*[1/len(self.lead_time)]))
                outputs = self.data[idx % self.__len__() + dt]
            # 如果是验证或者测试模式，那么每个预报步长都要遍历所有样本
            else:
                inputs = self.data[idx % (self.__len__() // len(self.lead_time))]
                dt = self.lead_time[idx // (self.__len__() // len(self.lead_time))]
                outputs = self.data[idx % (self.__len__() // len(self.lead_time)) + dt]
            # 标签需要从当前输入往后加上dt
            dt_shape = inputs.shape
            dt = dt * np.ones(dt_shape) / 100
            # 将dt与输入拼接
            inputs = np.concatenate([np.expand_dims(inputs, axis=0), np.expand_dims(dt, axis=0)], axis=0).astype(np.float32)
        # 如果训练多部预测模型，那么输出数据需要是间隔lead_time的一组数据
        elif self.multistep:
            inputs = self.data[idx]
            outputs = self.data[idx+self.lead_time:idx+(self.num_step+1)*self.lead_time:self.lead_time]
        # 如果只是训练direct predict模型，那么输出就为输入时刻往后计算lead_time
        else:
            inputs = self.data[idx]
            outputs = self.data[idx+self.lead_time]
        # clim数据用于计算ACC
        clim = self.clim
        if self.transforms:
            # 如果是多通道数据经过transforms后需要转换维度
            if self.continuous:
                inputs = self.transforms(inputs).permute(1, 2, 0)
            else:
                inputs = self.transforms(inputs)
            if self.multistep:
                outputs = self.transforms(outputs).permute(1, 2, 0)
            else:
                outputs = self.transforms(outputs)
        if self.transforms_tensor:
            clim = self.transforms_tensor(clim.astype(np.float32)) 
            
        return inputs, outputs, clim