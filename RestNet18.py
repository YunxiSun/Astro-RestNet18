from torch.nn import functional as F
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2

#定义参数
EPOCH=100 #训练epoch次数
BATCH_SIZE=3 #批训练的数量
LR=0.0001 #学习率

mask_img_path='./train/set_0'#定义路径
test_img_path='./train/set_1'
model_path='./model.pkl'
LOSS=[]

class ToTensor(object):
    def __call__(self, sample):
        image,labels=sample['image'],sample['labeles']
        return {'image':torch.from_numpy(image),
                'labels':torch.from_numpy(labels)}

data_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])#数据转换为tensor

def removeneg(im):
    """
    remove negativ value of array
    """
    im2 = np.copy(im)
    arr = np.isnan(im2)
    im2[arr] = 0
    arr2 = np.isinf(im2)
    im2[arr2] = 0
    im2 = np.maximum(im2,0)
    # im2[im1<2] =0
    # im2 = np.minimum(im2,1e12)
    # im2 = signal.medfilt2d(im2,kernel_size=5)

    return im2

#自定义dataset
class SDataset(Dataset):
    def __init__(self,mask_img_path,transform=None):
        self.mask_img_path = mask_img_path
        self.transform = transform
        self.mask_img = os.listdir(mask_img_path)
    def __len__(self):
        return len(self.mask_img)
    def __getitem__(self, idx):
        label_img_name=os.path.join(mask_img_path,"heap_{}.fits".format(idx))
        hdu=fits.open(label_img_name)
        hdu.verify('fix')
        mask_img=hdu[0].data
        image_raw = np.zeros([6,1024, 1024], dtype=np.float64)
        label_img_raw=np.zeros([1,1024,1024],dtype=np.float64)
        label_img_raw[0,:, :] = removeneg(mask_img[6].data)
        label_img_raw=np.log(label_img_raw+1)
        for num in range(0,5):
            image_raw[num,:,:]=removeneg(mask_img[num].data)
        image_raw=np.log(image_raw+1)
        sample = {'image': image_raw, 'labels': label_img_raw}
        return sample

#自定义卷积模型

class RestNetBasciBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(RestNetBasciBlock, self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
    def forward(self,x):
        output=self.conv1(x)
        output=F.relu(self.bn1(output))
        output=self.conv2(output)
        output=self.bn2(output)
        return F.relu(x+output)

class RestNetDownBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.extra=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride[0],padding=0),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self,x):
        extra_x=self.extra(x)
        output=self.conv1(x)
        out=F.relu(self.bn1(output))

        out=self.conv2(out)
        out=self.bn2(out)
        return F.relu(extra_x+out)

class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1=nn.Conv2d(6,64,kernel_size=7,stride=2,padding=3)
        self.conv2=nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=nn.Sequential(
            RestNetBasciBlock(64,64,1),
            RestNetBasciBlock(64,64,1)
        )
        self.layer2=nn.Sequential(
            RestNetDownBlock(64,128,[2,1]),
            RestNetBasciBlock(128,128,1)
        )
        self.layer3=nn.Sequential(
            RestNetDownBlock(128,256,[2,1]),
            RestNetBasciBlock(256,256,1)
        )
        self.layer4=nn.Sequential(
            RestNetDownBlock(256,521,[2,1]),
            RestNetBasciBlock(512,512,1)
        )
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1024,1024))
        #self.fc=nn.Linear(512,10)

    def forward(self,x):
        out=self.conv1(x)
        #ou=self.conv2(y)
        out=self.layer1(out)
        #out=self.layer2(out)
        #out=self.layer3(out)
        out=self.avgpool(out)
        #ou=self.avgpool(ou)
        #out=self.layer4(out)
        #out=out.reshape(x.shape[0],-1)
        #out=self.fc(out)

        return out

def PL1(data,label):
    data = data.cpu()
    label = label.cpu()
    # data=torch.mul(255).byte
    # data=data.numpy()
    data = data.detach().numpy()
    label = label.detach().numpy()
    loss1 = label[0][0] - data[0][0]
    x1=range(0,epoch+1)
    y1=LOSS
    plt.ion()
    plt.subplot(1, 4, 1)
    plt.imshow(data[0][0], cmap='gray', label='train')
    plt.title('train')
    plt.subplot(1, 4, 2)
    plt.imshow(label[0][0], cmap='gray', label='label')
    plt.title('label')
    plt.subplot(1, 4, 3)
    plt.imshow(loss1, cmap='gray', label='loss_image')
    plt.title('loss')
    plt.subplot(1,4,4)
    plt.plot(x1,y1,'-')
    plt.pause(1)
    plt.show()

#导入数据
traindata=SDataset(mask_img_path,transform=data_transform)
testdata=SDataset(test_img_path,transform=data_transform)
#print(test1['1'])
train_loder=DataLoader(dataset=traindata,batch_size=BATCH_SIZE,shuffle=True)
test_loder=DataLoader(dataset=testdata,batch_size=1,shuffle=True)

#导入CUDA
device=torch.device("cuda:0")
#model=CNN().to(device)
#if(os.path.exists('model.pkl')):

    #model=torch.load('\model.pkl')
#else:
    #model=FCN().to(device)
#model=torch.load('\model.pkl')
model=RestNet18()
if(os.path.exists(model_path)):
    model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

#损失函数
criterion=nn.L1Loss()

#优化器
optimizer=optim.Adam(model.parameters(),lr=LR)


#训练
for epoch in range(EPOCH):
    start_time=time.time()
    for i,data in enumerate(train_loder):
        inputs=data['image']
        inputs=inputs.type(torch.FloatTensor)
        labels=data['labels']
        labels=labels.type(torch.FloatTensor)
        inputs,labels=inputs.to(device),labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        LOSS.append(loss)
        if(epoch%10==0):
          PL1(outputs,labels)
        #清空上一轮梯度
        optimizer.zero_grad()
        #方向传播
        loss.backward()
        #参数更新
        optimizer.step()
    print('epoch{}  loss:{:.4f}  time:{:.4f}'.format(epoch+1,loss.item(),time.time()-start_time))

torch.save(model.state_dict(),model_path)