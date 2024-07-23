import os
from PIL import Image
from torchvision import transforms
import torch.utils.data as Data
import random
data_category_size=12
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path= os.path.join(current_dir, 'test_data')
train_size=data_category_size*100         # 0<train_size<7440 
test_size=12*240-train_size
class CustomDataset(Data.Dataset):
    def __init__(self, classes,images,size, transform=None):
        self.transform = transform
        self.classes = classes
        self.images=images
        self.size=size
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('L')  # 读取黑白图片并转换成灰度图（单通道）

        if self.transform:
            image = self.transform(image)

        return image, label

def getClassesAndImages(filePath):
    classes = sorted(os.listdir(filePath), key=lambda x: int(x))
    images,train_images,test_images= [],[],[]  # 存储图片路径和对应的标签
        
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(filePath, class_name)
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)

            images.append((img_path, i))  # (图片路径, 类别索引)            
    for j in range(0,data_category_size):
        for i in range (0,int(train_size/data_category_size)):
            train_images.append(images[j*240+i])    
    test_images=list(set(images)-set(train_images))
    random.shuffle(train_images)
    random.shuffle(test_images)
    return classes,train_images,test_images

# 定义预处理操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换成Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化处理
])

def getTestDataLoader():      
    testset = CustomDataset(classes,test_images,test_size,transform=transform)
    testloader = Data.DataLoader(testset, batch_size=24, shuffle=True)
    print("DataLoader Ready.")
    return testloader

def getTrainDataLoader():
        
    trainset = CustomDataset(classes,train_images,train_size,transform=transform)
    trainloader = Data.DataLoader(trainset, batch_size=24, shuffle=True)#,num_workers=3,num_workers=3
    print("DataLoader Ready.")
    return trainloader

classes,train_images,test_images=getClassesAndImages(data_path)
