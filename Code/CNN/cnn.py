import torch
import customData as customData
import torch.nn as nn
import UserInterface
import os

torch.manual_seed(1)
# 设置超参数
epoches =   0
batch_size = 2
learning_rate = 0.001
patience=5
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path= os.path.join(current_dir, 'cnn.pth')
dropout_prob=0.2

# 搭建CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()   # 继承__init__功能
        # 第一层卷积
        self.dropout = nn.Dropout(dropout_prob) #防止过拟合
        self.conv1 = nn.Sequential(
            # 输入[1,28,28]
            nn.Conv2d(
                in_channels=1,    # 输入图片的通道数
                out_channels=16,  # 输出图片的通道数
                kernel_size=5,    # 卷积核5x5
                stride=1,         # 卷积核步长为1
                padding=2,        # 给图外边补上0,共2圈,下一层的大小仍为28*28
            ),
            # 经过卷积层 输出[16,28,28] 传入池化层
            #nn.Dropout(0.2),    #防止过拟合
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # pooling  输出[16,14,14] 传入下一个卷积
        )
        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,   
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 经过卷积 输出[32, 14, 14] 传入池化层
            nn.Dropout(0.2),    
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # pooling 输出[32,7,7] 传入输出层
        )
        # 输出层
        self.output = nn.Linear(in_features=32*7*7, out_features=12)

    def forward(self, x):           #x [batch,1,28,28]
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)           # x变成四维张量[batch, 32,7,7]
        x = x.view(x.size(0), -1)   # 将输入特征向量展平  经过线性变换得到输出
        output = self.output(x)     
        return output



def main():
    # cnn 实例化
    cnn = CNN()
    if(UserInterface.isLoadFromFile()):
        cnn.load_state_dict(torch.load(save_path))
    print(cnn)
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()#mse_loss = nn.MSELoss()
    train_loader=customData.getTrainDataLoader()
    test_loader=customData.getTestDataLoader()
    
    train(cnn,train_loader,test_loader,optimizer,loss_function)
    test(cnn,test_loader,loss_function)
    if(UserInterface.isWriteToFile()):
        torch.save(cnn.state_dict(), save_path)
    

def train(cnn,train_loader,test_loader,optimizer,loss_function):
    # 开始训练
    maxAccuracy=0
    maxEpoch=0
    drop=0
    epochError=0
    for epoch in range(epoches):
        print("进行第{}个epoch".format(epoch+1))
        running_loss = 0.0
        # 枚举enumerate(trainloader)
        for i,(batch_inputs, batch_labels) in enumerate(train_loader):
            # 清空梯度值
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = cnn(batch_inputs)
            loss = loss_function(outputs, batch_labels)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 统计平均损失值
            running_loss += loss.item()
            if i % batch_size == 0:    
                print('[%d, %5d] loss: %.9f' %
                    (epoch + 1,  i+1, running_loss / batch_size))
                epochError+=running_loss / batch_size
            running_loss = 0.0
        '''drawImage(epoch,epochError)'''
        accuracy=test(cnn,test_loader,loss_function)
        if(accuracy>maxAccuracy):
            maxAccuracy=accuracy
            maxEpoch=epoch
        else :
            drop+=1
        if(drop>patience or maxAccuracy>0.999):
            print('at epoch ',maxEpoch,' max accuracy: ',maxAccuracy)
            break
    
    print('Finished Training')

def test(cnn,test_loader,loss_function):
    # 开始测试
    # 枚举enumerate(trainloader)
    cnn.eval()  # 将模型设为评估模式，不使用 dropout 和 batch normalization
    correct = 0  # 记录正确预测的样本数量
    total = 0  # 记录总样本数量
    running_loss = 0.0
    with torch.no_grad():  # 不计算梯度
        for batch_inputs, batch_labels in test_loader:
            outputs = cnn(batch_inputs)
            loss = loss_function(outputs, batch_labels)
            running_loss += loss.item()          
            _, predicted = torch.max(outputs, 1)  # 获取预测值中的最大值和对应的索引
            correct += (predicted == batch_labels).sum().item()  # 统计正确预测的样本数量
            total += batch_labels.size(0)  # 统计总样本数量   
    print('Test Loss: ', running_loss / len(test_loader))
    print('Accuracy: {:.5%}'.format(correct / total))
    print('Finished Testing')
    return correct / total

main()
