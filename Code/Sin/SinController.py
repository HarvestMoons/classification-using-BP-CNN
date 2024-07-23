from SinLayer import Layer  # 导入 Layer 模块中的 Layer 类
import numpy as np
import UserInterface
import random
import SinDataHandler
import math
import os
import matplotlib.pyplot as plt

input_size = 1  # 输入层大小
mid_size =7 # 中间层大小
output_size =1   # 输出层大小

epochs = 100000 #训练集训练次数
batch_size = 1000 # batch大小

mid_min_bias=-0.2
mid_max_bias=0.2
output_min_bias=-0.1
output_max_bias=0

# 定义取样范围和样本数量
x_min, x_max = -np.pi, np.pi
train_size = 1000
test_size=1000

current_dir = os.path.dirname(os.path.abspath(__file__))
input_save_path=os.path.join(current_dir, 'sin_inputLayer.json')
mid_save_path=os.path.join(current_dir, 'sin_midLayer.json')
output_save_path=os.path.join(current_dir, 'sin_outputLayer.json')

testingSet = np.random.uniform(x_min, x_max, test_size)
layers=[] 
def createLayers():
    global inputLayer,midLayer,outputLayer
    if(UserInterface.isLoadFromFile()==False):
        inputLayer=Layer(input_size,mid_size,0,0,None,None)  # 输入层，无bias，左边层不存在
        midLayer=Layer(mid_size,output_size,mid_min_bias,mid_max_bias,inputLayer,None)   # 中间层
        outputLayer=Layer(output_size,0,output_min_bias,output_max_bias,midLayer,None)   # 输出层，所有神经元的权重数量为0
    else: #从文件中加载数据
        inputLayer=Layer(input_size,mid_size,0,0,None,input_save_path)  # 输入层，无bias，左边层不存在
        midLayer=Layer(mid_size,output_size,mid_min_bias,mid_max_bias,inputLayer,mid_save_path)   # 中间层
        outputLayer=Layer(output_size,0,output_min_bias,output_max_bias,midLayer,output_save_path)   # 输出层，所有神经元的权重数量为0
    layers.append(inputLayer)
    layers.append(midLayer)
    layers.append(outputLayer)



# 训练神经网络
def train():
    totalError=0
    count=0
    for j in range(0,epochs):   # 训练epoch次
        trainingSet = np.random.uniform(x_min, x_max, train_size)
        random.shuffle(trainingSet)
        epochError=0
        for i in range(0,train_size):   #每次训练样本有train_size个     
            inputX=trainingSet[i]     #取训练集的第i个样本            
            inputLayer.neurons[0].params["output"]=inputX
            expectOutput=[np.sin(inputX)]
            inputLayer.calcInputForNextLayer()  # 生成中间层输入输出
            midLayer.calcInputForNextLayerWithoutActive()    # 生成输出层输入输出
            inputLayer.sin_backward(expectOutput)   #反向传播            
            totalError+=math.fabs(expectOutput[0]-outputLayer.neurons[0].params["output"])
            count+=1           
            if(batch_size==count):            
                #print("The average error of batch ",int((i+1)/batch_size)," is ",totalError/batch_size)
                      
                epochError+=totalError
                count=0
                totalError=0                
                Layer.updateBias(layers,batch_size) # 每个batch更新bias
                Layer.upDateWeight(layers,batch_size) # 每个batch更新权重
        print("The average error of epoch ",j," is ",epochError/train_size)
        if(epochError/train_size<0.008):
                     writeDataToJSON()  
        #x_values,y_values=test()
        #drawImage(x_values,y_values)
    print("Training process done.")

# 测试神经网络
def test():
    x_values=[]
    y_values=[]
    totalError=0
    for i in range (0,test_size):
        inputX=testingSet[i]     #取测试集的第i个样本
        x_values.append(inputX)
        inputLayer.neurons[0].params["output"]=inputX
        inputLayer.calcInputForNextLayer()  # 生成中间层输入输出
        midLayer.calcInputForNextLayerWithoutActive()    # 生成输出层输入输出
        y_values.append(outputLayer.neurons[0].params["output"])
        expectOutput=[np.sin(inputX)]
        totalError+=math.fabs(expectOutput[0]-outputLayer.neurons[0].params["output"])
    print("The average error on testing set is ",totalError/test_size)
    print("Testing process done.  ",)
    return x_values,y_values  

def writeDataToJSON():
    if(UserInterface.isWriteToFile()):
        SinDataHandler.writeDataToJSON(input_save_path,inputLayer.getListForJSON())
        SinDataHandler.writeDataToJSON(mid_save_path,midLayer.getListForJSON())
        SinDataHandler.writeDataToJSON(output_save_path,outputLayer.getListForJSON())
        print("现有的训练成果已写入JSON文件")

def drawImage(x_values,y_values):
    # 定义取样范围和样本数量
    num_samples = 1000

    # 均匀取样
    sin_x_values = np.linspace(x_min, x_max, num_samples)
    sin_y_values = np.sin(sin_x_values)

    # 绘制图像
    plt.clf()
    plt.scatter(x_values, y_values, color='blue', label='Data Points')
    plt.plot(sin_x_values, sin_y_values)
    plt.xlabel('x')
    plt.ylabel('y = sin(x)')
    plt.title('sin(x)')
    plt.grid(True)
    plt.show()

createLayers()
if(UserInterface.isTrain()):
    train()
x_values,y_values=test()
drawImage(x_values,y_values)
writeDataToJSON()