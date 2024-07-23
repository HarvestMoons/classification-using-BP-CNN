from Layer import Layer  # 导入 Layer 模块中的 Layer 类
import numpy as np
import DataHandler
import UserInterface
import random
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'test_data')
input_save_path=os.path.join(current_dir, 'inputLayer3.json')
mid_save_path=os.path.join(current_dir, 'midLayer3.json')
output_save_path=os.path.join(current_dir, 'outputLayer3.json')

input_size = 784  # 输入层大小（28x28）
mid_size =31  # 中间层大小
output_size =data_category_size=DataHandler.data_category_size   # 输出层大小

epochs =2  #训练集训练次数
batch_size = 24 # batch大小

mid_min_bias=-0.2
mid_max_bias=0.2
output_min_bias=-0.1
output_max_bias=0

data_by_folder=DataHandler.process_bmp_files_by_folder(data_path)
trainingSet=[]
for i in range(0,240):
    for j in range(0,data_category_size):
        trainingSet.append(data_by_folder[j+1][i])    
train_size=len(trainingSet) # 训练集大小 
print("训练集已加载完成。train_size:",train_size," total batch:",int(train_size/batch_size)*epochs)

testingSet=[]
for i in range(0,240):
    for j in range (0,data_category_size):
        testingSet.append(data_by_folder[j+1][i])
#testingSet=trainingSet
random.shuffle(testingSet)
test_size=len(testingSet)    # 测试单元数量
print("测试集已加载完成。test_size:",test_size)

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
        random.shuffle(trainingSet)
        for i in range(0,train_size):   #每次训练样本有train_size个     
            dataUnit=trainingSet[i]     #取训练集的第i个样本            
            inputLayer.initInputForInputLayer(dataUnit) # 导入初始化数据n
            expectOutput=np.zeros(output_size)
            expectOutput[int(dataUnit.category)-1]=1 # 生成期望输出的序列
            inputLayer.calcInputForNextLayer()  # 生成中间层输入输出
            midLayer.calcInputForNextLayer()    # 生成输出层输入输出            
            Layer.backward(layers,expectOutput)   #反向传播
            outputLayer.softmax() # 输出层输出调整            
            totalError+=outputLayer.calcError(expectOutput) #error计算
            count+=1            
            if(batch_size==count):            
                print("The average error of batch ",int((i+1)/batch_size)," is ",totalError/batch_size)               
                count=0
                totalError=0                
                Layer.updateBias(layers,batch_size) # 每个batch更新bias
                Layer.upDateWeight(layers,batch_size) # 每个batch更新权重
        print("epoch ",j," done." )    
    print("Training process done.")

# 测试神经网络
def test():
    correctNum=0
    for i in range (0,test_size):
        dataUnit=testingSet[i]     #取测试集的第i个样本
        inputLayer.initInputForInputLayer(dataUnit) # 导入初始化数据
        inputLayer.calcInputForNextLayer()  # 生成中间层输入输出
        midLayer.calcInputForNextLayer()    # 生成输出层输入输出
        outputLayer.softmax() # 输出层输出调整
        '''for j in range(0,data_category_size):
            print(j," out： ",outputLayer.neurons[j].params["output"])'''
        if(outputLayer.getCategoryFromOutput()==int(dataUnit.category)):    # 判断正确，计数加一
            correctNum+=1 
            print("correct!category: ",dataUnit.category,"Num: ",i)
        else:
            print("wrong!category: ",dataUnit.category," I tkink it's category ",outputLayer.getCategoryFromOutput(),"Num: ",i)
    print("Testing process done. CorrectRate: {:.5%}".format(correctNum/test_size)) 

def writeDataToJSON():
    if(UserInterface.isWriteToFile()):
        DataHandler.writeDataToJSON(input_save_path,inputLayer.getListForJSON())
        DataHandler.writeDataToJSON(mid_save_path,midLayer.getListForJSON())
        DataHandler.writeDataToJSON(output_save_path,outputLayer.getListForJSON())
        print("现有的训练成果已写入JSON文件")



createLayers()
if(UserInterface.isTrain()):
    train()
test()
writeDataToJSON()