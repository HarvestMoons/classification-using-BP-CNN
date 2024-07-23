from Node import Node
import MyMathFunc
import DataHandler
import numpy as np
from DataUnit import Data
w_learningRate = b_learningRate=0.0005
a=0 #a为动量项系数

class Layer:
    def __init__(self,nodeNum,weightNum,min_bias,max_bias,leftLayer,oldDataPath)->None:
        self.nodeNum = nodeNum  # 每层有的神经元个数
        self.leftLayer = leftLayer  # 该层的左层，可以为空
        self.rightLayer=None# 该层的右层，可以为空
        self.neurons = []  # 每层的神经元列表
        if(oldDataPath==None):
            for i in range(0, nodeNum):            
                n = Node(weightNum,min_bias,max_bias) 
                self.neurons.append(n)
        else:
            nodeDatas=DataHandler.readDataFromJSON(oldDataPath)                        
            for i in range(0, nodeNum):                                                           
                n = Node(weightNum,min_bias,max_bias)
                n.params["weight"] =np.array(nodeDatas[i]["weight"])
                n.params["bias"] =nodeDatas[i]["bias"]
                self.neurons.append(n)        
        # 同时设置此层的左层的右层为此层
        if self.leftLayer is not None:
            self.leftLayer.rightLayer = self


    # 初始化输入层的输入
    def initInputForInputLayer(self,data:Data)->None:
        if(self.leftLayer is not None):
            print("错误调用initInputForInputLayer：非输入层不可调用此方法")
            return
        for i in range(0,self.nodeNum):
            self.neurons[i].params["output"]=data.pixels[i]/255
                
    # 为下一层的每一个神经元计算输入，传入其calcOutput方法生成其输出
    def calcInputForNextLayer(self)->None:
        if(self.rightLayer is None):
            print("错误调用calcInputForNextLayer：输出层不可调用此方法")
            return None
        NextLayer=self.rightLayer
        for j in range(0,NextLayer.nodeNum):
            nodeInNextLayer=NextLayer.neurons[j]
            inputForNextLayer=nodeInNextLayer.params['bias']    #  inputForNextLayer=bias+w1o1+w2o2+...+wnon+...
            for i in range(0, self.nodeNum):
                n=self.neurons[i]
                inputForNextLayer+=n.params['output']*n.params['weight'][j]
            nodeInNextLayer.calcOutput(inputForNextLayer)
            

    # 为下一层的每一个神经元计算输入，传入其calcOutput方法生成其输出
    def calcInputForNextLayerWithoutActive(self)->None:
        if(self.rightLayer is None):
            print("错误调用calcInputForNextLayer：输出层不可调用此方法")
            return None
        NextLayer=self.rightLayer
        for j in range(0,NextLayer.nodeNum):
            nodeInNextLayer=NextLayer.neurons[j]
            inputForNextLayer=nodeInNextLayer.params['bias']    #  inputForNextLayer=bias+w1o1+w2o2+...+wnon+...
            for i in range(0, self.nodeNum):
                n=self.neurons[i]
                inputForNextLayer+=n.params['output']*n.params['weight'][j]
            nodeInNextLayer.params["output"]=MyMathFunc.tanh(inputForNextLayer)
            

    # 错误（Error）计算，只有输出层的这一方法有意义，expectOutput为期望输出，是一个列表
    def calcError(self,expectOutput)->float:
        if(self.rightLayer is not None):
            print("错误调用calcError：不是输出层")
            return None
        output=[]
        for i in range(0, self.nodeNum):
            output.append(self.neurons[i].params['output'])
        error=MyMathFunc.get_err(expectOutput-output)
        return error

    #一个batch后，更新中间层、输入层的权重，并清零各层的delta
    def upDateWeight(layers,batch_size):     
        for layer in layers:
            if(layer.rightLayer is not None):
                for i in range(0,layer.nodeNum):
                    node= layer.neurons[i]   
                    for j in range(0,layer.rightLayer.nodeNum):   
                        node.params['weight'][j]+=(node.params['delta_weight'][j]/batch_size)+a*node.params['last_delta_weight'][j]
                        node.params['last_delta_weight'][j]=node.params['delta_weight'][j]
                        node.params['delta_weight'][j]=0

    # 更新bias，把delta_bias置为0    
    def updateBias(layers,batch_size)->None:
        for layer in layers:
            if(layer.leftLayer is not None):
                for i in range (0,layer.nodeNum):
                    node=layer.neurons[i]
                    node.params['bias']+=(node.params['delta_bias']/batch_size)+a*node.params['last_delta_bias']
                    node.params['last_delta_bias']=node.params['delta_bias']
                    node.params['delta_bias']=0

    def softmax(self)->None:
        if(self.rightLayer is not None):
            print("错误调用getRealOutput：不是输出层")
            return None
        outputSum=0
        for i in range(0,self.nodeNum):
            outputSum+=np.exp(self.neurons[i].params['output'])
        for i in range(0,self.nodeNum):
            self.neurons[i].params['old_output']=self.neurons[i].params['output']
            self.neurons[i].params['output']=np.exp(self.neurons[i].params['output'])/outputSum

    #从输出中的最大值获取网络判断出的数据类型
    def getCategoryFromOutput(self)->int:
        if(self.rightLayer is not None):
            print("错误调用getRealOutput：不是输出层")
            return None
        category=0
        currentMaxOutput= self.neurons[0].params['output']
        for i in range(1,self.nodeNum):
            if(self.neurons[i].params['output']>currentMaxOutput):
                currentMaxOutput=self.neurons[i].params['output']
                category=i
        return category+1
  

    #反向传播方法，计算输入、中间层神经元的delta,中间、输出层的delta_bias
    def backward(layers,expectOutput):
        inputLayer=layers[0]
        midLayer=inputLayer.rightLayer
        outputLayer=midLayer.rightLayer

        for i in range(0,outputLayer.nodeNum):# 遍历输出神经元
            outputNode=outputLayer.neurons[i]      
            outputNode.params['delta_bias']+=b_learningRate*outputNode.old_tanh_derivative()*(expectOutput[i]-outputNode.params['output'])         
        
        for j in range(0, midLayer.nodeNum):# 遍历中间层神经元
            midNode=midLayer.neurons[j]
            outParam=0
            for i in range(0,outputLayer.nodeNum):# 遍历输出神经元      
                outputNode=outputLayer.neurons[i]
                outParam+=midNode.params['weight'][i]*outputNode.old_tanh_derivative()*(expectOutput[i]-outputNode.params['output'])  
                midNode.params['delta_weight'][i]+=w_learningRate*outputNode.old_tanh_derivative()*(expectOutput[i]-outputNode.params['output']) *midNode.params['output']
            midNode.params['temp_delta_bias']=b_learningRate*outParam*midNode.tanh_derivative() 
            midNode.params['delta_bias']+=midNode.params['temp_delta_bias']

        for k in range(0, inputLayer.nodeNum):# 遍历输入层神经元
            inputNode=inputLayer.neurons[k]
            for j in range(0,midLayer.nodeNum):# 遍历中间神经元
                midNode=midLayer.neurons[j]
                inputNode.params['delta_weight'][j]+=midNode.params['temp_delta_bias']*inputNode.params['output'] 

    def getListForJSON(self):
        nodesData=[]
        for node in self.neurons:
            nodesData.append(node.getListForJSON())
        return nodesData        