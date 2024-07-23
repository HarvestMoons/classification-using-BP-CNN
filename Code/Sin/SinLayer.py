from Node import Node
import MyMathFunc
import SinDataHandler
import numpy as np
w_learningRate = b_learningRate=0.05
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
            nodeDatas=SinDataHandler.readDataFromJSON(oldDataPath)                        
            for i in range(0, nodeNum):                                                           
                n = Node(weightNum,min_bias,max_bias)
                n.params["weight"] =np.array(nodeDatas[i]["weight"])
                n.params["bias"] =nodeDatas[i]["bias"]
                self.neurons.append(n)        
        # 同时设置此层的左层的右层为此层
        if self.leftLayer is not None:
            self.leftLayer.rightLayer = self
                
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
                        node.params['weight'][j]+=(node.params['delta_weight'][j]/batch_size)+a*node.params['last_delta_weight'][j]/batch_size
                        node.params['last_delta_weight'][j]=node.params['delta_weight'][j]
                        node.params['delta_weight'][j]=0

    # 更新bias，把delta_bias置为0    
    def updateBias(layers,batch_size)->None:
        for layer in layers:
            if(layer.leftLayer is not None):
                for i in range (0,layer.nodeNum):
                    node=layer.neurons[i]
                    node.params['bias']+=(node.params['delta_bias']/batch_size)+a*node.params['last_delta_bias']/batch_size
                    node.params['last_delta_bias']=node.params['delta_bias']
                    node.params['delta_bias']=0

    def sin_backward(self,expectOutput):
        if(self.leftLayer is not None or self.rightLayer is None):
            print("错误调用backward：不是输入层")
            return           
        midLayer=self.rightLayer
        outputLayer=midLayer.rightLayer
        inputNode=self.neurons[0]
        outputNode=outputLayer.neurons[0]         
        outputNode.params['delta_bias']+=b_learningRate*outputNode.tanh_derivative()*(expectOutput[0]-outputNode.params['output']) 
        
        for j in range(0, midLayer.nodeNum):# 遍历中间层神经元
            midNode=midLayer.neurons[j]               
            outParam=midNode.params['weight'][0]*outputNode.tanh_derivative()*(expectOutput[0]-outputNode.params['output'])  
            midNode.params['delta_weight']+=w_learningRate*outputNode.tanh_derivative()*(expectOutput[0]-outputNode.params['output']) *midNode.params['output']
            midNode.params['temp_delta_bias']=b_learningRate*outParam*midNode.tanh_derivative() 
            midNode.params['delta_bias']+=midNode.params['temp_delta_bias']
            inputNode.params['delta_weight'][j]+=midNode.params['temp_delta_bias']*inputNode.params['output'] 

    def getListForJSON(self):
        nodesData=[]
        for node in self.neurons:
            nodesData.append(node.getListForJSON())
        return nodesData        