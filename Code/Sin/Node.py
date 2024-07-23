import random
import MyMathFunc
class Node:
     def __init__(self,weightNum,minBias,maxBias) -> None:         
         self.params = {'weight':MyMathFunc.initWeight(weightNum),  # 权重（共有weightNum个）
                       'bias': random.uniform(minBias, maxBias),  # 偏移量
                       'input': 0.0,  # 输入
                       'output': 0.0,  # 输出                       
                       'delta_bias':0.0,     #bias调整方向 
                       'temp_delta_bias':0.0,     
                       'delta_weight': [0.0]*weightNum,   #weight调整方向（共有weightNum个）
                       'last_delta_weight':[0.0]*weightNum,   #权重动量式
                       'last_delta_bias':0.0, #动量式
                       'old_output':0.0
                       }
     def calcOutput(self,input:float)->None:
         self.params['input']=input         
         self.params['output']=MyMathFunc.tanh(self.params['input'])
         
     def tanh_derivative(self)->int:
          output = self.params['output']
          return 1-output**2
     
     def old_tanh_derivative(self)->int:
          output = self.params['old_output']
          return 1-output**2
     
     def getListForJSON(self):
          nodeData={"weight":self.params['weight'].tolist(),
                    "bias":self.params['bias']
                    }
          return nodeData