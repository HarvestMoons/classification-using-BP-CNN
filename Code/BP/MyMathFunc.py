import numpy as np
import math
# 定义激活函数及其导数
def sigmoid(x:float)->float:
    return 1 / (1 + np.exp(-x))

def tanh(x):
     return math.tanh(x)

def get_err(e):
    return 0.5*np.dot(e,e)

def initWeight(weightNum:int):
      normalizedWeight=np.random.normal(loc=0, scale=0.1, size=weightNum)      
      return np.clip(normalizedWeight,a_min=-1/35,a_max=1/35)

def errorCalculator(output,expectOutput)->float:
    errorTimesTwo=0
    for i in range (0,len(output)):
         errorTimesTwo+=(expectOutput[i]-output[i])**2
    error=errorTimesTwo/2
    return error