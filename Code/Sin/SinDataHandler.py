import json
def writeDataToJSON(filePath,dataList):
    with open(filePath, 'w') as f:#覆盖式写入
        json.dump(dataList, f, indent=4)

def readDataFromJSON(filePath):
    with open(filePath, 'r') as f:
        # 一次性读取整个文件内容
        content = f.read()
    
    # 使用 json.loads 直接解析整个文件内容
    objects = json.loads('[' + content.replace('}{', '},{') + ']')    
    return objects[0]

