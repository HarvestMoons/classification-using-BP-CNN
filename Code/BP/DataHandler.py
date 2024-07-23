from PIL import Image
from DataUnit import Data
import os
import json
data_category_size=12
def process_bmp_files_by_folder(root_folder):
        data_by_folder = {}  # 使用字典存储每个子文件夹的数据
        dataCategory=0  #数据类型为1,2,3...
        for subdir_name in os.listdir(root_folder): # 遍历根文件夹下的所有子文件夹          
            if(dataCategory==data_category_size):
                break
            dataCategory+=1
            subdir_path = os.path.join(root_folder, subdir_name)
            if os.path.isdir(subdir_path):# 检查子文件夹是否存在
                data_by_folder[dataCategory] = []  # 初始化每个子文件夹对应的数组
                for file_name in os.listdir(subdir_path):# 遍历子文件夹中的.bmp文件
                    file_path = os.path.join(subdir_path, file_name)
                    if file_name.lower().endswith('.bmp') and os.path.isfile(file_path):# 检查文件是否是.bmp文件
                        img_path = os.path.join(subdir_path, file_name)
                        img = Image.open(img_path) # 打开BMP图片
                        pixels = list(img.getdata())
                        data_by_folder[dataCategory].append(Data(pixels, dataCategory))
        print("数据处理完成")
        return data_by_folder

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

