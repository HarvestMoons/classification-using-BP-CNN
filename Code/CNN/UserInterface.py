    


def isLoadFromFile():
    while True:
        user_input = input("是否从文件中加载神经网络？请输入Y/y或N/n：").upper()  # 将用户输入转换为大写
        if user_input == 'Y' or user_input == 'N':
            return user_input== 'Y'
        else:
            print("输入错误，请重新输入。")

def isWriteToFile():
    while True:
        user_input = input("是否把现有的训练成果写入文件？请输入Y/y或N/n：").upper()  # 将用户输入转换为大写
        if user_input == 'Y' or user_input == 'N':
            return user_input== 'Y'
        else:
            print("输入错误，请重新输入。")