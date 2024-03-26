import os

file_dir = r'E:\！！！research\Navy_competition\0713\训练USV\待修改txts'  #你的文件路径

def getFlist(path):
    for root, dirs, files in os.walk(file_dir):
        print('root_dir:', root)  #当前路径
        print('sub_dirs:', dirs)   #子文件夹
        print('files:', files)     #文件名称，返回list类型
    return files


filenames = getFlist(file_dir)


# 外层循环读取每一个txt文档，内层循环读取每一行数据,并写入新的txt文档
for filename in filenames:
    # print(filename)
    # txts/是程序所在路径下的文件夹，存放了所有旧的txt文档
    f = open('待修改txts/' + filename)
    for line in f:
        new_line = str(int(line[0]) - 1) + line[1:]
        # 需要建立一个新文件夹来存放修改后的txt文档
        with open(r"E:\！！！research\Navy_competition\0713\训练USV\修改后txt/" + filename, 'a') as f:
            f.write(new_line)





