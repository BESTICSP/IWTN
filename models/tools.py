import linecache
import torch

def getboxinfo(image_path,anno_path):
    # key = image_path.split('\\')[-1]  # 1_00001.png

    file = open(anno_path, 'r', encoding='utf-8')
    file_content=[]
    for line in file.readlines():
        file_content.append(line)
    file.close()

    image_box = torch.zeros(4, 5)
    image_box[:, 0] = -1

    i = 0
    while i<=3 :
        #print("i:",i)
        path = image_path[i]
        key=path.split('/')[-1]
        #print("image_path[i]:", image_path[i])
        #print("key:", key)
        index = 0
        for line in file_content:
            index = index+1
            if line.find(key) != -1:
                #print("找到key",line)
                # print(index,line)
                param = linecache.getline(anno_path,index+4)
                #print('param:',param)
                # print(param)
                param = param.strip()
                params = param.split(' ')
                #print(params)
                image_box[i, 0] = int(params[0])
                image_box[i, 1] = float(params[1])
                image_box[i, 2] = float(params[2])
                image_box[i, 3] = float(params[3])
                image_box[i, 4] = float(params[4])
               # i = i + 1
        i = i+1 
    return image_box

def getBoxInfo4(image_path, anno_path):
    # 首先将标注信息文档的内容按行存放到一个 list 中去
    file = open(anno_path, 'r', encoding='utf-8')
    file_content = []
    for line in file.readlines():
        file_content.append(line)
    file.close()

    # 创建一个tensor向量用来存放anno信息
    image_box = torch.zeros(30,5)
    image_box[:, 0] = -1
    # box_info = []
    info_index = 0
    for p,path in enumerate(image_path):
        image_name = path.split('/')[-1]
        # print('这是第',p,'张图片',image_name)
        # 进入循环，为每一张图片寻找标注信息
        find_flag = 1
        for index,line in enumerate(file_content):
            if line.find(image_name) != -1 and find_flag == 1:
                # 先找到 这张图片所在的位置
                #print(index,line)
                find_flag = 0
                # 然后得到这张图片中一共有几个anno信息
                anno_nums = int(file_content[index+3])
                # print('anno nums',anno_nums)
                for i in range(min(anno_nums,2)):
                    # 然后开始获取每个anno标注的信息
                    param = file_content[index+4+i]
                    param = param.strip()
                    # print(p,'--',i,'--',param)
                    params = param.split(' ')
                    #print('param 0:',params[0])
                    #if info_index > 28:
                        #break;
                    # 应该是坐标没有缩放正确，导致的错误 illegal memory access
                    # 应该在坐标值后面乘上对应的比例 256/1208=0.13 256/1920=0.21
                    # 0.13是x 0.21是y
                    image_box[info_index, 0] = int(params[0])
                    image_box[info_index, 1] = float(params[1])*0.13
                    image_box[info_index, 2] = float(params[2])*0.21
                    image_box[info_index, 3] = float(params[3])*0.13
                    image_box[info_index, 4] = float(params[4])*0.21
                    info_index = info_index + 1
                    #print('info index',info_index)
    return image_box
def getBoxInfo5(image_path, anno_path):
    # 首先将标注信息文档的内容按行存放到一个 list 中去
    file = open(anno_path, 'r', encoding='utf-8')
    file_content = []
    for line in file.readlines():
        file_content.append(line)
    file.close()

    # 创建一个tensor向量用来存放anno信息
    image_box = torch.zeros(20,5)
    image_box[:, 0] = -1
    # box_info = []
    for p,path in enumerate(image_path):
        image_name = path.split('/')[-1]
        #print('这是第',p,'张图片',image_name)
        # 进入循环，为每一张图片寻找标注信息
        find_flag = 1
        info_index = 0
        for index,line in enumerate(file_content):
            if line.find(image_name) != -1 and find_flag == 1:
                # 先找到 这张图片所在的位置
                #print(index,line)
                find_flag = 0
                # 然后得到这张图片中一共有几个anno信息
                anno_nums = int(file_content[index+3])
                # print('anno nums',anno_nums)
                for i in range(min(anno_nums,2)):
                    # 然后开始获取每个anno标注的信息
                    param = file_content[index+4+i]
                    param = param.strip()
                    #print(p,'--',i,'--',param)
                    params = param.split(' ')
                    #print('param 0:',params[0])
                    #if info_index > 28:
                        #break;
                    # 应该是坐标没有缩放正确，导致的错误 illegal memory access
                    # 应该在坐标值后面乘上对应的比例 256/1208=0.13 256/1920=0.21
                    # 0.13是x 0.21是y
                    image_box[info_index+p*5, 0] = int(params[0])
                    image_box[info_index+p*5, 1] = float(params[1])*0.13
                    image_box[info_index+p*5, 2] = float(params[2])*0.21
                    image_box[info_index+p*5, 3] = float(params[3])*0.13
                    image_box[info_index+p*5, 4] = float(params[4])*0.21
                    info_index = info_index + 1
                    #print('info index',info_index)
    return image_box
