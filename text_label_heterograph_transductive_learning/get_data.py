import json
from config import Config
from common import clean_str
cfg=Config()

datasets_length=50000
text_length=cfg.sentence_len
#获取abstract 和  types
def get_abstract_types(filename=cfg.datasets_path):
    min_frequency=20
    abstracts=[]
    entitynames=[]
    types=[]
    count=0
    with open(filename,"r",encoding="utf8") as reader:
        for index,line in enumerate(reader):
            # if index>100:
            #     break
            line=line.strip()
            json_line=json.loads(line)
            abstract=json_line["ABSTRACT"]

            type=json_line["TYPES"]
            abstract=clean_str(abstract)
            abstract=clean_str(abstract)
            if len(abstract.split())<text_length:
                count+=1
            abstracts.append(abstract)
            entitynames.append( json_line["ENTITY"])
            types.append(type)
        abstracts=abstracts[:datasets_length]
        types=types[:datasets_length]

        #处理types 1.删除低频label 2.判断是否所有label同时出现在训练集测试集
        label_dict=get_label_dict(types)
        cleaned_types=[]
        for type_list in types:
            temp_list=[]
            for type in type_list:
                if type in label_dict.keys() and label_dict[type]>min_frequency:#
                    temp_list.append(type)
            cleaned_types.append(temp_list)
        all_label_frequency=0
        selected_label_frequency=0
        for label in label_dict.keys():
            if label_dict[label]>min_frequency:
                selected_label_frequency+=label_dict[label]
            all_label_frequency+=label_dict[label]
        print("句子单词个数小于",text_length,"的比例：",count/len(abstracts))
        print("标签频率大于：",min_frequency)
        print("处理后多少个标签：",selected_label_frequency)
        print("原数据集多少个标签：",all_label_frequency)
        print("处理后共多少种标签：",len(get_label_dict(cleaned_types).keys()))
        print("原数据集多少种标签：",len(get_label_dict(types).keys()))
        print("训练集多少种标签：",len(get_label_dict(cleaned_types[:int(0.8*len(cleaned_types))]).keys()))
        print("测试集多少种标签：",len(get_label_dict(cleaned_types[int(0.8*len(cleaned_types)):]).keys()))
        print("共多少数据：",len(abstracts))
    return abstracts,entitynames,cleaned_types



def get_label_dict(types):#获取当前需要的数据的types
    label_dict = {}
    for label_list in types:
        for label in label_list:
            if label in label_dict.keys():
                label_dict[label] += 1
            else:
                label_dict[label] = 1
    return label_dict
if __name__=="__main__":
    abstracts,types=get_abstract_types()
'''
    20
    处理后多少个标签： 161179
    原数据集多少个标签： 162282
    处理后共多少种标签： 232
    原数据集多少种标签： 399
    训练集多少种标签： 232
    测试集多少种标签： 232
'''