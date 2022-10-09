import json
from config import Config
from sklearn.preprocessing import MultiLabelBinarizer
cfg=Config()
import random
from get_data import get_abstract_types
def split_train_test():
    abstracts,entitynames, types=get_abstract_types()

    #选取第一个句子
    # temp_abstracts=[]
    # for i in range(len(abstracts)):
    #     temp_abstracts.append(abstracts[i].split(".")[0])
    # abstracts=temp_abstracts

    train_num=int(0.8*len(abstracts))
    train_x=abstracts[:train_num]
    train_entitynames=entitynames[:train_num]
    train_y=types[:train_num]
    test_x=abstracts[train_num:]
    test_entitynames=entitynames[train_num:]
    test_y=types[train_num:]

    multiLabelBinarizer=MultiLabelBinarizer().fit(train_y)
    train_y=multiLabelBinarizer.transform(train_y)
    # print(multiLabelBinarizer.classes_)
    test_y=multiLabelBinarizer.transform(test_y)
    return train_x,train_entitynames,train_y,test_x,test_entitynames,test_y

if __name__=="__main__":
    print("hi")
    # split_train_test()