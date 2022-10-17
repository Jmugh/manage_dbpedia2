import json
import math
import dgl
import torch
import torch.nn as nn
from scipy import sparse

from common import get_heirarchy_graph, get_label_level

filename = "./datasets_abstract2_15w.txt"
filename = "E:/python/code/mytasks/data/finall_file.txt"


def get_entity(filename=filename):
    with open(filename, "r", encoding="utf8") as reader:
        for line in reader:
            line = line.strip()
            json_line = json.loads(line)
            types = json_line["TYPES"]
            if (len(types) > 5 and len(json_line["CATEGORY"]) < 4 and len(json_line["ABSTRACT"].split()) > 50):
                # print("entityname:"+json_line["ENTITY"]+"\n")
                # print("infobox:"+json_line["INFOBOX"]+"\n")
                # print("category:"+json_line["CATEGORY"]+"\n")
                # print("abstract:"+json_line["ABSTRACT"]+"\n")
                # print("types:"+json_line["TYPES"]+"\n")

                print(line)


def test():
    import torch
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 1 2 3
    # 4  5   6
    b = torch.tensor([[2, 2, 2], [3, 3, 3], [4, 4, 4]])

    a = a.unsqueeze(dim=1)
    print(a)
    print(torch.mul(a, b))

def test_graph():
    _, src, dst = get_heirarchy_graph()
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)))
    import scipy.sparse as sp
    A=g.adjacency_matrix()
    print(A)
    print(sp.coo_matrix(A))
    print(sp.csr_matrix(A))

def test_cos():
    a=torch.tensor([[1.0,2.0,3.0],
                    [3.0,4.0,5.0]])
    b=torch.tensor([1.1,2.2,3.3])
    cos=torch.nn.CosineSimilarity(dim=-1)
    res= cos(a,b)
    res=torch.abs(res)
    print(res)
def test_cos1():
    a=torch.rand(64,1,232)
    b=torch.rand(232,232)
    cos=torch.nn.CosineSimilarity(dim=-1)
    res=cos(a,b)
    print(res.shape)
def construct_y(y,levels):
    zero_target=sum(y)/232
    adjust_label = [zero_target,0.7,0.75,0.8,0.85,0.9,0.95,1]#7  0.6-1=0.4   /8=0.05
    target_y=[0 for i in range(232)]
    for i in range(232):
        if y[i]==0:
            target_y[i]=zero_target
        else:
            target_y[i]=adjust_label[levels[i]]
    return target_y
def test_batchnorm():
    import torch.nn as nn
    em=nn.Embedding(5,2)
    embedding= em(torch.tensor([0,1,2,3,4]))
    print(embedding)
    print(embedding.shape)
    batch_norm = nn.BatchNorm1d(num_features=2)

    print(batch_norm(embedding))
def test_layernorm():
    import torch.nn as nn
    em=nn.Embedding(5,5)
    embedding= em(torch.tensor([0,1,2,3,4]))
    print(embedding)
    print(embedding.shape)
    layer_norm = nn.LayerNorm(5)
    print(layer_norm(embedding))

def mycos_pi_2():
    entity = torch.rand(128,1,232)
    label_embedding=torch.rand(232,232)
    cos = nn.CosineSimilarity(dim=-1)
    entity_sum = torch.sqrt(torch.sum(torch.square(entity), dim=-1)).unsqueeze(-1)  # []
    label_sum=torch.sqrt(torch.sum(torch.square(label_embedding), dim=-1)).unsqueeze(-1)
    entity=torch.div(entity,entity_sum)*math.pi
    label_embedding=torch.div(label_embedding,label_sum)*math.pi
    cos_output=cos(entity,label_embedding)
    print(cos_output)
if __name__ == '__main__':
    # test_cos()
    # test_cos1()
    # y = [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0] + [0 for i in range(217)]
    # labels, levels = get_label_level()
    # construct_y(y)
    # import time
    # print(time.time())
    # a={}
    # a[0]=1
    # test_layernorm()
    mycos_pi_2()