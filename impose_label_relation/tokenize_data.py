from common import load_vocab_vector,get_heirarchy_graph
from get_filtered_data import get_abstract_types,get_type_dict_idx
from sklearn.preprocessing import MultiLabelBinarizer
from config import Config
import dgl
import torch
cfg=Config()

def get_word_level(word_dict,entitynames):
    print("start tokenize entity name to idx ...")
    entitynames_word_index = []
    for ename in entitynames:
        ename = ename.lower()
        temp = [word_dict[x] for x in ename.split()[:cfg.entityname_word_num] if x in word_dict]
        if len(temp) < cfg.entityname_word_num:
            temp += [0 for i in range(cfg.entityname_word_num - len(temp))]
        entitynames_word_index.append(temp)
    print("tokenize entity name to idx successfully ...")
    return entitynames_word_index


def get_entity_level(word_dict,abstracts):
    print("start tokenize abstract to idx ...")
    abstracts_word_index=[]
    for sentence in abstracts:
        temp = [word_dict[x] for x in sentence.split()[:cfg.sentence_len] if x in word_dict]
        if len(temp) < cfg.sentence_len:
            temp += [0 for i in range(cfg.sentence_len - len(temp))]
        abstracts_word_index.append(temp)
    print("tokenize abstract to idx successfully ...")
    return abstracts_word_index
def get_paded_representation_idx_and_split():
    abstracts,cleaned_types,cleaned_entitynames=get_abstract_types()
    word_list, vector_list, word_dict=load_vocab_vector()
    vocab_size=len(word_list)
    abstract_word_index=get_entity_level(word_dict,abstracts)
    entitynames_word_index=get_word_level(word_dict,cleaned_entitynames)
    train_num=int(0.8*len(abstract_word_index))
    val_num=int(0.9*len(abstract_word_index))

    train_abstract_word_index=abstract_word_index[:train_num]
    train_entitynames_word_index=entitynames_word_index[:train_num]
    val_abstract_word_index = abstract_word_index[train_num:val_num]
    val_entitynames_word_index = entitynames_word_index[train_num:val_num]
    test_abstract_word_index = abstract_word_index[val_num:]
    test_entitynames_word_index = entitynames_word_index[val_num:]

    train_types=cleaned_types[:train_num]
    val_types=cleaned_types[train_num:val_num]
    test_types=cleaned_types[val_num:]

    multiLabelBinarizer=MultiLabelBinarizer().fit(train_types)
    train_labels=multiLabelBinarizer.transform(train_types)
    val_labels=multiLabelBinarizer.transform(val_types)
    test_labels=multiLabelBinarizer.transform(test_types)
    return train_abstract_word_index,train_entitynames_word_index,train_labels,val_abstract_word_index,val_entitynames_word_index, val_labels,test_abstract_word_index,test_entitynames_word_index, test_labels,vocab_size,vector_list





# return : g,src,dst,frequency_weight_list
def get_cooccurence():
    abstracts, cleaned_types, cleaned_entitynames = get_abstract_types()
    train_types=cleaned_types[:int(0.8*len(cleaned_types))]
    type_to_idx, idx_to_type=get_type_dict_idx()
    edge_frequency_dict={}#  key:(src,dst)  value:frequency
    for type_list in train_types:
        for i in range(len(type_list)):
            for j in range(i,len(type_list)):
                if(type_to_idx[type_list[i]],type_to_idx[type_list[j]]) in edge_frequency_dict.keys():
                    edge_frequency_dict[(type_to_idx[type_list[i]],type_to_idx[type_list[j]])]+=1
                else:
                    edge_frequency_dict[(type_to_idx[type_list[i]],type_to_idx[type_list[j]])] = 1

                if (type_to_idx[type_list[j]],type_to_idx[type_list[i]]) in edge_frequency_dict.keys():
                    edge_frequency_dict[(type_to_idx[type_list[j]],type_to_idx[type_list[i]])] += 1
                else:
                    edge_frequency_dict[(type_to_idx[type_list[j]],type_to_idx[type_list[i]])] = 1
    max_feq=0#
    up_boundry=1024
    down_boundry=4
    in_boundry=0
    lower_down_boundry=0
    higher_up_boundry=0
    max_log=0
    #上下边界统计   过滤上下边界的值,然后取对数，求最大值
    import math

    for key in edge_frequency_dict.keys():
        if max_feq<edge_frequency_dict[key]:
            max_feq = edge_frequency_dict[key]
        #统计上下边界
        if(edge_frequency_dict[key]>up_boundry):
            higher_up_boundry+=1
            edge_frequency_dict[key]=up_boundry
        elif(edge_frequency_dict[key]<down_boundry):
            edge_frequency_dict[key]=1
            lower_down_boundry+=1
        else:
            in_boundry+=1
        value=edge_frequency_dict[key]
        value=math.log2(value)
        if(value>max_log):
            max_log=value
        edge_frequency_dict[key]=value
    print("大于上边界的边的个数", higher_up_boundry)
    print("在范围之内的个数：", in_boundry, " 比例:", in_boundry / len(edge_frequency_dict.keys()))
    print("小于下边界的边的个数", lower_down_boundry)
    print("dataset==",cfg.datasets_path)#14312
    print("边的最大频率==",max_feq)#
    print("一共多少个边：", len(edge_frequency_dict.keys()))
    print("取对数后的最大值",max_log)
    #归一化 根据共现频率得到的边的权重
    for key in edge_frequency_dict.keys():
        edge_frequency_dict[key]/=max_log

    # constructed by coocurence
    src = []
    dst = []
    frequency_weight_list = []
    for label in edge_frequency_dict.keys():
        src.append(label[0])
        dst.append(label[1])
        if (label[0] != label[1]):
            frequency_weight_list.append(edge_frequency_dict[label])
        else:
            frequency_weight_list.append(1.0)
    print(frequency_weight_list)
    # 构建每个节点的特征,增加自连接边，权重为1.0
    g=dgl.graph((torch.tensor(src),torch.tensor(dst)))
    g.edata["frequency_weight"]=torch.tensor(frequency_weight_list)
    print(g)
    print(len(frequency_weight_list))
    dgl.save_graphs(filename="../cooccurence_graph.pth", g_list=[g])

    return g,src,dst,torch.tensor(frequency_weight_list)


def get_semantic_and_feats():
    type_to_idx, idx_to_type = get_type_dict_idx()
    import wordninja
    node_feat_dict={}
    # 根据bert获取两个label的cls计算他们的cos相似度，  避免出现负数，weight=weight+lamda*(A+1)/2
    # 这里 用cls表示节点特征
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_path)
    bert_model = BertModel.from_pretrained(cfg.bert_path)
    bert_model = bert_model.to(cfg.device)
    src = []
    dst = []
    cosin_weight_list = []
    feats = []
    for label_name in type_to_idx.keys():
        label = " ".join(wordninja.split(label_name)).lower()
        tokenized_dict = tokenizer(label, return_tensors="pt")
        tokenized_dict = tokenized_dict.to(cfg.device)
        outputs = bert_model(**tokenized_dict)
        last_hidden_states = outputs.last_hidden_state.detach().to("cpu")
        # 保存节点特征
        if label_name not in node_feat_dict.keys():
            node_feat_dict[label_name] = last_hidden_states[0][0].numpy().tolist()
        feats.append(last_hidden_states[0][0].numpy().tolist())
    cosin_weight_dict={}# (src,dst):weight,对weight进行排序，保存部分权重
    max_cosin_weight=0
    for s in range(len(idx_to_type.keys())):
        for d in range(s+1,len(idx_to_type.keys())):
            s_feat=torch.tensor(node_feat_dict[idx_to_type[s]])
            d_feat = torch.tensor(node_feat_dict[idx_to_type[d]])
            cos_weight = torch.dot(s_feat, d_feat) / (torch.dot(s_feat, s_feat) * torch.dot(d_feat, d_feat))
            cosin_weight_dict[(s,d)]=cos_weight
            if max_cosin_weight<cos_weight:
                max_cosin_weight=cos_weight
    ranked_list=sorted(cosin_weight_dict.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)
    min_cos_weight=0.0047
    # 构建每个节点的特征,增加自连接边，权重为1.0
    print(len(node_feat_dict.keys()))
    for i in range(652):
        item=ranked_list[i]
        src.append(item[0][0])
        dst.append(item[0][1])
        dst.append(item[0][0])
        src.append(item[0][1])
        if min_cos_weight>item[1]:
            min_cos_weight=item[1]
        cosin_weight_list.append((item[1]-min_cos_weight)/(max_cosin_weight-min_cos_weight))
        cosin_weight_list.append((item[1] - min_cos_weight) / (max_cosin_weight - min_cos_weight))
    # 添加自连接边和权重
    for i in range(232):
        src.append(i)
        dst.append(i)
        cosin_weight_list.append(1.0)
    print(len(feats))
    print("src",len(src))
    print("dst",len(dst))
    print(cosin_weight_list)
    # print(cosin_weight_dict)
    print(min_cos_weight)
    g=dgl.graph((src,dst))
    return feats,g,src,dst,torch.tensor(cosin_weight_list)

def get_heirarchy():
    print()
    # constructed by hierarchy
    g_heirarchy,src,dst = get_heirarchy_graph()
    dgl.save_graphs(filename="../heirarchy_graph.pth", g_list=[g_heirarchy])
    return g_heirarchy,src,dst
def get_hetero():
    _,coocurrence_src,coocurrence_dst,coocurrence_e=get_cooccurence()
    feats,_, cosin_src, cosin_dst, cosin_e = get_semantic_and_feats()
    _,heirarchy_src,heirarchy_dst=get_heirarchy()
    g = dgl.heterograph({
        ("label", "coocurrence", "label"): (coocurrence_src, coocurrence_dst)  # 句子与句子之间的边  自连接
        , ("label", "cosin", "label"): (cosin_src,cosin_dst)
        , ("label", "heirarchy", "label"): (heirarchy_src, heirarchy_dst)
    })
    g.nodes["label"].data["feats"]=torch.tensor(feats)#list
    g.edges["coocurrence"].data["feats"]=coocurrence_e
    g.edges["cosin"].data["feats"] = cosin_e
    dgl.save_graphs(filename="../hetero_graph.pth", g_list=[g])
'''

'''
if __name__ == '__main__':
    print()
    # g_heirarchy, src, dst=get_heirarchy()
    # print(g_heirarchy)
    # get_semantic_and_feats()
    # get_hetero()
    # get_cooccurence()
