import math

import torch
import torch.nn as nn
from transformers import BertModel
# import dgl
# from dgl.nn.pytorch import GraphConv
# from dgl.nn.pytorch import GATConv
import torch.nn.functional as F
from config import Config
cfg=Config()

class MLP(nn.Module):
    def __init__(self,n_classes,in_dim=768,hidden_dim1=cfg.hidden_dim1,hidden_dim2=cfg.hidden_dim2,use_label="cooccurence_cosin_label_graph.pth"):
        super(MLP, self).__init__()
        print("--------------------MLP-------------------------")
        # self.lamda = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.mu = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.lamda.data = torch.tensor([0.5])
        # self.mu.data = torch.tensor([0.5])
        self.in_dim=in_dim
        self.hidden_dim1=hidden_dim1
        self.hidden_dim2=hidden_dim2
        self.n_classes=n_classes
        self.use_label=use_label
        self.bert=BertModel.from_pretrained(cfg.bert_path)
        for p in self.bert.parameters():
            p.requires_grad_(False)
        self.unfreeze_layers = ['layer.11', 'bert.pooler']
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in self.unfreeze_layers:
                if ele in name:
                    param.requires_grad = True

        # self.gcn_cooccurence = GraphConv(in_feats=768, out_feats=n_classes)
        # self.gcn_heirarchy = GraphConv(in_feats=768, out_feats=n_classes)
        # self.gat_cooccurence = GATConv(in_feats=768, out_feats=int(n_classes/(cfg.num_heads)),num_heads=cfg.num_heads)
        # self.gat_heirarchy = GATConv(in_feats=768, out_feats=int(n_classes/(cfg.num_heads)),num_heads=cfg.num_heads)
        # g_list = dgl.load_graphs(filename="./saved_graphs/cooccurence_graph.pth")
        # g_cooccurence=g_list[0][0]
        # self.g_cooccurence = g_cooccurence.to(cfg.device)
        # g_list = dgl.load_graphs(filename="./saved_graphs/heirarchy_graph.pth")
        # g_heirarchy = g_list[0][0]
        # self.g_heirarchy= g_heirarchy.to(cfg.device)




        # self.dropout=nn.Dropout(0.2)
        self.linear1=nn.Linear(in_features=self.in_dim,out_features=self.hidden_dim1)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(in_features=hidden_dim1,out_features=hidden_dim2)
        self.dropout = nn.Dropout(0.3)
        self.linear3 = nn.Linear(in_features=hidden_dim2, out_features=n_classes)
        self.sigmoid=nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output = output.last_hidden_state#[batch_size,sequence_len,hidden_dim]
        # batch_cls=output[:,0]
        batch_tokens=torch.mean(output,dim=1)
        output=self.linear1(batch_tokens)
        output = self.dropout(output)
        output=self.linear2(output)
        output=self.relu(output)
        output=self.linear3(output)
        '''
        #共现关系的gcn
        feat_cooccurence = self.gcn_cooccurence(self.g_cooccurence, self.g_cooccurence.ndata["feat"],edge_weight=self.g_cooccurence.edata["frequency_weight"])
        self.g_cooccurence.ndata["out"] = feat_cooccurence
        g_cooccurence_out = dgl.max_nodes(self.g_cooccurence, feat="out")
        
        #共现关系的gat
        feat_cooccurence = self.gat_cooccurence(self.g_cooccurence, self.g_cooccurence.ndata["feat"])
        feat_cooccurence = feat_cooccurence.reshape(-1, self.n_classes)
        self.g_cooccurence.ndata["out"] = feat_cooccurence
        g_cooccurence_out = dgl.max_nodes(self.g_cooccurence, feat="out")
        '''
        '''
        #层级关系的gcn
        feat_heirarchy = self.gcn_heirarchy(self.g_heirarchy, self.g_heirarchy.ndata["feat"])
        self.g_heirarchy.ndata["out"] = feat_heirarchy
        g_heirarchy_out = dgl.max_nodes(self.g_heirarchy, feat="out")
        '''
        # 层级关系的gat
        # feat_heirarchy = self.gat_heirarchy(self.g_heirarchy, self.g_heirarchy.ndata["feat"])
        # feat_heirarchy = feat_heirarchy.reshape(-1, self.n_classes)
        # self.g_heirarchy.ndata["out"] = feat_heirarchy
        # g_heirarchy_out = dgl.max_nodes(self.g_heirarchy, feat="out")
        #
        # output = torch.mul(output, g_heirarchy_out)

        output=self.sigmoid(output)
        return output


#
# class Hetero_label(nn.Module):
#     def __init__(self,n_classes,in_dim=768,hidden_dim1=cfg.hidden_dim1,hidden_dim2=cfg.hidden_dim2):
#         super(Hetero_label, self).__init__()
#         self.in_dim=in_dim
#         self.hidden_dim1=hidden_dim1
#         self.hidden_dim2=hidden_dim2
#         self.n_classes=n_classes
#         self.use_label=use_label
#         self.bert=BertModel.from_pretrained(cfg.bert_path)
#         for p in self.bert.parameters():
#             p.requires_grad_(False)
#         g_list = dgl.load_graphs(filename="../hetero_graph.pth")
#         g = g_list[0][0]
#         self.g = g.to(cfg.device)
#         self.etypes = ["coocurrence", "cosin", "heirarchy"]
#         # self.etypes = ["heirarchy"]
#         self.heterograph = dgl.nn.pytorch.HeteroGraphConv(
#             {etype: GATConv(in_feats=in_dim, out_feats=int(n_classes/4), num_heads=4)
#              for etype in self.etypes}, aggregate="sum")
#         self.linear1=nn.Linear(in_features=self.in_dim,out_features=self.hidden_dim1)
#         self.relu=nn.ReLU()
#         self.linear2=nn.Linear(in_features=hidden_dim1,out_features=hidden_dim2)
#         self.dropout = nn.Dropout(0.3)
#         self.linear3 = nn.Linear(in_features=hidden_dim2, out_features=n_classes)
#         self.sigmoid=nn.Sigmoid()
#     def forward(self, input_ids, token_type_ids, attention_mask):
#         # print(len(input_ids))
#         output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#         output = output.last_hidden_state#[batch_size,sequence_len,hidden_dim]
#         batch_cls=output[:,0]
#         # batch_tokens=torch.mean(output,dim=1)
#         output=self.linear1(batch_cls)
#         output = self.dropout(output)
#         output=self.linear2(output)
#         output=self.relu(output)
#         output=self.linear3(output)
#         feat_dict = {}
#         feat_dict["label"] = self.g.nodes["label"].data["feats"]
#         feat_dict = self.heterograph(self.g, feat_dict)
#         g_out=feat_dict["label"]
#         g_out = g_out.reshape(-1, self.n_classes)
#         g_out=torch.max(g_out,dim=-1).values
#         output = torch.mul(output, g_out)
#         output=self.sigmoid(output)
#         return output



if __name__ == '__main__':
    # Hetero_label()
    print()
