from dgl.nn.pytorch import GraphConv
from config import Config
cfg=Config()

import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
class DGLMultiHeadHeteroGraphLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads1, num_heads2):
        super(DGLMultiHeadHeteroGraphLayer, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads1 = num_heads1
        self.num_heads2 = num_heads2
        self.etypes=["doc_label","label_doc","label_label","doc_doc"]
        self.heterograph1=dgl.nn.pytorch.HeteroGraphConv({etype:GATConv(in_feats=in_dim,out_feats=hidden_dim,num_heads=num_heads1)
                                                          for etype in self.etypes},aggregate="sum")
        self.heterograph2=dgl.nn.pytorch.HeteroGraphConv({etype:GATConv(in_feats=hidden_dim*num_heads1,out_feats=out_dim,num_heads=num_heads2)
                                                          for etype in self.etypes},aggregate="sum")
    def forward(self,G0,G1,feat_dict):
        feat_dict=self.heterograph1(G0,feat_dict)
        for key in feat_dict.keys():
            feat_dict[key] = F.relu(feat_dict[key].reshape(-1,self.num_heads1*self.hidden_dim))
        feat_dict = self.heterograph2(G1, feat_dict)
        for key in feat_dict.keys():
            feat_dict[key] = F.relu(feat_dict[key].reshape(-1,self.num_heads2*self.out_dim))
        return feat_dict
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,n_classes):
        super(Classifier, self).__init__()
        self.in_dim=in_dim
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim
        self.n_classes=n_classes
        self.num_heads1 = cfg.num_heads
        self.num_heads2 = cfg.num_heads
        self.gatlayer=DGLMultiHeadHeteroGraphLayer(in_dim=in_dim,hidden_dim=hidden_dim,out_dim=out_dim,num_heads1=self.num_heads1,num_heads2=self.num_heads2)
        self.dropout=nn.Dropout(0.4)
        self.linear_doc=nn.Linear(in_features=out_dim*4,out_features=n_classes)
        self.sigmoid=nn.Sigmoid()
    def forward(self,blocks,batch_inputs):
        feat_dict=self.gatlayer(blocks[0],blocks[1],batch_inputs)
        for key in feat_dict.keys():
            feat_dict[key]=self.sigmoid(self.linear_doc(self.dropout(feat_dict[key])))
        return feat_dict


if __name__ == '__main__':
    # Hetero_label()
    print()
