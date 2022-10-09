import platform
class Config(object):
    def __init__(self):
        self.sentence_len=300
        self.epochs=60
        self.learning_rate=0.001
        self.embedding_dim=768
        self.hidden_dim1=256
        self.hidden_dim2=256
        self.num_heads=4
        operating = platform.system()
        if operating == 'Linux':
            self.bert_path = '../../bert-base-uncased'
        elif operating == 'Windows':
            self.bert_path = 'D:\\bert-base-uncased'
        self.device="cuda"
        self.parameter=""
        self.batch_size=128
        self.setup_seed(20)
        self.datasets_path= "../data/datasets_abstract2_5w.txt"
        self.word2vec_path= ""

    # 设置随机数种子
    def setup_seed(self,seed):
        print("set random seed:",seed)
        import torch
        import numpy as np
        import random
        import dgl
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        dgl.seed(seed)
if __name__ == '__main__':
    print(platform.system())
