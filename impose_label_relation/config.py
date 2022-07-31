import platform
import torch
print(torch.cuda.is_available())
class Config():
    def __init__(self):
        self.lamda=100
        self.bidirectional=False
        if(self.bidirectional):
            self.directions=2
        else:
            self.directions = 1
        self.batch_size = 128#840
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = 0.0005
        self.epoches=120
        self.hidden_dim = 256
        self.lstm_hidden_dim = 128
        self.embedding_dim = 100
        self.dropout_query=0.5
        self.dropout_attention=0.4
        self.dropout_lstm=0.6
        self.n_layers=1

        self.datasets_path = "../data/datasets_abstract2_5w.txt"
        self.sentence_len=256
        self.entityname_word_num=10

        operating = platform.system()
        if operating == 'Linux':
            self.bert_path = '../../../bert-base-uncased'
            self.word2vec_path = "../../glove/glove50/word_vector.txt"
            self.word2vec_path = "../../glove/glove100/word_vector.txt"
            self.word2vec_path = "../../glove/glove300/word_vector.txt"
        else:
            self.bert_path = "D:\\bert-base-uncased"
            # self.word2vec_path="D:\glove\glove50\word_vector.txt"
            self.word2vec_path = "D:\glove\glove100\word_vector.txt"
            # self.word2vec_path="D:\glove\glove300\word_vector.txt"
        self.setup_seed(20)




    # 设置随机数种子
    def setup_seed(self, seed):
        print("set random seed:", seed)
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
    import torch
    a=torch.tensor([[1,2,3,4],[0,1,4,5]])
    print(torch.max(a,dim=0).values)

