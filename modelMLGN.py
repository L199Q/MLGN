import torch
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
from labelnet import LabelModel
from torch.nn import functional as F
class MLGN(torch.nn.Module):
    def __init__(self, label_number, feature_layers, bert_hidden_size):
        super(MLGN, self).__init__()

        self.feature_layers = feature_layers
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(feature_layers * bert_hidden_size, label_number)
        self.labelmodel = LabelModel(label_number)
        # self.linear = torch.nn.Linear(bert_hidden_size, label_number)
    
    # 此处x是bert的输出，所以MLGN模块是bert的微调
    def forward(self, x):
        # 取'hidden_states'的原因：获取每层的输出结果，此处获取后10层的cls向量
        out = x['hidden_states']
        #out = out[-1][:, 0]
        out = torch.cat([out[-i][:, 0] for i in range(1, self.feature_layers + 1)], dim=-1)
        out = self.dropout(out)
        out2 = out
        # 将 out--->映射到标签数量的维数，这样才能训练
        out = self.linear(out)
        out = out + self.labelmodel(out)
        return out, F.normalize(out2,dim=1)





