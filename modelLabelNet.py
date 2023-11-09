import torch
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
from labelnet import LabelModel

class MLGN(torch.nn.Module):
    def __init__(self, label_number, feature_layers, bert_hidden_size):
        super(MLGN, self).__init__()

        self.feature_layers = feature_layers
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(feature_layers * bert_hidden_size, label_number)
        self.LabelModel = LabelModel(label_number)
    def forward(self, x):
        out = x['hidden_states']
        out = torch.cat([out[-i][:, 0] for i in range(1, self.feature_layers + 1)], dim=-1)
        out = self.dropout(out)
        out = self.linear(out)
        out = out + self.LabelModel(out)
        
        return out


