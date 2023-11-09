import torch
import torch.nn as nn
import torch.nn.functional as F

ACT2FN = {'elu': F.elu, 'relu': F.relu, 'sigmoid': torch.sigmoid, 'tanh': torch.tanh}

# 标签数量 output_size 此处是每个标签的概率
# context_size 瓶颈层
class labelmodel(nn.Module):
    def __init__(self, context_size, output_size, cornet_act='sigmoid', **kwargs):
        super(labelmodel, self).__init__()
        self.dstbn2cntxt = nn.Linear(output_size, context_size)
        self.dstbn2cntxt2 = nn.Linear(context_size, context_size)
        self.dstbn2cntxt3 = nn.Linear(context_size, context_size)
        self.dstbn2cntxt4 = nn.Linear(context_size, context_size)
        self.dstbn2cntxt5 = nn.Linear(context_size, context_size)
        self.cntxt2dstbn = nn.Linear(context_size, output_size)
        self.act_fn = ACT2FN[cornet_act]
    
    def forward(self, output_dstrbtn):        
        identity_logits = output_dstrbtn        
        output_dstrbtn = self.act_fn(output_dstrbtn)
        context_vector = self.dstbn2cntxt(output_dstrbtn)
        context_vector = F.elu(context_vector)
        context_vector = self.dstbn2cntxt2(context_vector)
        context_vector = F.elu(context_vector)
        context_vector = self.dstbn2cntxt3(context_vector)
        context_vector = F.elu(context_vector)
        context_vector = self.dstbn2cntxt4(context_vector)
        context_vector = F.elu(context_vector)
        context_vector = self.dstbn2cntxt5(context_vector)
        context_vector = F.elu(context_vector)
        output_dstrbtn = self.cntxt2dstbn(context_vector)
        output_dstrbtn = output_dstrbtn + identity_logits        
        return output_dstrbtn
    
# 标签数量 output_size 此处是每个标签的概率
# 瓶颈层 cornet_dim
    
class LabelModel(nn.Module):
    def __init__(self, output_size, corlayer_dim=600, n_labelmodel_blocks=1, **kwargs):
        super(LabelModel, self).__init__()
        self.intlv_layers = nn.ModuleList([labelmodel(corlayer_dim, output_size, **kwargs) for _ in range(n_labelmodel_blocks)])
        for layer in self.intlv_layers:
            nn.init.xavier_uniform_(layer.dstbn2cntxt.weight)
            nn.init.xavier_uniform_(layer.dstbn2cntxt2.weight)
            nn.init.xavier_uniform_(layer.dstbn2cntxt3.weight)
            nn.init.xavier_uniform_(layer.dstbn2cntxt4.weight)
            nn.init.xavier_uniform_(layer.dstbn2cntxt5.weight)
            nn.init.xavier_uniform_(layer.cntxt2dstbn.weight)

    def forward(self, logits):        
        for layer in self.intlv_layers:
            logits = layer(logits)        
        return logits