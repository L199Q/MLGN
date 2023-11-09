from log import Logger
import torch
from tqdm import tqdm
from modelLabelInfo import MLGN
from util import Accuracy, read_configuration, save
from dataset import get_train_data_loader, get_test_data_loader, get_label_num
from transformers import AdamW, get_scheduler, logging, BertTokenizer, BertModel
from loss import SupConLoss

from transformers import BertTokenizer, BertConfig, BertModel
def init(config):
    logging.set_verbosity_error()
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    config['device'] = torch.device(f"cuda:{config['gpu_id']}")

bert_path = './model/bert-base-uncased'
def build_train_model(config):
    print("get label number")
    label_number = get_label_num(config['dataset'])

    print("build bert model")
    model_config = BertConfig.from_pretrained(bert_path)
    model_config.output_hidden_states = True
    bert = BertModel.from_pretrained(bert_path, config=model_config)
    bert = bert.to(config['device'])
    tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)

    
    print("build MLGN model")
    mlgn = MLGN(label_number=label_number, feature_layers=5, bert_hidden_size=bert.config.hidden_size).to(config['device'])

    print("build train data loader")
    train_data_loader = get_train_data_loader(config['dataset'], tokenizer=tokenizer, batch_size=config['train_batch_size'])

    print("build test data loader")
    test_data_loader = get_test_data_loader(config['dataset'], tokenizer=tokenizer, batch_size=config['test_batch_size'])

    print("build BCE loss")
    classify_loss_function = torch.nn.BCEWithLogitsLoss()

    contrastive_loss = SupConLoss(temperature=5)
 

    print("build optimizer")
    bert_train_parameters = [parameter for parameter in bert.parameters() if parameter.requires_grad]
    mlgn_train_parameters = [parameter for parameter in mlgn.parameters() if parameter.requires_grad]
    parameters = bert_train_parameters + mlgn_train_parameters
    optimizer = AdamW(parameters, lr=config['lr'])

    print("build lr_scheduler")
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=config['epochs'] * len(train_data_loader))

    return train_data_loader, test_data_loader, bert, mlgn,tokenizer, classify_loss_function,contrastive_loss, optimizer, lr_scheduler



def run_train_batch(config, data, bert, mlgn, classify_loss_function,contrastive_loss, optimizer, lr_scheduler):
    batch_text_input_ids, batch_text_padding_mask, batch_text_token_type_ids, \
    batch_label_input_ids, batch_label_padding_mask, batch_label_token_type_ids, \
    batch_label_one_hot = data

    batch_text_input_ids = batch_text_input_ids.to(config['device'])
    batch_text_padding_mask = batch_text_padding_mask.to(config['device'])
    batch_text_token_type_ids = batch_text_token_type_ids.to(config['device'])
    batch_label_input_ids = batch_label_input_ids.to(config['device'])
    batch_label_padding_mask = batch_label_padding_mask.to(config['device'])
    # 16
    batch_label_token_type_ids = batch_label_token_type_ids.to(config['device'])
    # 14
    batch_label_one_hot = batch_label_one_hot.to(config['device'])


    text_bert_out = bert(input_ids=batch_text_input_ids, attention_mask=batch_text_padding_mask,
                    token_type_ids=batch_text_token_type_ids, output_hidden_states=True)
    # def forward(self, x, flag_text=False,x_text=None):
    out_outcome,a = mlgn(text_bert_out)
    text_classify_loss = classify_loss_function(out_outcome, batch_label_one_hot)


    label_bert_out = bert(input_ids=batch_label_input_ids, attention_mask=batch_label_padding_mask,
                    token_type_ids=batch_label_token_type_ids, output_hidden_states=True)




    out,b= mlgn(label_bert_out)
    label_classify_loss = classify_loss_function(out, batch_label_one_hot)
    #对比学习'''
    text_label = torch.stack([a,b],1).reshape([a.shape[0]*2,3840])
    # label = torch.stack([batch_label_one_hot,batch_label_one_hot],1).reshape([a.shape[0]*2,batch_label_one_hot.shape[1]])
    
    mask = torch.arange(0,a.shape[0])
    mask = torch.stack([mask,mask],1).reshape([1,a.shape[0]*2])
    contrastive = contrastive_loss(text_label, mask)

    loss = text_classify_loss  + label_classify_loss + 0.01*contrastive

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    return text_classify_loss, label_classify_loss, contrastive



def run_test_batch(config, data, bert, mlgn, loss_function, accuracy):
    batch_text_input_ids, batch_text_padding_mask, batch_text_token_type_ids, \
    batch_label_input_ids, batch_label_padding_mask, batch_label_token_type_ids, \
    batch_label_one_hot = data

    batch_text_input_ids = batch_text_input_ids.to(config['device'])
    batch_text_padding_mask = batch_text_padding_mask.to(config['device'])
    batch_text_token_type_ids = batch_text_token_type_ids.to(config['device'])
    batch_label_one_hot = batch_label_one_hot.to(config['device'])

    bert_out = bert(input_ids=batch_text_input_ids, attention_mask=batch_text_padding_mask,
                    token_type_ids=batch_text_token_type_ids, output_hidden_states=True)

    out, a = mlgn(bert_out)

    loss = loss_function(out, batch_label_one_hot)

    accuracy.calc(out, batch_label_one_hot)

    return loss


if __name__ == '__main__':
    config = read_configuration("./config.yaml")

    init(config)
    print(config)
    
    accuracy = Accuracy()

    train_data_loader, test_data_loader, bert, mlgn, tokenizer, loss_function,contrastive_loss, optimizer, lr_scheduler = build_train_model(config)

    save_acc1, save_acc3, save_acc5 = 0, 0, 0

    max_only_p5 = 0

    for epoch in range(config['epochs']):
        bert.train()
        mlgn.train()
        with tqdm(train_data_loader, ncols=200) as batch:
            for data in batch:
                text_classify_loss,label_classify_loss,contrastive = run_train_batch(config, data, bert, mlgn, loss_function, contrastive_loss, optimizer, lr_scheduler)
                batch.set_description(f"train epoch:{epoch + 1}/{config['epochs']}")
                batch.set_postfix(text_classify_loss=text_classify_loss.item(), label_classify_loss=label_classify_loss.item(), contrastive=contrastive.item())
        with torch.no_grad():
            bert.eval()
            mlgn.eval()
            accuracy.reset_acc()
            with tqdm(test_data_loader, ncols=200) as batch:
                for data in batch:
                    _loss = run_test_batch(config, data, bert, mlgn, loss_function, accuracy)
                    batch.set_description(f"test epoch:{epoch + 1}/{config['epochs']}")
                    loss=_loss.item()
                    p1=accuracy.get_acc1()
                    p3=accuracy.get_acc3()
                    p5=accuracy.get_acc5()
                    d3=accuracy.get_ndcg3()
                    d5=accuracy.get_ndcg5()
                    batch.set_postfix(loss=loss, p1=p1, p3=p3, p5=p5, d3=d3, d5=d5)
        log_str = f'{epoch:>2}: {p1:.4f}, {p3:.4f}, {p5:.4f}, {d3:.4f}, {d5:.4f}, train_loss:{loss}'
        LOG = Logger(config['bert_version']+'LabelInfo')
        LOG.log(log_str)
        if max_only_p5 < p5:
            max_only_p5 = p5
            save(config['bert_version']+'LabelInfo', log_str, bert, tokenizer, mlgn)