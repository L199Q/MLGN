
import torch
from tqdm import tqdm
from modelMLGN import MLGN
from util import Accuracy, read_configuration, save
from dataset import get_train_data_loader, get_test_data_loader, get_label_num
from transformers import AdamW, get_scheduler, logging, BertTokenizer, BertModel

from transformers import BertTokenizer, BertConfig, BertModel
import numpy as np
from log import Logger
import pickle
import json
def init(config):
    logging.set_verbosity_error()
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    config['device'] = torch.device(f"cuda:{config['gpu_id']}")

def build_train_model(config):
    print("get label number")
    label_number = get_label_num(config['dataset'])
    
    print("build bert model")
    bert_path = f"./checkpoint/{config['dataset']}_final"
    model_config = BertConfig.from_pretrained(bert_path)
    model_config.output_hidden_states = True
    bert = BertModel.from_pretrained(bert_path, config=model_config)
    bert = bert.to(config['device'])
    tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)

    print("build MLGN model")
    mlgn = MLGN(label_number=label_number, feature_layers=5, bert_hidden_size=bert.config.hidden_size).to(config['device'])
    mlgn.load_state_dict(torch.load(f"./checkpoint/{config['dataset']}_final/student_model_file.bin"))

    print("build test data loader")
    test_data_loader = get_test_data_loader(config['dataset'], tokenizer=tokenizer, batch_size=config['test_batch_size'])

    print("build BCE loss")
    classify_loss_function = torch.nn.BCEWithLogitsLoss()

    with  open(f"./data/{config['dataset']}/label_dict.pkl", "rb") as dict_file:
        label_dict = pickle.load(dict_file)
    new_data = dict(zip(label_dict.values(), label_dict.keys()))

    return test_data_loader, bert, mlgn, tokenizer, classify_loss_function,new_data




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

    out,_ = mlgn(bert_out)

    loss = loss_function(out, batch_label_one_hot)

    accuracy.calc(out, batch_label_one_hot)

    return loss, torch.sigmoid(out), batch_label_one_hot

def get_pre(a,new_data,logits, labels):
    logits = logits.detach().cpu()
    labels = labels.cpu().numpy()
    scores, indices = torch.topk(logits, k=10)   
    
    for index, label in enumerate(labels):
        dict={}
        label = list(set(np.nonzero(label)[0]))
        ture_label = [new_data[i] for i in label]
        
        labels = indices[index, :5].numpy()
        pre_label = [new_data[i] for i in labels]
        pre_score = scores[index, :5].numpy().tolist()

        dict["ture_label"] = ture_label
        dict["pre_label"] = pre_label
        dict["pre_score"] = pre_score
        a.append(dict)


    return a


if __name__ == '__main__':
    config = read_configuration("./config.yaml")

    init(config)
    print(config)
    
    accuracy = Accuracy()

    test_data_loader, bert, mlgn, tokenizer, classify_loss_function,new_data = build_train_model(config)

    save_acc1, save_acc3, save_acc5 = 0, 0, 0

    max_only_p5 = 0

    with torch.no_grad():
        bert.eval()
        mlgn.eval()
        # total、acc都为0
        accuracy.reset_acc()
        with tqdm(test_data_loader, ncols=200) as batch:
            b = []
            for data in batch:
               _loss, logits, labels = run_test_batch(config, data, bert, mlgn, classify_loss_function, accuracy)
                # 最后的
               json_pre = get_pre(b, new_data,logits, labels)
    
    filename = "./results/pre_MLGN.json"
    with open(filename, 'w') as file_obj:
        json.dump(json_pre, file_obj, indent=4)
    
            