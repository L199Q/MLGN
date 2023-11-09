import json
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from util import read_configuration
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class AppDataset(Dataset):
    def __init__(self, dataset, data_type, tokenizer):
        super(AppDataset, self).__init__()

        config = read_configuration("./config.yaml")

        with open(f"./data/{dataset}/processed_{data_type}.json", "r", encoding="utf-8") as data_file, \
                open(f"./data/{dataset}/label_dict.pkl", "rb") as dict_file:
            self.json_data_list = json.load(data_file)
            self.label_dict = pickle.load(dict_file)
            self.tokenizer = tokenizer



    def __getitem__(self, item):
        json_data = self.json_data_list[item]
        text_encoder = self.tokenizer.encode(json_data['text'], add_special_tokens=True, max_length=512, truncation=True)
        label_encoder = self.tokenizer.encode(" ".join(json_data['label']), add_special_tokens=True)
        label_one_hot = torch.zeros(len(self.label_dict)).scatter(0,
                                                                  torch.tensor([self.label_dict[data] for data in json_data["label"]]),
                                                                  torch.tensor([1.0 for _ in json_data["label"]]))
        return text_encoder, label_encoder, label_one_hot

    def __len__(self):
        return len(self.json_data_list)

    def process_data(self, cache_save_path):
        try:
            data = torch.load(cache_save_path)
            return data["text_encoder_list"], data["label_encoder_list"], data["label_one_hot_list"]
        except FileNotFoundError:
            text_encoder_list = []
            label_encoder_list = []
            label_one_hot_list = []
            for json_data in tqdm(self.json_data_list):
                text_encoder = self.tokenizer.encode(json_data['text'], add_special_tokens=True, max_length=512, truncation=True)
                label_encoder = self.tokenizer.encode(" ".join(json_data['label']), add_special_tokens=True)
                label_one_hot = torch.zeros(len(self.label_dict)).scatter_(0, torch.tensor(
                    [self.label_dict[data] for data in json_data["label"]]), 1)

                text_encoder_list.append(text_encoder)
                label_encoder_list.append(label_encoder)
                label_one_hot_list.append(label_one_hot)

            data = {
                "text_encoder_list": text_encoder_list,
                "label_encoder_list": label_encoder_list,
                "label_one_hot_list": label_one_hot_list
            }
            torch.save(data, cache_save_path)

            return text_encoder_list, label_encoder_list, label_one_hot_list


def bert_collate_fn(batches):
    batch_text = []
    batch_label = []
    batch_label_one_hot = []
    for batch in batches:
        batch_text.append(batch[0])
        batch_label.append(batch[1])
        batch_label_one_hot.append(batch[2])

    batch_text_input_ids, text_padding_mask, text_token_type_ids = padding(batch_text, 0)
    batch_label_input_ids, label_padding_mask, label_token_type_ids = padding(batch_label, 0)

    return batch_text_input_ids, text_padding_mask, text_token_type_ids, \
           batch_label_input_ids, label_padding_mask, label_token_type_ids, \
           torch.stack(batch_label_one_hot)



def padding(inputs, pad_idx, max_len=None):
    if max_len is None:
        lengths = [len(inp) for inp in inputs]
        max_len = max(lengths)
    padded_inputs = torch.as_tensor([inp + [pad_idx] * (max_len - len(inp)) for inp in inputs], dtype=torch.long)
    # mask
    masks = torch.as_tensor([[1] * len(inp) + [0] * (max_len - len(inp)) for inp in inputs], dtype=torch.int)
    # token_type_ids
    token_type_ids = torch.zeros(padded_inputs.shape, dtype=torch.int)
    return padded_inputs, masks, token_type_ids


def get_train_data_loader(dataset, tokenizer, batch_size=16, num_workers=6):
    train_dataset = AppDataset(dataset, "train", tokenizer)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=bert_collate_fn)

    return train_data_loader


def get_test_data_loader(dataset, tokenizer, batch_size=16, num_workers=6):
    test_dataset = AppDataset(dataset, "test", tokenizer)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=bert_collate_fn)

    return test_data_loader


def get_train_sampler_data_loader(dataset, tokenizer, batch_size=16, num_workers=6, num_replicas=None, rank=None):
    train_dataset = AppDataset(dataset, "train", tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=num_replicas, rank=rank)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=bert_collate_fn, sampler=train_sampler)

    return train_data_loader, train_sampler


def get_test_sampler_data_loader(dataset, tokenizer, batch_size=16, num_workers=6, num_replicas=None, rank=None):
    test_dataset = AppDataset(dataset, "test", tokenizer)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=num_replicas, rank=rank)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=bert_collate_fn, sampler=test_sampler)

    return test_data_loader


def get_label_num(dataset):
    with open(f"./data/{dataset}/label_dict.pkl", "rb") as file:
        label_dict = pickle.load(file)
        return len(label_dict)

