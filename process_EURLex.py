import sys
sys.path.append('..')
import pandas
import pickle
import json
from util import NLP


def clear_text(text: str):
    result = text.replace('"', " ").replace("\n", " ").replace("'", " ").replace("`", " ")
    return nlp.word_tokenize(result, lower=True)


def build_label_dict():
    with open("data/EURLex-4k/labels.txt", "r", encoding="utf-8") as file:
        label_list = [data for data in file.read().splitlines()]
        label_dict = {label: idx for idx, label in enumerate(label_list)}
        with open("data/EURLex-4k/label_dict.pkl", "wb") as pkl_file:
            pickle.dump(label_dict, pkl_file)


def build_data_csv_2_json():
    file_name_list = ["train.csv", "test.csv"]
    for file_name in file_name_list:
        csv_data = pandas.read_csv(f"data/EURLex-4k/{file_name}", header=0)
        text_list = csv_data['text']
        label_list = csv_data['label']

        data_list = list()
        for text, label in zip(text_list, label_list):
            temp_dict = dict()
            temp_dict["text"] = clear_text(text)
            temp_dict["label"] = [data for data in label.split(",")]

            data_list.append(temp_dict)

        with open(f"data/EURLex-4k/processed_{file_name[:-4]}.json", "w", encoding="utf-8") as file:
            json.dump(data_list, file)


if __name__ == '__main__':
    nlp = NLP(path='en_core_web_sm')
    build_label_dict()
    build_data_csv_2_json()

