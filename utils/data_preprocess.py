"""
数据预处理
"""
import os
import torch
from tqdm import tqdm
from config_info.train_config import Config
from sklearn.model_selection import train_test_split
from utils.dataset import MyDataset
from torch.utils.data import DataLoader


class DataPreProcess(Config):
    def __init__(self, tokenizer, n_ctx):
        """
        :param tokenizer:
        :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的dialogue进行截断
        """
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx

    def get_train_data(self):
        """
        加载原始数据
        :param
        """
        print("tokenizing raw data,raw data path:{}, token output path:{}" \
              .format(self.train_raw_path, self.train_tokenized_path))
        with open(self.train_raw_path, 'rb') as f:
            data = f.read().decode("utf-8")
        if "\r\n" in data:
            train_data = data.split("\r\n\r\n")
        else:
            train_data = data.split("\n\n")
        print("there are {} dialogue in raw dataset".format(len(train_data)))
        return train_data

    def preprocess_raw_data(self, train_data):
        """
        对原始语料进行处理，将原始语料转换为用于train的token id，对于每个dialogue，
        将其处于成如下形式"[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
        :param train_data: 处理好的原始数据
        :return:
        """
        with open(self.train_tokenized_path, "w", encoding="utf-8") as f:
            for dialogue_index, dialogue in enumerate(tqdm(train_data)):
                if "\r\n" in dialogue:
                    utterances = dialogue.split("\r\n")
                else:
                    utterances = dialogue.split("\n")
                dialogue_ids = [self.tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
                for utterance in utterances:
                    dialogue_ids.extend([self.tokenizer.convert_tokens_to_ids(word) for word in utterance])
                    dialogue_ids.append(self.tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
                # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
                dialogue_ids = dialogue_ids[:self.n_ctx]
                for dialogue_id in dialogue_ids:
                    f.write(str(dialogue_id) + ' ')
                # 最后一条记录不添加换行符
                if dialogue_index < len(train_data) - 1:
                    f.write("\n")
        print("finish preprocessing raw data,the result is stored in {}".format(self.train_tokenized_path))

    def preprocess_mmi_raw_data(self, train_data):
        """
        对原始语料进行处理，将原始语料的每段对话进行翻转，然后转换为用于train MMI模型的token id，
        对于每个dialogue，将其处于成如下形式"[CLS]utterance N[SEP]utterance N-1[SEP]utterance N-2[SEP]"
        :param
        :return:
        """
        print("tokenizing MMI raw data,raw data path:{}, token output path:{}"\
              .format(self.train_raw_path, self.train_mmi_tokenized_path))
        with open(self.train_mmi_tokenized_path, "w", encoding="utf-8") as f:
            for dialogue_index, dialogue in enumerate(tqdm(train_data)):
                if "\r\n" in dialogue:
                    utterances = dialogue.split("\r\n")
                else:
                    utterances = dialogue.split("\n")
                dialogue_ids = [self.tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
                for utterance in reversed(utterances):  # 将一段对话进行翻转
                    dialogue_ids.extend([self.tokenizer.convert_tokens_to_ids(word) for word in utterance])
                    dialogue_ids.append(self.tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
                # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
                dialogue_ids = dialogue_ids[:self.n_ctx]
                for dialogue_id in dialogue_ids:
                    f.write(str(dialogue_id) + ' ')
                # 最后一条记录不添加换行符
                if dialogue_index < len(train_data) - 1:
                    f.write("\n")
        print("finish preprocessing raw data,the result is stored in {}".format(self.train_tokenized_path))

    @staticmethod
    def collate_fn(batch):
        """
        计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
        :param batch:
        :return:
        """
        pad_id = 0
        input_ids = []
        btc_size = len(batch)
        max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
        # 计算该batch中input的最大长度
        for btc_idx in range(btc_size):
            if max_input_len < len(batch[btc_idx]):
                max_input_len = len(batch[btc_idx])
        # 使用pad_id对小于max_input_len的input_id进行补全
        for btc_idx in range(btc_size):
            input_len = len(batch[btc_idx])
            input_ids.append(batch[btc_idx])
            input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))
        return torch.tensor(input_ids, dtype=torch.long)

    def run(self):
        # 对原始数据进行预处理,将原始语料转换成对应的token_id
        if not os.path.exists(self.train_mmi_tokenized_path) and self.train_mmi:  # 如果当前是要训练MMI模型
            print('即将训练mini模型')
            train_data = self.get_train_data()
            self.preprocess_mmi_raw_data(train_data)
        elif not os.path.exists(self.train_tokenized_path) and not self.train_mmi:  # 如果当前是要训练对话生成模型
            train_data = self.get_train_data()
            self.preprocess_raw_data(train_data)
        else:
            print('已发现tokenizer文本，本次已无需预训练')
        # 正式加载数据
        print("开始加载训练数据")
        if self.train_mmi:  # 如果是训练MMI模型
            with open(self.train_mmi_tokenized_path, "r", encoding="utf8") as f:
                data = f.read()
        else:  # 如果是训练对话生成模型
            with open(self.train_tokenized_path, "r", encoding="utf8") as f:
                data = f.read()
        data_list = data.split("\n")
        train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=1)
        print('加载数据成功')
        # 构建训练测试集
        train_dataset = MyDataset(train_list)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers,
                                      collate_fn=self.collate_fn)
        test_dataset = MyDataset(test_list)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.num_workers,
                                     collate_fn=self.collate_fn)
        print('dataloader加载成功')
        return train_dataloader, test_dataloader
