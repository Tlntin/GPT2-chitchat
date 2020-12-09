"""
第一步：文件预处理
"""
from config_info.train_config import Config
from tqdm import tqdm
from jieba import analyse, lcut
from jieba import posseg
from collections import Counter
import os
import time

text_rank = analyse.textrank
config = Config()


class DataPreProcess(object):
    """
    文件预处理
    :param file_path1: item的文件路径
    :param file_path2: content的文件路径
    """
    def __init__(self, file_path1, file_path2, min_length=60, max_length=150):
        """
        :param
        """
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.min_length = min_length
        self.max_length = max_length

    @staticmethod
    def generate_dialog(title, text):
        """
        抽取关键词
        :param
        """
        keyword_set = set(text_rank(text))
        title_set = set(text_rank(title))
        keyword_set = keyword_set.intersection(title_set)
        return keyword_set

    def process_file1(self):
        """
        对文件1进行处理
        :param
        """
        with open(self.file_path1, 'rt') as f:
            text_list = f.readlines()
        title_list = [text.split()[0] for text in text_list]
        content_list = [text.split()[-1] for text in text_list]
        assert len(title_list) == len(content_list)
        print('开始构建语料库1')
        material_list = [[title, content] for title, content in zip(title_list, content_list) if\
                         self.min_length < len(content) < self.max_length]
        print('过滤前共有{}条语料，过滤后有{}条语料'.format(len(content_list), len(material_list)))
        data_iter = tqdm(material_list)
        result_text = ''
        i = 0
        for title, content in data_iter:
            # 抽取关键词
            data_iter.set_description('正在抽取关键词，生成文案问答语料')
            keyword_set = self.generate_dialog(title, content)
            keyword_text = ' '.join(keyword_set)
            keyword_list = posseg.lcut(keyword_text)
            # 抽取名词关键词
            keyword_list = [k.strip() for k, v in keyword_list if 'n' in v and len(k.strip()) > 0]
            if len(keyword_list) > 0:
                for keyword in keyword_set:
                    temp_text = keyword + '\n' + title.strip() + '\n\n'
                    temp_text += keyword + '\n' + content.strip() + '\n\n'
                    result_text += temp_text

            if i > 2000:
                break
            i += 1
        with open(os.path.join(config.data_dir, 'train1.txt'), 'wt', encoding='utf-8') as f:
            f.write(result_text)
        print('语料1写入成功')

    def process_file2(self):
        """
        对文件2进行处理
        :param
        """
        print('开始构建预料库2')
        with open(self.file_path2, 'rt') as f:
            text_list = f.readlines()
        title_list = [text.split()[0] for text in text_list]
        material_list = [title for title in title_list if self.min_length < len(title) < self.max_length]
        print('过滤前共有{}条语料，过滤后有{}条语料'.format(len(title_list), len(material_list)))
        result_text = ''
        i = 0
        data_iter = tqdm(material_list)
        for title in data_iter:
            # 抽取关键词
            data_iter.set_description('正在抽取关键词，生成文案问答语料')
            keyword_list = text_rank(title)
            keyword_text = ' '.join(keyword_list)
            keyword_list = posseg.lcut(keyword_text)
            # 抽取名词关键词
            print(keyword_list)
            keyword_list = [k.strip() for k, v in keyword_list if 'n' in v and len(k.strip()) > 0]
            if len(keyword_list) > 0:
                for keyword in keyword_list:
                    temp_text = keyword + '\n' + title.strip() + '\n'
                    temp_text += '\n'  # 多一个换行代表该广告结束
                    result_text += temp_text
            if i > 100:
                break
            i += 1
        with open(os.path.join(config.data_dir, 'train2.txt'), 'wt', encoding='utf-8') as f:
            f.write(result_text)
        print('语料2写入成功')


if __name__ == '__main__':
    file1 = os.path.join(config.data_dir, 'item_desc_dataset.txt')
    file2 = os.path.join(config.data_dir, 'content_tag_dataset.txt')
    data1 = DataPreProcess(file1, file2)
    # data1.filter_file()
    data1.process_file1()
    # data1.process_file2()


