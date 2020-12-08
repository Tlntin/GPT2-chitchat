import torch
import os
from config_info.train_config import Config
from train import set_random_seed, create_model, train
from utils.data_preprocess import DataPreProcess
from transformers import BertTokenizer

config = Config()

# 当用户使用GPU,并且GPU可用时
device1 = torch.device('cuda' if config.cuda else 'cpu')
print('using device:{}'.format(device1))
# 为CPU设置种子用于生成随机数，以使得结果是确定的
# 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
# 当得到比较好的结果时我们通常希望这个结果是可以复现
if config.seed:
    set_random_seed(config)
# 设置使用哪些显卡进行训练
os.environ["CUDA_VISIBLE_DEVICES"] = config.device
# 初始化tokenizer
tokenizer = BertTokenizer(vocab_file=config.vocab_path)
# tokenizer的字典大小
vocab_size1 = len(tokenizer)
# 加载GPT2模型
model1, n_ctx1, multi_gpu1 = create_model(vocab_size1, device1)
# 加载数据集
data = DataPreProcess(tokenizer, n_ctx1)
train_dataloader1, test_dataloader1 = data.run()
# 开始训练
train(model1, device1, train_dataloader1, test_dataloader1, multi_gpu1, config)