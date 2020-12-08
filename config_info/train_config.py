import os
import torch


class Config(object):
    device = '0'  # 默认使用第1块显卡
    no_cuda = False  # 不使用显卡，默认为False
    cuda = torch.cuda.is_available() and not no_cuda
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 父路径
    config_dir = os.path.join(parent_dir, 'config_info')
    model_config = os.path.join(config_dir, 'model_config_dialogue_small.json')  # 模型配置文件路径
    vocab_path = os.path.join(config_dir, 'vocab_small.txt')  # 字典所在路径
    data_dir = os.path.join(parent_dir, 'data')  # 数据存放文件夹
    train_raw_path = os.path.join(data_dir, 'train.txt')  # 原始训练集
    train_tokenized_path = os.path.join(data_dir, 'train_tokenized.txt')  # 字典化后的训练集
    log_path = os.path.join(data_dir, 'training.log')  # 日志存放

    # -- 模型相关参数 --#
    batch_size = 8
    epochs = 20  # 训练20次
    lr = 1.5e-4
    warmup_steps = 2000
    gradient_accumulation = 1  # 梯度累积，多少次更新一次
    max_grad_norm = 1  # 梯度剪裁
    dialogue_model_output_path = os.path.join(data_dir, 'dialogue_model')  # 对话模型输出路径

    # -- 模型其它参数 -- #
    writer_dir = os.path.join(data_dir, 'tensorboard_summary')
    seed = None   # 随机种子
    num_workers = 1  # 加载dataloader时用的线程数
    log_step = 100  # 每隔100个记录一次
    train_mmi = False  # 默认不训练mini模型
    train_mmi_tokenized_path = os.path.join(data_dir, 'train_mmi_tokenized.txt')
    mmi_model_output_path = os.path.join(data_dir, 'mmi_model')

    if train_mmi:
        # 创建MMI模型的输出目录
        if not os.path.exists(mmi_model_output_path):
            os.mkdir(mmi_model_output_path)
        checkpoint_path = os.path.join(mmi_model_output_path, 'checkpoint')
        best_checkpoint_path = os.path.join(mmi_model_output_path, 'best_checkpoint')
        state_path = os.path.join(mmi_model_output_path, 'state.json')
    else:
        # 创建对话模型的输出目录
        if not os.path.exists(dialogue_model_output_path):
            os.mkdir(dialogue_model_output_path)
        checkpoint_path = os.path.join(dialogue_model_output_path, 'checkpoint')
        best_checkpoint_path = os.path.join(dialogue_model_output_path, 'best_checkpoint')
        state_path = os.path.join(dialogue_model_output_path, 'state.json')
