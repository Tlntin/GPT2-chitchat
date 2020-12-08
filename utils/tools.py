from torch.nn import CrossEntropyLoss
import torch
import json
import os
from config_info.train_config import Config

config = Config()


def calculate_loss_and_accuracy(outputs, labels, device, pad_id):
    """
    计算非pad_id的平均loss和准确率
    :param outputs: 输出值
    :param labels: 标签值
    :param device: 训练设备
    :param pad_id: pad隐码，需要忽略
    :return:
    """
    logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
    # 用前n-1个token，预测出第n个token
    # 用第i个token的prediction_score用来预测第i+1个token。
    # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

    correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
    correct = correct.float().sum()

    # accuracy = correct / num_targets
    # loss = loss / num_targets
    return loss, correct, num_targets


def save_checkpoint(epoch, epochs_since_improvement, model, best_loss, is_best):
    """
    此函数用于保存最佳模型
    :param epoch: 第几轮训练
    :param epochs_since_improvement: 距离上次最佳模型已经经过了几轮
    :param model: 模型
    :param best_loss: 验证集最好的loss
    :param is_best: 是否是最佳模型
    """
    # 记录状态字典
    state_json = {'epoch': epoch,
                  'epochs_since_improvement': epochs_since_improvement,
                  'best_loss': best_loss
                  }
    # 储存当前状态到json中
    with open(config.state_path, 'wt', encoding='utf-8') as f:
        json.dump(state_json, f)
    # 储存模型、防止出现多卡GPU,所以需要判断是否有module这个模块
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(config.checkpoint_path)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    # 储存最佳模型
    if is_best:
        model_to_save.save_pretrained(config.best_checkpoint_path)
        print('最佳模型已经保存模型在{}路径'.format(config.best_checkpoint_path))


def load_checkpoint():
    # 加载状态字典
    if os.path.exists(config.state_path):
        with open(config.state_path, 'rt', encoding='utf-8') as f:
            state_json = json.load(f)
        epoch = state_json['epoch']
        epochs_since_improvement = state_json['epochs_since_improvement']
        best_loss = state_json['best_loss']
    else:
        epoch = 0
        epochs_since_improvement = 0
        best_loss = 10

    return epoch, epochs_since_improvement, best_loss


def save_epoch_csv(epoch, epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc):
    """
    此文件用户储存训练过程中训练集与验证集的准确率与loss
    """
    epoch_csv_path = os.path.join(config.data_dir, 'epoch.csv')
    if not os.path.exists(epoch_csv_path) or epoch == 0:
        f = open(epoch_csv_path, 'wt', encoding='utf-8-sig')
        columns = ['train_loss', 'train_acc', 'valid_loss', 'valid_acc']
        f.write(','.join(columns) + '\n')
        data1 = [epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc]
        data1 = ['{:4f}'.format(d) for d in data1]
        f.write(','.join(data1) + '\n')
        f.close()
    else:
        f = open(epoch_csv_path, 'at', encoding='utf-8-sig')
        data1 = [epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc]
        data1 = ['{:4f}'.format(d) for d in data1]
        f.write(','.join(data1) + '\n')
        f.close()



