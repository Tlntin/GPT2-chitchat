import random
from datetime import datetime
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from utils.data_preprocess import *
from utils.tools import calculate_loss_and_accuracy, save_checkpoint, load_checkpoint, save_epoch_csv

config = Config()
PAD = '[PAD]'
pad_id = 0


def set_random_seed(args):
    """
    设置训练的随机种子
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(vocab_size, device):
    """

    :param vocab_size:字典大小
    :param device: 设备CPU/GPU
    :return:
    """
    if os.path.exists(config.checkpoint_path):  # 如果上次训练文件存在
        model = GPT2LMHeadModel.from_pretrained(config.checkpoint_path)
    else:  # 若没有指定预训练模型，则初始化模型
        model_config = GPT2Config.from_json_file(config.model_config)
        model = GPT2LMHeadModel(config=model_config)
    # 根据tokenizer的vocabulary调整GPT2模型的voca的大小
    model.resize_token_embeddings(vocab_size)
    nct_x = model.config.to_dict().get("n_ctx")
    # print('model config:\n{}'.format(model.config.to_json_string()))
    model = model.to(device)
    # 是否使用多块GPU进行并行运算
    if config.cuda and torch.cuda.device_count() > 1:
        print("开始使用多GPU进行训练")
        model = DataParallel(model, device_ids=[int(i) for i in config.device.split(',')])
        multi_gpu = True
    elif config.cuda:
        print('当前使用单张GPU进行训练')
        multi_gpu = False
    else:
        print('当前使用CPU进行训练')
        multi_gpu = False
    # 记录模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of model parameters: {}'.format(num_parameters))
    return model, nct_x, multi_gpu


def train(model, device, train_dataloader, test_dataloader, multi_gpu, args):
    # 初始参数
    start_epoch, epochs_since_improvement, best_loss = load_checkpoint()
    model.train()
    # 计算所有epoch进行参数优化的总步数total_steps
    total_steps = int(len(train_dataloader) * args.epochs / args.gradient_accumulation)
    print('total training steps = {}'.format(total_steps))
    # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)
    print('开始训练')
    # 用于统计每次梯度累计的loss
    running_loss = 0
    # 统计一共训练了多少个step
    overall_step = 0
    # 记录tensorboardX
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    # 记录 out of memory的次数
    # 开始训练
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = datetime.now()
        running_train_loss = 0
        running_train_correct = 0  # 记录预测正确的值
        running_train_num = 0
        data_iter = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, input_ids in data_iter:
            # 注意：GPT2模型的forward()函数，是对于给定的context，生成一个token，而不是生成一串token
            # GPT2Model的输入为n个token_id时，输出也是n个hidden_state，使用第n个hidden_state预测第n+1个token
            input_ids = input_ids.to(device)
            outputs = model.forward(input_ids=input_ids)
            loss, correct, num_targets = calculate_loss_and_accuracy(outputs, input_ids, device, pad_id)
            # -- 加入epoch_loss, epoch_acc -- #
            running_train_loss += loss.item()
            running_train_correct += correct
            running_train_num += num_targets
            # train_temp_loss, train_temp_acc
            loss = loss / num_targets
            accuracy = correct / num_targets
            if multi_gpu:
                loss = loss.mean()
                accuracy = accuracy.mean()
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
                accuracy = accuracy / args.gradient_accumulation
            loss.backward()
            # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                running_loss += loss.item()
                # 更新参数
                optimizer.step()
                # 清空梯度信息
                optimizer.zero_grad()
                # 进行warm up
                scheduler.step()
                overall_step += 1
                # 更新tnesorboardX信息
                if (overall_step + 1) % args.log_step == 0:
                    tb_writer.add_scalar('loss', loss.item(), overall_step)
            # 输出最新loss与acc
            data_iter.set_description("epoch:{}/{}, train_loss:{:.4f}, train_acc:{:.2f}%"\
                                      .format(epoch + 1, config.epochs, loss, accuracy * 100))
        epoch_train_loss = running_train_loss / running_train_num
        epoch_train_acc = running_train_correct / running_train_num
        epoch_valid_loss, epoch_valid_acc = evaluate(epoch, model, device, test_dataloader, multi_gpu, args)
        print('epoch: {} / {} train_loss:{:.4f}, train_acc:{:.2f}% \
                 valid_loss:{:.4f}, valid_acc:{:.2f}%'. \
              format(epoch + 1, config.epochs, epoch_train_loss,epoch_train_acc * 100,
                     epoch_valid_loss, epoch_valid_acc * 100))
        # 保存epoch训练结果
        save_epoch_csv(epoch, epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc)
        # 开始保存模型
        is_best = epoch_valid_loss < best_loss
        best_loss = min(epoch_valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            best_loss = epoch_valid_loss
            epochs_since_improvement = 0
        save_checkpoint(epoch, epochs_since_improvement, model, best_loss, is_best)
        epoch_finish_time = datetime.now()
        print('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
    print('training finished')


def evaluate(epoch, model, device, test_dataloader, multi_gpu, args):
    """
    评估模型
    :param
    """
    print("开始评估模型")
    model.eval()
    running_valid_loss = 0
    running_valid_correct = 0  # 记录预测正确的值
    running_valid_num = 0
    # 记录tensorboardX
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    with torch.no_grad():
        data_iter = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for batch_idx, input_ids in data_iter:
            input_ids = input_ids.to(device)
            outputs = model.forward(input_ids=input_ids)
            loss, correct, num_targets = calculate_loss_and_accuracy(outputs, input_ids, device, pad_id)
            # -- 加入epoch_loss, epoch_acc -- #
            running_valid_loss += loss.item()
            running_valid_correct += correct
            running_valid_num += num_targets
            # 计算step_loss
            loss = loss / num_targets
            accuracy = correct / num_targets
            if multi_gpu:
                loss = loss.mean()
                accuracy = accuracy.mean()
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
                accuracy = accuracy / args.gradient_accumulation
            data_iter.set_description("epoch:{}/{}, valid_loss:{:.4f}, valid_acc:{:.2f}%"\
                                      .format(epoch, config.epochs, loss, accuracy * 100))
            # tb_writer.add_scalar('loss', loss.item(), overall_step)
    # 计算epoch acc, epoch_loss
    epoch_valid_loss = running_valid_loss / running_valid_num
    epoch_valid_acc = running_valid_correct / running_valid_num
    return epoch_valid_loss, epoch_valid_acc


if __name__ == '__main__':
    pass

