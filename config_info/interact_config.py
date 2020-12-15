from config_info.train_config import Config
import os


class InteractConfig(Config):
    def __init__(self):
        self.temperature = 1  # 生成温度
        self.top_k = 8  # 最高8选1
        self.top_prob = 0  # 最高积累概率
        self.save_samples_path = os.path.join(self.data_dir, 'sample')
        self.repetition_penalty = 1  # 重复惩罚参数，若生成的对话重复性较高，可适当提高该参数
        self.max_len = 120  # 每个utterance的最大长度,超过指定长度则进行截断
        self.max_history_len = 5  # dialogue history的最大长度
