## 简单说明

- 原版地址：https://github.com/yangjianxin1/GPT2-chitchat
- 简单修改了一下代码，对代码进行拆分，将log改成了print,方便在jupyter中进行训练，加入了valid loss求最佳模型的代码。
- 原版功能为对话，新版为文案生成（待优化）
- 数据集地址：https://tianchi.aliyun.com/dataset/dataDetail?dataId=9717
- 数据集预处理代码：`dialogue_generate.py`

- 代码说明：

```shell
├── config_info  # 配置文件
│   ├── interact_config.py  # 交互环境配置
│   ├── model_config_dialogue_small.json  # 模型训练配置文件，注意根据文章长度修改
│   ├── train_config.py  # 模型训练配置
│   └── vocab_small.txt  # 模型字典
├── data
│   ├── content_tag_dataset.txt  # 原始数据集1
│   ├── dialogue_model  # 导出的模型
│   │   └── best_checkpoint  #　最佳模型
│   │       ├── config.json
│   │       └── pytorch_model.bin
│   ├── item_desc_dataset.txt　# 原始数据集2
│   ├── sample  # 交互文件导出
│   │   └── samples.txt
│   ├── tensorboard_summary  # tensorboradx产生，用于绘图
│   ├── training.log  # 日志
│   ├── train_tokenized.txt  # tokenize后的训练集
│   └── train.txt  # 正式训练集
├── interact  # 交互，用于看代码生成效果
│   ├── interact_mmi.py  # 一次生成多个
│   └── interact.py  # 一次生成一个
├── main.py  # 主程序
├── README1.md  # 原版说明文档
├── README.md  # 说明文档
├── requirements.txt  # 配置文件
├── train.py  # 训练文件
└── utils  # 其它工具
    ├── data_preprocess.py  # 数据预处理
    ├── dataset.py  # 构建Dataset
    ├── dialogue_generate.py  # 用于处理原始数据集
    ├── generate_dialogue_subset.py  # 用于train_toenized.txt分割
    └── tools.py  # 用于保存模型，计算准确率等等。
```

- demo_数据集与模型地址（待完善）：[链接](https://cloud.189.cn/t/2URrqeQNbI7z)（访问码：of4m）

