## Facenet-Torch 人脸识别Python后端接口
---

## 部署步骤

需要安装`conda`环境

```bash
conda create -n face python=3.8

conda activate face
```

```bash
cd waiter-face-torch-main
```

```bash
pip install -r requirements.txt
```

启动service
```bash
python service.py
```

- 本地需要配置好数据库，然后在`service.py`中配置数据库相关信息


## 训练步骤
1. 本文使用如下格式进行训练。
```
|-datasets
    |-people0
        |-123.jpg
        |-234.jpg
    |-people1
        |-345.jpg
        |-456.jpg
    |-...
```  
2. 下载好数据集，将训练用的CASIA-WebFaces数据集以及评估用的LFW数据集，解压后放在根目录。
3. 在训练前利用txt_annotation.py文件生成对应的cls_train.txt。  
4. 利用train.py训练facenet模型，训练前，根据自己的需要选择backbone，model_path和backbone一定要对应。
5. 运行train.py即可开始训练。

## 评估步骤
1. 下载好评估数据集，将评估用的LFW数据集，解压后放在根目录
2. 在eval_LFW.py设置使用的主干特征提取网络和网络权值。
3. 运行eval_LFW.py来进行模型准确率评估。

## Reference
https://github.com/davidsandberg/facenet  
https://github.com/timesler/facenet-pytorch  
