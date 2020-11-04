# 鼻咽癌结构化方法探究
我们采用BERT-CRF模型来对MRI鼻咽癌报告进行结构化，并且生成了鼻咽癌的知识网络。
## 环境要求
以下是我进行实验的packages版本  
python --> 3.6  
tensorflow --> 1.13.1  
numpy --> 1.16.4  
openpyxl --> 2.6.2  
BERT预训练模型：[chinese_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

## 数据预处理
总共有769条数据，按照6：2：2划分为训练集、验证集和测试集。数据标注采用BISO标注法，实体标签为三元组 <(P,S),A,V>标签。我们已经将文本数据和对应的标签转化为id的形式，
分别保存为naso_id_train.pkl, naso_id_dev.pkl, naso_id_test.pkl。以便于输入NER模型。
## 模型训练
我们比较了五个常用的命名实体识别模型（CNN-CRF、IDCNN-CRF、LSTM-CRF、BiLSTM-CRF、BERT-CRF），并选择最优模型做为实验的NER模型。
其中，前四个模型可以在文件夹ner_train中获取，如果您想自己训练模型，只需运行对应模型的py文件即可。由于BERT模型文件太大，需要您自己下载BERT预训练模型，
并且在您的机器上训练。在训练BERT模型时，您可以在`train_helper.py`中自定义模型的一些参数，然后运行`BERT_CRF.py`即可。  

以下是我们在训练模型使用的参数：  
词嵌入维度为50，  
最大序列长度为50，  
dropout层的keep_prob为0.8，  
batch_size为100，  
epochs为50，  
学习率为0.001

个别参数如下：  
**CNN-CRF**：卷积层深度为3层，卷积个数为18个  
**IDCNN-CRF**：卷积层深度为3层，，卷积个数为18个，膨胀因子分别为1，1，2 ，迭代次数为4  
**LSTM-CRF**：隐藏层神经元数目为18  
**BiLSTM-CRF**：隐藏层神经元数目为18  
**BERT-CRF**：由于机器配置的限制，BERT的batch_size为8，epochs为10
## 结果比较
BERT-CRF的性能是最优的，因此我们选择BERT-CRF做为我们的NER模型。  
![模型对比](https://github.com/saynHuang/npc_structure/raw/main/data/model%20comparison.png)
## 文本结构化
若要实现文本结构化，运行`main.py`即可，此函数提供两种模式：一是句子级别的结构化，对应text_predict函数，输入参数是一段文字，程序会自动进行结构化，输出结构化信息；
二是文档级别的结构化，对应batch_process函数，输入参数有两个，一个是需要结构化的文本文件(txt)，一个是结构化后保存的文件(xlsx)。
## 知识网络
我们统计了出现次数大于等于20的主实体词，然后从这些词语中筛选出有关鼻咽部的描述，最终以这些词语的三元组内容进行知识网络的可视化。  
![知识网络](https://github.com/saynHuang/npc_structure/raw/main/data/knowledge%20network.png)



# Nasopharyngeal Carcinoma Structure
We use the BERT-CRF model to structure Chinese MRI reports, and visualized the nasopharyngeal carcinoma knowledge network.
## requirments
python --> 3.6  
tensorflow --> 1.13.1  
numpy --> 1.16.4  
openpyxl --> 2.6.2  
BERT pre-train model: [chinese_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
## Data preprocessing
There are a total of 769 pieces of data, divided into training set, validation set and test set according to 6:2:2. 
Data labeling adopts the BISO labeling method, and the entity label is the triplet <(P,S),A,V>. 
We have converted the text data and corresponding labels into the id, which matches the input of the network.
Save as naso_id_train.pkl, naso_id_dev.pkl, naso_id_test.pkl.
## Training Model
We compare five named entity recognition models (CNN-CRF、IDCNN-CRF、LSTM-CRF、BiLSTM-CRF、BERT-CRF). 
Except for BERT-CRF, the other four trained models are stored in ner_train. If you want to train the BERT-CRF model, 
you should download the pre-train model and run `BERT_CRF.py`. 
You can also set the hyperparameters of the BERT-CRF model in the `train_helper.py`.  
The following are the hyperparameters we used in the experiment:  
embedding_dim = 50  
max_seq_length = 50  
dropout_keep = 0.8  
batch_size = 100  
epochs = 50  
learning_rate = 0.001  

Additional parameters for different models：  
**CNN-CRF**：convolution_depth=3, num_of_filter=18  
**IDCNN-CRF**：convolution_depth=3, dilation is {1,1,2}, num_of_filter=18, iterations=4  
**LSTM-CRF**：Number of hidden layer units is 18   
**BiLSTM-CRF**：Number of hidden layer units is 18  
**BERT-CRF**：batch_size=8，epochs=10

## Comparison
The performance of BERT-CRF model is better than the other four, so we choose BERT-CRF as our NER model.  
![model comparison](https://github.com/saynHuang/npc_structure/raw/main/data/model%20comparison.png)

## Text structure
run `main.py` for text structure. This function provides two modes: one is sentence-level structuring, 
it will call text_predict function, the input is text, the program will print the stuctured triplet;
The second is document-level structuring, call the batch_process function, there are two input parameters, 
one is text direction(.txt) that needs to be structured, and the other is output file direction(.xlsx).
## Knowledge Network
We counted the primary entities that appeared more than or equal to 20, and then filtered out the description of the nasopharynx from these words, 
and finally visualized the nasopharyngeal carcinoma knowledge network.  
![knowledge network](https://github.com/saynHuang/npc_structure/raw/main/data/knowledge%20network.png)

