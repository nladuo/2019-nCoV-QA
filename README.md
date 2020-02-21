# 2019-nCov-QA
跑一遍问答的流程，一个问答系统的Demo，数据为2019肺炎新闻数据。

## 架构
IR + MRC
## 检索模块（IR）
0使用ElasticSearch。

#### 1.安装ElasticSearch7

#### 2.导入数据到Mongo
```bash
cd ir
git clone https://github.com/BlankerL/DXY-COVID-19-Data.git
```
 

## 阅读理解模块（MRC）
基于baidu/DuReader数据集，代码数据来自：[https://github.com/yanx27/DuReader_QANet_BiDAF](https://github.com/yanx27/DuReader_QANet_BiDAF)。

## 效果

