# UniLM_chatbot
使用UniLM和HNSW构建任务型+闲聊型机器人。

+ 意图识别：fasttext

+ 检索模型：
  - 使用Word Average Model (WAM) 表示句向量
  - 使用Hierarchical Navigable Small World(HNSW)做召回，包括两个包的使用：hnswlib和Faiss
  - 使用LightGBM做排序，手工构建特征包括：
    - 基于字符串距离的（编辑距离、列文斯坦距离、LCS）
    - 基于向量距离的（cosine、Euclidian、Jaccard、WMD）
    - 基于统计量的（BM25、Pearson Correlation）
    - 基于深度匹配模型的

+ 生成模型：UniLM（重头写一个BERT，然后修改attention_mask和计算loss的部分即可得到UniLM）



训练生成模型的数据：https://cloud.tsinghua.edu.cn/f/f131a4d259184566a29c/

