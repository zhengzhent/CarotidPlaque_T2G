import os

import torch
import torch.nn as nn
from bin.t2g_former import T2GFormer, Tokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from pyecharts import options as opts
from pyecharts.charts import Graph
# 读取csv数据
df = pd.read_csv('F:\A_MyWork\Research\CarotidPlaqueVASystem\PlaqueData\Temp3.csv',encoding='utf-8')

#测试和弦图

def seed_everything(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)
# 获取输入特征维度
namelist=df.columns
print(namelist)
d_numerical = 536 # 数值特征列数
categories = [2,3,2,2,2,2,2,1,1,2,1,2,2,1,1,2,1,2,5] # 分类特征的类别列表
d_token = 512  #what is d_token
# 定义模型
model = T2GFormer(
    n_layers=6,  #层数
    d_token=d_token,#模型隐层的向量维度
    n_heads=8,
    d_ffn_factor=4,
    attention_dropout=0.1,
    ffn_dropout=0.1,
    residual_dropout=0,
    activation='gelu',
    prenormalization=True,
    initialization='kaiming',
    kv_compression=None,
    kv_compression_sharing=None,
    d_out=1,
    d_numerical=d_numerical,
    categories=categories,
    token_bias=True
)

tokenizer = Tokenizer(d_numerical, categories, d_token, True)
num_cols = df.iloc[:, 20:556].astype(np.float32)
x_num = torch.tensor(num_cols.values, dtype=torch.float32)  #数值列变量
cat_cols=df.iloc[:, 1:20].astype(np.int64)
x_cat = torch.tensor(df.iloc[:, 1:20].values, dtype=torch.long)  #分类列变量

print(x_num.max())
print(x_num.min())
print(x_cat.max())
print(x_cat.min())

x = tokenizer(x_num, x_cat)
attn_weights, graphs = model(x_num, x_cat, return_fr=True)
print('attn_weights:',attn_weights)  #attention头?
print()

print('layer of graphs:',len(graphs))
print()

graphs = graphs[-2]  #倒数第二层
print("graph's shape:",graphs.shape)
print()
print("The graph:",graphs)
print()

adj_matrix = graphs[0,2].detach().numpy()  #选择第一个记录的第二个头
print("The adj_matrix:",adj_matrix)
print("Size of adj_matrix",adj_matrix.shape)

#print(adj_matrix.shape)
print()

init_opts = opts.InitOpts(width="100%",  # 图宽
                          height="900px",  # 图高
                          renderer="canvas",  # 渲染模式 svg 或 canvas，即 RenderType.CANVAS 或 RenderType.SVG
                          page_title="Feature Graph",  # 网页标题
                          js_host=""  # js主服务位置 留空则默认官方远程主服务
                          )
label_opts = opts.LabelOpts(is_show=True,position='right')
nodes = []
for i in range(555):
    for j in range (555):
        if adj_matrix[i,j] > 0.03:
            nodes.append({'name': ''+namelist[i+1]})

linestyle_opts = opts.LineStyleOpts(is_show=True,
                                            # width=adj_matrix[i,j]*100,
                                            opacity=0.6,
                                            curve=0.3,
                                            type_="solid",
                                            color="orange")
edges = []
for i in range(555):
    for j in range(555):
        if adj_matrix[i,j] > 0.03 :
            edges.append({'source':namelist[i+1], 'target':namelist[j+1]})
print(nodes)
#print(edges)
c = Graph(init_opts)
c.add("", nodes, edges,
      repulsion=8000,
      edge_length=100,
      label_opts=label_opts,
      layout="circular",
      is_rotate_label=True,
      )
c.set_global_opts(
    title_opts=opts.TitleOpts(title="Feature Graph")
)
c.render('CarotidPlaqueResults\Temp3_ChrodGraph_LessEdges.html')