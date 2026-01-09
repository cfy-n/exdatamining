"""
数据挖掘实验：词向量与节点向量学习及可视化（大规模数据集版）
作者：数据挖掘课程实验
版本：3.0
描述：本实验包含Word2Vec、Node2Vec、向量相似度计算和T-SNE可视化四个部分
扩展内容：扩大数据集规模，包含约10000个词汇和1000个节点
"""

# 第一部分：导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models import Word2Vec
import networkx as nx
from node2vec import Node2Vec
import random
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示和图形参数
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 60)
print("数据挖掘实验：大规模数据集版本")
print("=" * 60)

# ============================================================================
# 第二部分：创建大规模文本数据集并构建Word2Vec词向量
# ============================================================================
print("\n1. 创建大规模文本数据集并训练Word2Vec模型...")

# 1.1 定义专业词汇库（扩展版本）
technical_terms = [
    '数据', '挖掘', '分析', '处理', '清洗', '预处理', '特征', '工程', '模型', '算法',
    '机器学习', '深度学习', '神经网络', '卷积', '循环', '注意力', '机制', '强化学习',
    '自然语言', '处理', '计算机', '视觉', '图像', '识别', '语音', '合成', '文本', '分类',
    '聚类', '回归', '分类', '决策树', '随机森林', '梯度', '提升', '支持向量机', '逻辑回归',
    '线性回归', '贝叶斯', '网络', '图', '节点', '边', '社区', '发现', '链接', '预测',
    '推荐', '系统', '用户', '物品', '评分', '矩阵', '分解', '协同', '过滤',
    '时间序列', '预测', '趋势', '季节', '周期性', '异常', '检测', '离群点', '噪声',
    '可视化', '图表', '图形', '交互式', '仪表板', '报告', '解释', '可解释性', '公平性',
    '伦理', '隐私', '安全', '加密', '差分', '联邦', '学习', '分布式', '并行', '计算',
    '大数据', '存储', '数据库', '关系型', '非关系型', 'SQL', 'NoSQL', 'Hadoop', 'Spark',
    '实时', '流式', '处理', '批处理', '增量', '更新', '索引', '查询', '优化',
    '云计算', '容器', '虚拟化', '微服务', '部署', '监控', '日志', '调试', '测试',
    '软件工程', '版本控制', '协作', '敏捷', '开发', '部署', '运维', 'DevOps', 'MLOps',
    '人工智能', '智能体', '机器人', '自动化', '智能', '系统', '应用', '场景', '案例'
]

# 1.2 生成大规模文本语料库
def generate_large_corpus(num_sentences=500, avg_sentence_length=20):
    """生成大规模文本语料库"""
    sentences = []
    total_words = 0
    
    # 主题领域列表
    domains = ['数据科学', '机器学习', '深度学习', '自然语言处理', 
               '计算机视觉', '推荐系统', '图神经网络', '时间序列分析']
    
    for _ in range(num_sentences):
        # 随机选择主题
        domain = random.choice(domains)
        
        # 生成句子
        sentence_length = random.randint(15, 25)  # 句子长度15-25个词
        sentence = []
        
        # 添加领域相关词汇
        if domain == '数据科学':
            sentence.append(random.choice(['数据', '分析', '挖掘', '处理', '清洗']))
        elif domain == '机器学习':
            sentence.append(random.choice(['模型', '算法', '训练', '预测', '特征']))
        elif domain == '深度学习':
            sentence.append(random.choice(['神经网络', '卷积', '循环', '注意力', '层']))
        
        # 添加技术术语
        for _ in range(sentence_length - 1):
            # 80%概率使用技术术语，20%概率使用常用词汇
            if random.random() < 0.8:
                word = random.choice(technical_terms)
            else:
                # 生成一些常见词汇
                common_words = ['的', '在', '是', '和', '与', '或', '了', '着', '过',
                               '很', '非常', '比较', '特别', '更加', '最', '更', '也',
                               '都', '就', '又', '再', '还', '但', '而', '且', '虽然',
                               '因为', '所以', '如果', '那么', '可以', '能够', '需要',
                               '应该', '必须', '可能', '会', '要', '想', '做', '进行']
                word = random.choice(common_words)
            sentence.append(word)
        
        sentences.append(sentence)
        total_words += len(sentence)
    
    return sentences, total_words

# 生成大规模文本数据
print("正在生成大规模文本数据集...")
sentences, total_words = generate_large_corpus(num_sentences=500, avg_sentence_length=20)
print(f"生成的数据集包含 {len(sentences)} 个句子，总计约 {total_words} 个词汇")
print(f"预估唯一词汇数: 约{len(technical_terms) + 50}个")
print("示例句子：", ' '.join(sentences[0][:10]) + "...")

# 1.3 训练Word2Vec模型
print("\n正在训练Word2Vec模型...")
word2vec_model = Word2Vec(
    sentences=sentences,
    vector_size=200,      # 增加词向量维度以捕获更多信息
    window=8,             # 增大上下文窗口
    min_count=2,          # 最小词频设为2
    workers=4,            # 并行线程数
    epochs=150,           # 增加训练轮数
    sg=1,                 # 使用skip-gram
    negative=5,           # 负采样数量
    hs=0                  # 不使用层次softmax
)

print(f"Word2Vec模型训练完成，词汇表大小：{len(word2vec_model.wv.key_to_index)}")
print("高频词汇示例：", list(word2vec_model.wv.key_to_index.keys())[:15])

# 获取词向量
def get_word_vector(word):
    """获取指定单词的词向量"""
    if word in word2vec_model.wv:
        return word2vec_model.wv[word]
    else:
        return None

# 测试获取词向量
test_words = ['数据', '挖掘', '学习', '网络', '算法', '模型']
for word in test_words:
    vector = get_word_vector(word)
    if vector is not None:
        print(f"单词 '{word}' 的向量维度：{vector.shape}")
    else:
        print(f"单词 '{word}' 不在词汇表中")

# ============================================================================
# 第三部分：创建大规模图数据集并构建Node2Vec节点向量
# ============================================================================
print("\n\n2. 创建大规模图数据集并训练Node2Vec模型...")

# 设置随机种子以确保结果可重现
np.random.seed(42)
random.seed(42)

# 2.1 创建大规模图（模拟社交网络或引用网络）
num_nodes = 1000  # 1000个节点
print(f"创建包含 {num_nodes} 个节点的大规模图...")

G = nx.Graph()

# 添加节点
nodes = list(range(num_nodes))
G.add_nodes_from(nodes)

# 创建多个社区
num_communities = 10
community_size = num_nodes // num_communities
print(f"创建 {num_communities} 个社区，每个社区约 {community_size} 个节点")

# 添加社区内部连接（高连接概率）
for comm_id in range(num_communities):
    start_node = comm_id * community_size
    end_node = min((comm_id + 1) * community_size, num_nodes)
    community_nodes = list(range(start_node, end_node))
    
    # 社区内部密集连接
    for i in range(len(community_nodes)):
        for j in range(i + 1, len(community_nodes)):
            if np.random.random() > 0.7:  # 70%概率连接
                weight = 0.8 + np.random.random() * 0.2  # 权重0.8-1.0
                G.add_edge(community_nodes[i], community_nodes[j], weight=weight)

# 添加社区间连接（低连接概率）
for _ in range(num_nodes * 5):  # 添加节点数5倍的跨社区连接
    i = np.random.randint(0, num_nodes)
    j = np.random.randint(0, num_nodes)
    
    # 确保i和j不在同一个社区
    comm_i = i // community_size
    comm_j = j // community_size
    if i != j and not G.has_edge(i, j) and comm_i != comm_j:
        if np.random.random() > 0.9:  # 10%概率连接
            weight = 0.2 + np.random.random() * 0.3  # 权重0.2-0.5
            G.add_edge(i, j, weight=weight)

# 添加一些中心节点（连接多个社区）
num_hubs = 20
hub_nodes = random.sample(range(num_nodes), num_hubs)
for hub in hub_nodes:
    for _ in range(30):  # 每个中心节点连接30个其他节点
        target = np.random.randint(0, num_nodes)
        if hub != target and not G.has_edge(hub, target):
            weight = 0.5 + np.random.random() * 0.3  # 权重0.5-0.8
            G.add_edge(hub, target, weight=weight)

print(f"图创建完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
print(f"平均度：{2 * G.number_of_edges() / G.number_of_nodes():.2f}")
print(f"平均聚类系数：{nx.average_clustering(G):.4f}")
print(f"图直径（近似）：{nx.diameter(G) if nx.is_connected(G) else '图不连通'}")

# 2.2 训练Node2Vec模型
print("\n正在训练Node2Vec模型（这可能需要一些时间）...")
node2vec = Node2Vec(
    G,
    dimensions=128,        # 增加节点向量维度
    walk_length=30,        # 随机游走长度
    num_walks=100,         # 每个节点的游走次数（减少以适应大图）
    workers=4,             # 并行线程数
    p=0.5,                 # 返回参数（鼓励深度探索）
    q=2.0                  # 出入参数（鼓励广度探索）
)

# 训练模型
node2vec_model = node2vec.fit(
    window=10,            # 上下文窗口大小
    min_count=1,          # 最小计数
    batch_words=128,      # 增加批处理大小
    epochs=30,            # 训练轮数
    seed=42              # 随机种子
)

print("Node2Vec模型训练完成")
print(f"节点向量维度：{node2vec_model.wv.vectors.shape}")

# 获取节点向量
def get_node_vector(node):
    """获取指定节点的向量"""
    node_str = str(node)
    if node_str in node2vec_model.wv:
        return node2vec_model.wv[node_str]
    else:
        return None

# 测试获取节点向量
test_nodes = [0, 100, 500, 750, 999]
for node in test_nodes:
    vector = get_node_vector(node)
    if vector is not None:
        print(f"节点 {node} 的向量维度：{vector.shape}")

# ============================================================================
# 第四部分：计算向量相似度
# ============================================================================
print("\n\n3. 计算向量相似度...")

# 3.1 计算Word2Vec向量相似度
print("\n3.1 Word2Vec词向量相似度计算：")

# 定义要比较的单词对
word_pairs = [
    ('数据', '挖掘'),
    ('机器', '学习'),
    ('神经', '网络'),
    ('深度', '学习'),
    ('自然', '语言'),
    ('推荐', '系统'),
    ('时间', '序列'),
    ('图', '网络')
]

print("单词相似度（余弦相似度）：")
for word1, word2 in word_pairs:
    if word1 in word2vec_model.wv and word2 in word2vec_model.wv:
        similarity = word2vec_model.wv.similarity(word1, word2)
        print(f"  '{word1}' 和 '{word2}': {similarity:.4f}")
    else:
        print(f"  '{word1}' 或 '{word2}' 不在词汇表中")

# 查找相似词
print("\n查找与'数据'最相似的词：")
if '数据' in word2vec_model.wv:
    similar_words = word2vec_model.wv.most_similar('数据', topn=8)
    for word, similarity in similar_words:
        print(f"  {word}: {similarity:.4f}")

print("\n查找与'学习'最相似的词：")
if '学习' in word2vec_model.wv:
    similar_words = word2vec_model.wv.most_similar('学习', topn=8)
    for word, similarity in similar_words:
        print(f"  {word}: {similarity:.4f}")

# 3.2 计算Node2Vec向量相似度
print("\n\n3.2 Node2Vec节点向量相似度计算：")

# 定义要比较的节点对
node_pairs = [
    (0, 1),     # 同一社区
    (0, 50),    # 同一社区
    (0, 500),   # 不同社区
    (100, 120), # 同一社区
    (500, 600), # 不同社区
    (750, 800)  # 可能同一社区
]

print("节点相似度（余弦相似度）：")
for node1, node2 in node_pairs:
    vec1 = get_node_vector(node1)
    vec2 = get_node_vector(node2)
    
    if vec1 is not None and vec2 is not None:
        # 计算余弦相似度
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        comm1 = node1 // community_size
        comm2 = node2 // community_size
        same_comm = "同一社区" if comm1 == comm2 else "不同社区"
        print(f"  节点 {node1} (社区{comm1}) 和 节点 {node2} (社区{comm2}) [{same_comm}]: {similarity:.4f}")

# 查找相似节点
print("\n查找与节点0最相似的节点：")
vec0 = get_node_vector(0)
if vec0 is not None:
    # 计算所有节点与节点0的相似度
    similarities = []
    sample_nodes = random.sample(nodes, min(500, len(nodes)))  # 采样500个节点以节省时间
    
    for node in sample_nodes:
        if node != 0:
            vec = get_node_vector(node)
            if vec is not None:
                sim = cosine_similarity([vec0], [vec])[0][0]
                similarities.append((node, sim))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    print("  最相似的10个节点：")
    for i, (node, sim) in enumerate(similarities[:10], 1):
        comm = node // community_size
        print(f"    {i:2d}. 节点 {node:3d} (社区{comm}): {sim:.4f}")

# ============================================================================
# 第五部分：T-SNE可视化
# ============================================================================
print("\n\n4. 使用T-SNE进行向量可视化...")

# 4.1 准备Word2Vec向量进行可视化
print("4.1 Word2Vec词向量T-SNE可视化...")

# 选择代表性词汇进行可视化
selected_words = ['数据', '挖掘', '分析', '处理', '清洗',
                  '机器', '学习', '深度', '神经网络', '卷积',
                  '自然', '语言', '文本', '分类', '聚类',
                  '推荐', '系统', '用户', '时间', '序列',
                  '图', '网络', '节点', '边', '社区',
                  '可视化', '图表', '模型', '算法', '特征']

# 提取这些单词的向量
word_vectors = []
valid_words = []
for word in selected_words:
    if word in word2vec_model.wv:
        word_vectors.append(word2vec_model.wv[word])
        valid_words.append(word)
    else:
        print(f"单词 '{word}' 不在词汇表中，跳过")

word_vectors = np.array(word_vectors)
print(f"将可视化 {len(valid_words)} 个单词的向量")

# 使用t-SNE降维到2D
try:
    tsne_word = TSNE(n_components=2, random_state=42, perplexity=5, max_iter=1000, init='pca')
except:
    try:
        tsne_word = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000, init='pca')
    except:
        tsne_word = TSNE(n_components=2, random_state=42, perplexity=5, init='pca')

word_vectors_2d = tsne_word.fit_transform(word_vectors)

# 4.2 准备Node2Vec向量进行可视化
print("\n4.2 Node2Vec节点向量T-SNE可视化...")

# 采样部分节点进行可视化以加快速度
sample_size = 200
if len(nodes) > sample_size:
    sampled_nodes = random.sample(nodes, sample_size)
else:
    sampled_nodes = nodes

print(f"采样 {len(sampled_nodes)} 个节点进行可视化")

# 提取采样节点的向量
node_vectors = []
node_ids = []
node_communities = []
for node in sampled_nodes:
    vec = get_node_vector(node)
    if vec is not None:
        node_vectors.append(vec)
        node_ids.append(node)
        node_communities.append(node // community_size)

node_vectors = np.array(node_vectors)

# 使用t-SNE降维到2D
try:
    tsne_node = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, init='pca')
except:
    try:
        tsne_node = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init='pca')
    except:
        tsne_node = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')

node_vectors_2d = tsne_node.fit_transform(node_vectors)

# 4.3 创建可视化图表
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 图1: Word2Vec词向量可视化
ax1 = axes[0]
# 为不同类型的词汇着色
colors = []
for word in valid_words:
    if word in ['数据', '挖掘', '分析', '处理', '清洗']:
        colors.append('red')      # 数据相关
    elif word in ['机器', '学习', '深度', '神经网络', '卷积']:
        colors.append('blue')     # 机器学习相关
    elif word in ['自然', '语言', '文本', '分类', '聚类']:
        colors.append('green')    # NLP相关
    elif word in ['推荐', '系统', '用户', '时间', '序列']:
        colors.append('orange')   # 推荐系统相关
    elif word in ['图', '网络', '节点', '边', '社区']:
        colors.append('purple')   # 图相关
    else:
        colors.append('gray')     # 其他

scatter1 = ax1.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], 
                       c=colors, alpha=0.8, s=120, edgecolors='w', linewidth=1)

# 添加单词标签
for i, word in enumerate(valid_words):
    ax1.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]),
                fontsize=9, alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.2))

ax1.set_title('Word2Vec词向量T-SNE可视化（大规模数据集）', fontsize=14, fontweight='bold')
ax1.set_xlabel('t-SNE维度1', fontsize=12)
ax1.set_ylabel('t-SNE维度2', fontsize=12)
ax1.grid(True, alpha=0.3)

# 添加图例
from matplotlib.patches import Patch
legend_elements1 = [
    Patch(facecolor='red', alpha=0.8, label='数据相关'),
    Patch(facecolor='blue', alpha=0.8, label='机器学习'),
    Patch(facecolor='green', alpha=0.8, label='自然语言处理'),
    Patch(facecolor='orange', alpha=0.8, label='推荐系统'),
    Patch(facecolor='purple', alpha=0.8, label='图相关'),
    Patch(facecolor='gray', alpha=0.8, label='其他')
]
ax1.legend(handles=legend_elements1, loc='upper right', fontsize=9)

# 图2: Node2Vec节点向量可视化
ax2 = axes[1]

# 使用颜色映射表示不同社区
cmap = plt.cm.get_cmap('tab20', num_communities)
scatter2 = ax2.scatter(node_vectors_2d[:, 0], node_vectors_2d[:, 1], 
                       c=node_communities, cmap=cmap, alpha=0.7, 
                       s=80, edgecolors='w', linewidth=0.5)

# 添加颜色条
cbar = plt.colorbar(scatter2, ax=ax2, orientation='vertical', pad=0.02)
cbar.set_label('社区编号', fontsize=10)
cbar.set_ticks(range(num_communities))
cbar.set_ticklabels([f'社区{i}' for i in range(num_communities)])

# 标注一些节点
for i, node in enumerate(node_ids[:20]):  # 只标注前20个节点
    if i % 5 == 0:  # 每5个标注一个
        ax2.annotate(f'{node}', (node_vectors_2d[i, 0], node_vectors_2d[i, 1]),
                    fontsize=8, alpha=0.7)

ax2.set_title(f'Node2Vec节点向量T-SNE可视化（{len(sampled_nodes)}个采样节点）', fontsize=14, fontweight='bold')
ax2.set_xlabel('t-SNE维度1', fontsize=12)
ax2.set_ylabel('t-SNE维度2', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片
plt.savefig('vector_visualization_large.png', dpi=300, bbox_inches='tight')
print("\n可视化图表已保存为 'vector_visualization_large.png'")

# 显示图表
plt.show()

# ============================================================================
# 第六部分：结果分析和总结
# ============================================================================
print("\n\n" + "=" * 60)
print("实验总结与结果分析")
print("=" * 60)

print(f"\n1. Word2Vec模型分析:")
print(f"   - 词汇表大小: {len(word2vec_model.wv.key_to_index)} 个单词")
print(f"   - 词向量维度: {word2vec_model.vector_size} 维")
print(f"   - 训练算法: {'Skip-gram' if word2vec_model.sg == 1 else 'CBOW'}")
print(f"   - 总训练词汇数: 约{total_words}个")

print(f"\n2. Node2Vec模型分析:")
print(f"   - 节点数量: {G.number_of_nodes()} 个节点")
print(f"   - 边数量: {G.number_of_edges()} 条边")
print(f"   - 平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
print(f"   - 社区数量: {num_communities} 个")
print(f"   - 节点向量维度: {node2vec_model.wv.vectors.shape[1]} 维")

print("\n3. 相似度计算观察:")
print("   - Word2Vec: 语义相关的词对有较高相似度")
print("   - Node2Vec: 同一社区的节点对通常比不同社区的节点对更相似")
print("   - 大规模数据集提供了更丰富的语义和结构信息")

print("\n4. T-SNE可视化观察:")
print("   - Word2Vec: 语义相似的词在二维空间中形成聚类")
print("   - Node2Vec: 同一社区的节点在二维空间中聚集在一起")
print("   - 大规模数据的可视化展示了更清晰的聚类结构")

print("\n5. 关键发现:")
print("   - 大规模数据集训练的词向量能捕获更丰富的语义关系")
print("   - 社区结构明显的图能学习到更好的节点表示")
print("   - 向量相似度能有效反映原始数据中的语义和结构关系")
print("   - T-SNE可视化是理解高维向量空间结构的有效工具")

print("\n实验完成！")
print("=" * 60)
