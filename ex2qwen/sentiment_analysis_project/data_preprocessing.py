import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def load_and_preprocess_data(file_path):
    """
    加载和预处理CSV数据
    """
    # 读取数据（根据提供的格式，标签在前，文本在后）
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line) > 2:  # 确保行不为空且有足够长度
                # 第一个字符是标签，后面是文本
                label = line[0]
                text = line[2:].strip()
                if label in ['1', '2'] and text:
                    data.append({'text': text, 'label': int(label)})
    
    df = pd.DataFrame(data)
    print(f"原始数据量: {len(df)}")
    print(f"标签分布:\n{df['label'].value_counts()}")
    
    return df

def clean_text(text):
    """
    文本清洗函数
    """
    if pd.isna(text):
        return ""
    
    # 转换为小写
    text = text.lower()
    
    # 移除特殊字符和多余空格
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def balance_dataset(df):
    """
    平衡数据集（处理类别不平衡）
    """
    # 分离两个类别
    df_negative = df[df['label'] == 1]
    df_positive = df[df['label'] == 2]
    
    print(f"负面样本数: {len(df_negative)}")
    print(f"正面样本数: {len(df_positive)}")
    
    # 上采样少数类
    if len(df_negative) < len(df_positive):
        df_negative_upsampled = resample(df_negative,
                                        replace=True,
                                        n_samples=len(df_positive),
                                        random_state=42)
        df_balanced = pd.concat([df_positive, df_negative_upsampled])
    else:
        df_positive_upsampled = resample(df_positive,
                                        replace=True,
                                        n_samples=len(df_negative),
                                        random_state=42)
        df_balanced = pd.concat([df_negative, df_positive_upsampled])
    
    print(f"平衡后数据量: {len(df_balanced)}")
    return df_balanced.sample(frac=1, random_state=42)  # 打乱数据

def prepare_data(file_path):
    """
    完整的数据预处理流程
    """
    # 1. 加载数据
    df = load_and_preprocess_data(file_path)
    
    # 2. 文本清洗
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # 3. 平衡数据集
    df_balanced = balance_dataset(df)
    
    # 4. 划分训练集和测试集
    train_df, test_df = train_test_split(df_balanced, 
                                        test_size=0.2, 
                                        random_state=42,
                                        stratify=df_balanced['label'])
    
    # 将标签转换为0/1（负面/正面）
    train_df['label'] = train_df['label'] - 1  # 1->0(负面), 2->1(正面)
    test_df['label'] = test_df['label'] - 1
    
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    
    return train_df, test_df

# 测试数据预处理
if __name__ == "__main__":
    train_df, test_df = prepare_data("test.csv")
    print("\n示例数据:")
    print(train_df[['text', 'label']].head())