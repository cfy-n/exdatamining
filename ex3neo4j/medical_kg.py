#!/usr/bin/env python3
"""
医学知识图谱构建与可视化系统
功能：从医学文本中提取实体和关系，构建Neo4j知识图谱
作者：AI编程助手
日期：2024
"""

import re
import json
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict
import pandas as pd

# ==================== 数据模型定义 ====================

class EntityType(Enum):
    """实体类型枚举"""
    GENE = "基因"
    DISEASE = "疾病"
    SYMPTOM = "症状"
    DRUG = "药物"
    TREATMENT = "治疗"
    ORGAN = "器官"
    CHROMOSOME = "染色体"
    PROTEIN = "蛋白质"
    PATHWAY = "通路"

class RelationType(Enum):
    """关系类型枚举"""
    CAUSES = "导致"
    TREATS = "治疗"
    MANIFESTS = "表现为"
    TARGETS = "靶向"
    LOCATED_ON = "位于"
    METABOLIZES = "代谢"
    ASSOCIATED_WITH = "与...相关"
    INHIBITS = "抑制"
    ACTIVATES = "激活"
    REGULATES = "调节"
    MUTATES_TO = "突变为"
    INDUCES = "诱发"
    PREDISPOSES = "易患"

@dataclass
class Entity:
    """实体类"""
    name: str
    entity_type: EntityType
    attributes: Dict = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

@dataclass
class Relation:
    """关系类"""
    source: Entity
    target: Entity
    relation_type: RelationType
    attributes: Dict = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

# ==================== 文本处理器 ====================

class MedicalTextProcessor:
    """医学文本处理器"""
    
    def __init__(self):
        # 初始化实体识别模式
        self.entity_patterns = {
            EntityType.GENE: [
                r'[A-Z0-9]{2,}(?:-[A-Z0-9]+)*基因',
                r'(?:EGFR|ALK|HER2|neu|P53|ER|PR|SCN5A|VKORC1|CYP2D6|CYP2C19|SLC6A4|HPRT1)基因',
                r'[A-Z]{2,}[0-9]*[A-Z]*基因',
                r'[A-Z]+基因'
            ],
            EntityType.DISEASE: [
                r'[^\s，。；]*?(?:癌|瘤|综合征|症|病|疾病)',
                r'非小细胞肺癌|乳腺癌|Brugada综合征|Lesch-Nyhan综合征',
                r'[^\s，。；]*?肿瘤',
                r'[^\s，。；]*?抑郁症'
            ],
            EntityType.SYMPTOM: [
                r'[^\s，。；]*?(?:痛|咳|血|闷|困难|肿|转移|障碍|异常|改变)',
                r'[^\s，。；]*?(?:症状|表现)',
                r'干咳|咯血|胸闷|呼吸困难|骨痛|头痛|呕吐|淋巴结肿大|发热|猝死|晕厥',
                r'情绪低落|兴趣减退|快感缺失|思维迟缓|自伤行为'
            ],
            EntityType.DRUG: [
                r'[^\s，。；]*?(?:替尼|单抗|韦|嗪|酮|醇|平|素)',
                r'厄洛替尼|吉非替尼|阿法替尼|达克替尼|奥希替尼|克唑替尼|阿来替尼|劳拉替尼',
                r'曲妥珠单抗|帕妥珠单抗|T-DM1|吡咯替尼|卡培他滨|奎尼丁|华法林|舍曲林|帕罗西汀|西酞普兰|别嘌醇|地西泮'
            ],
            EntityType.TREATMENT: [
                r'[^\s，。；]*?(?:治疗|疗法|手术|靶向治疗|化疗|放疗|免疫治疗)',
                r'靶向治疗|化疗|内分泌治疗|基因治疗|康复治疗|对症支持治疗'
            ],
            EntityType.CHROMOSOME: [
                r'[0-9XY]{1,2}号染色体',
                r'[0-9XY]{1,2}号'
            ]
        }
        
        # 关系提取模式
        self.relation_patterns = {
            r'([^\s，。；]+基因)[^\s，。；]*?(?:导致|引起|引发|诱发|调控|影响|决定|关联)[^\s，。；]*?([^\s，。；]+疾病)': RelationType.CAUSES,
            r'([^\s，。；]+药物)[^\s，。；]*?(?:治疗|针对|用于|缓解|抑制)[^\s，。；]*?([^\s，。；]+疾病)': RelationType.TREATS,
            r'([^\s，。；]+疾病)[^\s，。；]*?(?:表现为|症状包括|特征为)[^\s，。；]*?([^\s，。；]+症状)': RelationType.MANIFESTS,
            r'([^\s，。；]+药物)[^\s，。；]*?(?:靶向|作用于|针对)[^\s，。；]*?([^\s，。；]+基因)': RelationType.TARGETS,
            r'([^\s，。；]+基因)[^\s，。；]*?位于[^\s，。；]*?([^\s，。；]+染色体)': RelationType.LOCATED_ON,
            r'([^\s，。；]+基因)[^\s，。；]*?(?:代谢|编码)[^\s，。；]*?([^\s，。；]+药物)': RelationType.METABOLIZES
        }
        
        # 已知实体词典（用于提高识别准确率）
        self.known_entities = {
            EntityType.GENE: {
                'EGFR', 'ALK', 'HER2', 'neu', 'P53', 'ER', 'PR', 'SCN5A', 
                'VKORC1', 'CYP2D6', 'CYP2C19', 'SLC6A4', 'HPRT1'
            },
            EntityType.DISEASE: {
                '非小细胞肺癌', '乳腺癌', 'Brugada综合征', 'Lesch-Nyhan综合征',
                '肿瘤', '心血管疾病', '神经代谢性疾病', '抑郁症'
            },
            EntityType.DRUG: {
                '厄洛替尼', '吉非替尼', '阿法替尼', '达克替尼', '奥希替尼',
                '克唑替尼', '阿来替尼', '劳拉替尼', '曲妥珠单抗', '帕妥珠单抗',
                'T-DM1', '吡咯替尼', '卡培他滨', '奎尼丁', '华法林', '舍曲林',
                '帕罗西汀', '西酞普兰', '别嘌醇', '地西泮'
            }
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """从文本中提取实体"""
        entities = []
        seen = set()
        
        # 首先匹配已知实体
        for entity_type, known_set in self.known_entities.items():
            for entity_name in known_set:
                if entity_name in text and entity_name not in seen:
                    entities.append(Entity(
                        name=entity_name,
                        entity_type=entity_type,
                        attributes={'source': 'known_dictionary'}
                    ))
                    seen.add(entity_name)
        
        # 使用正则表达式匹配
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]  # 处理分组匹配
                    if match and match not in seen:
                        entities.append(Entity(
                            name=match,
                            entity_type=entity_type,
                            attributes={'source': 'regex_pattern', 'pattern': pattern}
                        ))
                        seen.add(match)
        
        return entities
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """从文本中提取关系"""
        relations = []
        
        # 创建实体名称到对象的映射
        entity_map = {entity.name: entity for entity in entities}
        
        # 使用正则表达式匹配关系
        for pattern, relation_type in self.relation_patterns.items():
            matches = re.findall(pattern, text)
            for source_name, target_name in matches:
                if source_name in entity_map and target_name in entity_map:
                    relations.append(Relation(
                        source=entity_map[source_name],
                        target=entity_map[target_name],
                        relation_type=relation_type,
                        attributes={'source': 'regex_pattern', 'pattern': pattern}
                    ))
        
        # 基于共现的简单关系提取（如果实体在同一个句子中出现）
        sentences = re.split(r'[。！？；]', text)
        for sentence in sentences:
            sentence_entities = [e for e in entities if e.name in sentence]
            for i, source in enumerate(sentence_entities):
                for target in sentence_entities[i+1:]:
                    if source.entity_type != target.entity_type:
                        # 根据实体类型推测关系类型
                        relation_type = self._infer_relation_type(source, target)
                        if relation_type:
                            relations.append(Relation(
                                source=source,
                                target=target,
                                relation_type=relation_type,
                                attributes={'source': 'co-occurrence', 'sentence': sentence[:50] + '...'}
                            ))
        
        return relations
    
    def _infer_relation_type(self, source: Entity, target: Entity) -> RelationType:
        """根据实体类型推断关系类型"""
        type_pairs = {
            (EntityType.GENE, EntityType.DISEASE): RelationType.CAUSES,
            (EntityType.GENE, EntityType.DRUG): RelationType.TARGETS,
            (EntityType.DRUG, EntityType.DISEASE): RelationType.TREATS,
            (EntityType.DISEASE, EntityType.SYMPTOM): RelationType.MANIFESTS,
            (EntityType.DRUG, EntityType.SYMPTOM): RelationType.TREATS,
        }
        
        return type_pairs.get((source.entity_type, target.entity_type), None)
    
    def process_text(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """处理文本，提取实体和关系"""
        entities = self.extract_entities(text)
        relations = self.extract_relations(text, entities)
        return entities, relations

# ==================== Neo4j 数据库管理器 ====================

class Neo4jManager:
    """Neo4j数据库管理器"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", 
                 password: str = "password"):
        """
        初始化Neo4j连接
        
        参数:
            uri: Neo4j数据库URI
            username: 用户名
            password: 密码
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
    def connect(self):
        """连接到Neo4j数据库"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, 
                                              auth=(self.username, self.password))
            print(f"✅ 成功连接到Neo4j数据库: {self.uri}")
            return True
        except ImportError:
            print("❌ 未安装neo4j驱动，请运行: pip install neo4j")
            return False
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def create_entity_node(self, entity: Entity):
        """创建实体节点"""
        if not self.driver:
            print("❌ 未连接到数据库")
            return False
            
        with self.driver.session() as session:
            query = """
            MERGE (n:Entity {name: $name})
            SET n.type = $type,
                n.attributes = $attributes,
                n.confidence = $confidence,
                n.created_at = timestamp()
            RETURN n
            """
            result = session.run(query, 
                               name=entity.name,
                               type=entity.entity_type.value,
                               attributes=entity.attributes,
                               confidence=entity.confidence)
            return result.single() is not None
    
    def create_relation(self, relation: Relation):
        """创建关系"""
        if not self.driver:
            print("❌ 未连接到数据库")
            return False
            
        with self.driver.session() as session:
            # 确保源节点和目标节点存在
            self.create_entity_node(relation.source)
            self.create_entity_node(relation.target)
            
            # 创建关系
            query = """
            MATCH (source:Entity {name: $source_name})
            MATCH (target:Entity {name: $target_name})
            MERGE (source)-[r:RELATIONSHIP {type: $rel_type}]->(target)
            SET r.attributes = $attributes,
                r.confidence = $confidence,
                r.created_at = timestamp()
            RETURN r
            """
            result = session.run(query,
                               source_name=relation.source.name,
                               target_name=relation.target.name,
                               rel_type=relation.relation_type.value,
                               attributes=relation.attributes,
                               confidence=relation.confidence)
            return result.single() is not None
    
    def create_schema(self):
        """创建数据库约束和索引"""
        with self.driver.session() as session:
            # 创建唯一约束
            constraints = [
                "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE",
                "CREATE CONSTRAINT relationship_type IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() REQUIRE r.type IS NOT NULL"
            ]
            
            # 创建索引
            indexes = [
                "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)",
                "CREATE INDEX relationship_source IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.source_name)"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except:
                    pass
            
            for index in indexes:
                try:
                    session.run(index)
                except:
                    pass
    
    def clear_database(self):
        """清空数据库（谨慎使用！）"""
        confirm = input("⚠️  确定要清空数据库吗？(yes/no): ")
        if confirm.lower() != 'yes':
            print("操作已取消")
            return
            
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ 数据库已清空")
    
    def query_entities(self, entity_type: str = None, limit: int = 100):
        """查询实体"""
        with self.driver.session() as session:
            if entity_type:
                query = "MATCH (n:Entity) WHERE n.type = $type RETURN n LIMIT $limit"
                result = session.run(query, type=entity_type, limit=limit)
            else:
                query = "MATCH (n:Entity) RETURN n LIMIT $limit"
                result = session.run(query, limit=limit)
            
            entities = []
            for record in result:
                entities.append(record["n"])
            return entities
    
    def query_relations(self, relation_type: str = None, limit: int = 100):
        """查询关系"""
        with self.driver.session() as session:
            if relation_type:
                query = """
                MATCH (source)-[r:RELATIONSHIP]->(target) 
                WHERE r.type = $type 
                RETURN source.name as source, r.type as type, target.name as target 
                LIMIT $limit
                """
                result = session.run(query, type=relation_type, limit=limit)
            else:
                query = """
                MATCH (source)-[r:RELATIONSHIP]->(target) 
                RETURN source.name as source, r.type as type, target.name as target 
                LIMIT $limit
                """
                result = session.run(query, limit=limit)
            
            relations = []
            for record in result:
                relations.append({
                    'source': record["source"],
                    'type': record["type"],
                    'target': record["target"]
                })
            return relations
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            print("✅ 数据库连接已关闭")

# ==================== 知识图谱构建器 ====================

class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687"):
        self.processor = MedicalTextProcessor()
        self.neo4j = Neo4jManager(neo4j_uri)
        self.entities = []
        self.relations = []
    
    def build_from_text(self, text: str):
        """从文本构建知识图谱"""
        print("🔍 正在处理文本，提取实体和关系...")
        
        # 提取实体和关系
        self.entities, self.relations = self.processor.process_text(text)
        
        print(f"✅ 提取完成: 找到 {len(self.entities)} 个实体, {len(self.relations)} 个关系")
        
        # 显示统计信息
        self._show_statistics()
        
        return self.entities, self.relations
    
    def save_to_neo4j(self):
        """保存到Neo4j数据库"""
        print("💾 正在保存到Neo4j数据库...")
        
        # 连接数据库
        if not self.neo4j.connect():
            print("❌ 无法连接到Neo4j，请确保Neo4j服务正在运行")
            print("💡 启动Neo4j: neo4j start (命令行)")
            return False
        
        # 创建schema
        self.neo4j.create_schema()
        
        # 保存实体
        entity_count = 0
        for entity in self.entities:
            if self.neo4j.create_entity_node(entity):
                entity_count += 1
        
        # 保存关系
        relation_count = 0
        for relation in self.relations:
            if self.neo4j.create_relation(relation):
                relation_count += 1
        
        print(f"✅ 保存完成: {entity_count} 个实体, {relation_count} 个关系")
        
        return True
    
    def _show_statistics(self):
        """显示统计信息"""
        print("\n" + "="*50)
        print("📊 提取统计信息:")
        print("="*50)
        
        # 实体类型统计
        entity_stats = defaultdict(int)
        for entity in self.entities:
            entity_stats[entity.entity_type.value] += 1
        
        print("\n🔷 实体统计:")
        for entity_type, count in sorted(entity_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {entity_type}: {count}")
        
        # 关系类型统计
        relation_stats = defaultdict(int)
        for relation in self.relations:
            relation_stats[relation.relation_type.value] += 1
        
        print("\n🔗 关系统计:")
        for relation_type, count in sorted(relation_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {relation_type}: {count}")
        
        # 显示前10个实体
        print("\n🏷️  前10个实体:")
        for i, entity in enumerate(self.entities[:10]):
            print(f"  {i+1}. {entity.name} ({entity.entity_type.value})")
        
        # 显示前10个关系
        print("\n🔗 前10个关系:")
        for i, relation in enumerate(self.relations[:10]):
            print(f"  {i+1}. {relation.source.name} --[{relation.relation_type.value}]--> {relation.target.name}")
        
        print("="*50)
    
    def export_to_json(self, filename: str = "knowledge_graph.json"):
        """导出到JSON文件"""
        data = {
            'entities': [
                {
                    'name': entity.name,
                    'type': entity.entity_type.value,
                    'attributes': entity.attributes
                }
                for entity in self.entities
            ],
            'relations': [
                {
                    'source': relation.source.name,
                    'target': relation.target.name,
                    'type': relation.relation_type.value,
                    'attributes': relation.attributes
                }
                for relation in self.relations
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已导出到 {filename}")
        return filename
    
    def visualize_with_networkx(self):
        """使用networkx进行可视化（本地展示）"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图
            G = nx.DiGraph()
            
            # 添加节点
            for entity in self.entities:
                G.add_node(entity.name, type=entity.entity_type.value)
            
            # 添加边
            for relation in self.relations:
                G.add_edge(relation.source.name, relation.target.name, 
                          type=relation.relation_type.value)
            
            # 设置节点颜色
            node_colors = []
            for node in G.nodes():
                node_type = G.nodes[node]['type']
                color_map = {
                    '基因': '#FF6B6B',
                    '疾病': '#4ECDC4',
                    '症状': '#FFD166',
                    '药物': '#06D6A0',
                    '治疗': '#118AB2',
                    '染色体': '#EF476F',
                    '蛋白质': '#073B4C',
                    '通路': '#7209B7'
                }
                node_colors.append(color_map.get(node_type, '#888888'))
            
            # 绘制图
            plt.figure(figsize=(16, 12))
            
            # 使用spring布局
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 node_size=800, alpha=0.8)
            
            # 绘制边
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                 arrows=True, arrowsize=20, alpha=0.6)
            
            # 绘制标签
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            # 绘制边标签
            edge_labels = nx.get_edge_attributes(G, 'type')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            
            plt.title("医学知识图谱可视化", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#FF6B6B', label='基因'),
                Patch(facecolor='#4ECDC4', label='疾病'),
                Patch(facecolor='#FFD166', label='症状'),
                Patch(facecolor='#06D6A0', label='药物'),
                Patch(facecolor='#118AB2', label='治疗'),
                Patch(facecolor='#EF476F', label='染色体'),
                Patch(facecolor='#888888', label='其他')
            ]
            plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            plt.savefig('knowledge_graph.png', dpi=300, bbox_inches='tight')
            print("✅ 已保存可视化图像: knowledge_graph.png")
            plt.show()
            
        except ImportError as e:
            print(f"❌ 缺少可视化库: {e}")
            print("💡 请安装: pip install networkx matplotlib")
        except Exception as e:
            print(f"❌ 可视化失败: {e}")

# ==================== 主程序 ====================

def main():
    """主函数"""
    print("="*60)
    print("🧬 医学知识图谱构建与可视化系统")
    print("="*60)
    
    # 示例文本（在实际使用中，您可以从文件读取）
    with open('medical_text.txt', 'r', encoding='utf-8') as f:
        medical_text = f.read()
    
    # 或者使用提供的文本
    if not medical_text:
        medical_text = """
        基因、症状与药物响应的核心医学关联解析

        现代医学研究证实，基因变异是调控疾病发生发展、症状表现及药物治疗效果的关键内在因素。人体基因组的微小差异，不仅决定了个体对疾病的易感性和症状异质性，更直接影响药物在体内的代谢效率、疗效发挥及不良反应风险。本文聚焦临床高发的肿瘤、心血管疾病及神经代谢性疾病，系统梳理核心关联基因、典型临床症状、针对性药物及疗效机制，构建精准医疗视角下的基因-症状-药物关联体系。

        一、肿瘤领域：驱动基因导向的靶向治疗

        肿瘤的发生本质是遗传物质异常累积的结果，特征性基因突变、融合等变异通过调控细胞增殖、凋亡通路推动肿瘤进展，同时决定了临床症状谱和靶向药物敏感性。其中非小细胞肺癌和乳腺癌的基因靶向治疗是精准医疗的典范。

        （一）非小细胞肺癌：EGFR/ALK基因与靶向干预

        核心关联基因方面，非小细胞肺癌（NSCLC）占肺癌总数的80%-85%，表皮生长因子受体（EGFR）基因突变和间变性淋巴瘤激酶（ALK）融合基因是最主要的驱动基因。EGFR基因位于7号染色体，19外显子缺失、21外显子L858R点突变为经典敏感突变，亚裔人群发生率达30%-50%；ALK融合（如EML4-ALK）多见于年轻不吸烟肺腺癌患者，发生率5%-7%，通过持续激活PI3K/AKT通路促进肿瘤增殖。

        临床症状上，EGFR突变型患者早期常表现为刺激性干咳、少量咯血、活动后胸闷，进展后可出现胸痛、胸腔积液所致呼吸困难，以及骨转移骨痛、脑转移头痛呕吐等症状。ALK融合型患者除上述共性症状外，更易出现纵隔淋巴结肿大和早期远处转移，部分以不明原因发热为首发症状，且对常规化疗敏感性较低。

        针对性药物及疗效显著，EGFR敏感突变的靶向药物为酪氨酸激酶抑制剂（TKI），已发展至第三代。第一代厄洛替尼、吉非替尼使晚期患者中位生存期从化疗的12个月延长至18-20个月，客观缓解率（ORR）50%-70%，但易出现T790M耐药；第二代阿法替尼、达克替尼通过不可逆结合靶点延缓耐药，中位生存期达20-24个月，仅不良反应略升高；第三代奥希替尼可特异性针对T790M突变，安全性更优，中位生存期突破30个月，对脑转移疗效突出，成为一线优选。ALK融合患者可选克唑替尼（第一代，ORR 60%-70%，中位PFS 10个月）、阿来替尼（第二代，中位PFS 34.8个月，脑穿透性强）及劳拉替尼（第三代，覆盖耐药突变，ORR 40%-60%）。

        （二）乳腺癌：HER2基因靶向治疗体系

        人表皮生长因子受体2（HER2/neu）基因位于17号染色体，其扩增或过度表达是乳腺癌重要驱动因素，发生率15%-20%，导致肿瘤细胞增殖侵袭能力增强，患者易发生淋巴转移、预后差。此外，P53基因突变与肿瘤恶性程度升高相关，ER/PR基因状态决定内分泌治疗响应。

        临床症状以乳房无痛性肿块为早期典型表现，质地硬、边界不清、活动度差，部分伴随乳头溢液、内陷或乳房皮肤"橘皮样"改变。晚期易发生肺、肝、骨、脑转移，出现咳嗽、肝区疼痛、骨痛等转移症状。三阴性乳腺癌（ER/PR、HER2均阴性）症状更具侵袭性，复发转移风险高。

        靶向药物以曲妥珠单抗为核心，作为人源化单克隆抗体，通过结合HER2受体抑制信号通路并介导细胞毒性作用，联合化疗使早期患者复发风险降低30%-50%，晚期ORR 40%-60%。针对耐药患者，帕妥珠单抗可与曲妥珠单抗协同作用，使晚期中位PFS延长至18.5个月；T-DM1（抗体药物偶联物）对耐药患者ORR达30%-40%；口服小分子TKI吡咯替尼联合卡培他滨ORR可达78.5%，为晚期患者提供便捷治疗方案。

        二、心血管系统疾病：基因调控的个体化用药

        心血管疾病的发生发展与遗传因素密切相关，基因变异通过影响心肌离子通道功能、药物代谢酶活性，导致疾病易感性增加和药物响应差异。Brugada综合征及华法林抗凝治疗的基因调控机制研究较为成熟。

        （一）Brugada综合征：SCN5A基因与心律失常干预

        Brugada综合征为遗传性离子通道病，主要由SCN5A基因突变引起，该基因位于3号染色体，编码心肌钠通道，突变导致钠通道功能丧失，引发心肌复极异常，特征性心电图改变（V1-V3导联J波增大、ST段抬高）。疾病多见于东南亚人群，男性占比85%。

        部分患者无症状，仅体检发现心电图异常；有症状者核心表现为多形性室性心动过速、心室颤动，进而导致晕厥或夜间猝死，与运动无关。发热或使用钠通道阻滞剂、部分抗抑郁药可诱发症状加重，约10%患者伴随心房颤动。

        治疗以预防猝死为核心，植入心脏复律除颤器（ICD）是最有效手段。药物治疗以奎尼丁为主要辅助药物，通过抑制早期外向钾电流纠正复极异常，减少恶性心律失常发生。需严格避免氟卡尼、普罗帕酮等钠通道阻滞剂，无症状患者需结合电生理检查进行风险分层。

        （二）华法林抗凝：VKORC1/CYP2C9基因剂量调控

        华法林是临床常用口服抗凝药，用于预防血栓栓塞性疾病，但其剂量需求个体差异大，主要由VKORC1和CYP2C9基因变异决定。VKORC1基因编码维生素K环氧还原酶（华法林作用靶点），-1639G>A突变使酶表达降低，患者对华法林敏感性增加；CYP2C9基因编码代谢酶，*2、*3突变降低酶活性，导致药物蓄积、出血风险升高。

        药物相关症状具有双向性：剂量不足时无法抑制血栓，可引发缺血性脑卒中（肢体偏瘫、言语不清）、深静脉血栓（肢体肿胀疼痛）、肺栓塞（呼吸困难、胸痛）；剂量过高时出血风险增加，表现为皮肤瘀斑、牙龈出血、胃肠道出血（黑便、呕血），严重时颅内出血（头痛、意识障碍）危及生命。

        基因检测可精准指导剂量调整：VKORC1-1639AA基因型联合CYP2C9*2/*3突变患者，初始剂量需降至常规的1/3-1/2；VKORC1-1639GG联合CYP2C9*1/*1野生型患者需较高剂量。临床数据显示，基因指导用药可使出血事件发生率从6.7%降至4.0%，缩短达目标INR（2.0-3.0）时间40%，目前全球超80国将相关基因检测纳入医保。

        三、神经代谢性疾病：基因缺陷与对症干预

        神经代谢性疾病多由基因缺陷导致酶活性异常，引发代谢产物蓄积或缺乏，表现特征性症状。抑郁症的药物代谢基因调控及Lesch-Nyhan综合征的基因缺陷干预具有代表性。

        （一）抑郁症：CYP450基因与抗抑郁药响应

        抑郁症发病与血清素、多巴胺神经递质失衡相关，药物疗效受CYP2D6和CYP2C19基因调控。CYP2D6基因编码酶代谢舍曲林、帕罗西汀等SSRI类药物，变异分为快、中、慢代谢型；CYP2C19基因代谢西酞普兰等药物，*2、*3突变导致慢代谢。此外，SLC6A4基因多态性影响SSRI疗效，长等位基因携带者响应率更高。

        核心症状为情绪低落、兴趣减退、快感缺失，伴随思维迟缓、注意力不集中、自责自罪、睡眠障碍（失眠/嗜睡）、食欲体重改变及疲劳乏力，部分患者存在自杀观念，症状个体差异与遗传、环境相关。

        SSRI类为一线药物，通过抑制5-羟色胺再摄取发挥作用。CYP2D6慢代谢型患者服用帕罗西汀时，药物蓄积导致恶心、头晕等不良反应升高，需减量；快代谢型患者可能因药物快速代谢疗效不佳，需增量或换药。基因检测可明确代谢表型，如CYP2C19慢代谢型患者选择舍曲林等不经该酶代谢药物，可使治疗响应率提高30%-40%，缩短治疗周期。

        （二）Lesch-Nyhan综合征：HPRT1基因缺陷与对症治疗

        Lesch-Nyhan综合征为X连锁隐性遗传代谢病，由X染色体上HPRT1基因缺陷引起，该基因编码次黄嘌呤-鸟嘌呤磷酸核糖转移酶，参与嘌呤补救合成，缺陷导致酶活性丧失，尿酸生成过多并影响神经发育。

        典型三联征为高尿酸血症、神经系统障碍及自伤行为。患儿出生后尿布可见橙色尿酸盐结晶，随年龄增长出现高尿酸尿症、痛风性关节炎、尿酸性肾病变；神经系统症状包括运动迟缓、肌张力异常、反射亢进、舞蹈徐动症、构音障碍及智力发育障碍；自伤行为多见于3-5岁，表现为咬唇、咬手指、撞头等，严重致肢体损伤。

        目前尚无根治方法，治疗以对症支持为主。别嘌醇通过抑制黄嘌呤氧化酶减少尿酸生成，缓解高尿酸相关症状；苯二氮卓类药物（如地西泮）可缓解肌张力增高和躁动，但疗效有限。自伤行为需依赖防护措施，康复治疗可改善运动和生活能力。基因治疗为研究热点，通过病毒载体导入正常HPRT1基因，目前处于实验阶段。

        四、总结

        基因作为核心调控要素，贯穿疾病发生、症状表现及药物响应全过程。从肿瘤驱动基因靶向治疗，到心血管疾病药物剂量精准调控，再到神经代谢病的基因缺陷干预，基因检测已成为优化临床诊疗的关键工具。未来，随着高通量测序和基因编辑技术的发展，精准医疗将进一步实现疾病的早期预测、个体化治疗及根治性干预，为患者提供更高效、安全的医疗服务。同时，基因信息解读需结合临床症状、病史等多因素，兼顾伦理安全，推动医学事业向个性化、精准化方向迈进。
        """
    
    # 创建知识图谱构建器
    print("1. 初始化知识图谱构建器...")
    kg_builder = KnowledgeGraphBuilder()
    
    # 构建知识图谱
    print("\n2. 从文本构建知识图谱...")
    entities, relations = kg_builder.build_from_text(medical_text)
    
    # 导出到JSON
    print("\n3. 导出到JSON文件...")
    kg_builder.export_to_json()
    
    # 保存到Neo4j
    print("\n4. 保存到Neo4j数据库...")
    if kg_builder.save_to_neo4j():
        print("   Neo4j数据库操作:")
        print("   - 在浏览器中打开: http://localhost:7474")
        print("   - 用户名: neo4j")
        print("   - 密码: password")
        print("   - 运行查询: MATCH (n) RETURN n LIMIT 25")
    
    # 本地可视化
    print("\n5. 生成可视化图表...")
    kg_builder.visualize_with_networkx()
    
    # 查询示例
    print("\n6. 示例查询:")
    if kg_builder.neo4j.driver:
        print("   在Neo4j浏览器中运行以下查询:")
        print("   a) 查找所有基因:")
        print('      MATCH (g:Entity {type: "基因"}) RETURN g.name LIMIT 10')
        print("\n   b) 查找EGFR相关的所有关系:")
        print('      MATCH (source:Entity {name: "EGFR"})-[r]->(target)')
        print('      RETURN source.name, r.type, target.name')
        print("\n   c) 查找治疗非小细胞肺癌的药物:")
        print('      MATCH (drug:Entity {type: "药物"})-[r:RELATIONSHIP {type: "治疗"}]->(disease:Entity {name: "非小细胞肺癌"})')
        print('      RETURN drug.name')
        print("\n   d) 可视化整个知识图谱:")
        print('      MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50')
    
    # 关闭连接
    if kg_builder.neo4j.driver:
        kg_builder.neo4j.close()
    
    print("\n" + "="*60)
    print("🎉 知识图谱构建完成!")
    print("="*60)
    print("\n📁 生成的文件:")
    print("   - medical_kg_builder.py: 主程序")
    print("   - knowledge_graph.json: 知识图谱数据")
    print("   - knowledge_graph.png: 可视化图像")
    print("\n💡 下一步:")
    print("   1. 启动Neo4j: neo4j start")
    print("   2. 访问Neo4j浏览器: http://localhost:7474")
    print("   3. 修改代码中的数据库连接信息（如果需要）")
    print("   4. 运行程序: python medical_kg_builder.py")

if __name__ == "__main__":
    main()
