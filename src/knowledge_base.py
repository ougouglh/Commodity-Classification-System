"""
知识库构建模块
从CSV数据构建品类定义知识库
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class CategoryKnowledgeBase:
    """品类定义知识库"""

    def __init__(self, csv_path: Optional[str] = None):
        """
        初始化知识库
        Args:
            csv_path: 品类定义CSV文件路径，默认使用配置中的路径
        """
        self.csv_path = csv_path or (config.DATA_DIR / '品类定义详情.csv')
        self.df = None
        self.category_hierarchy = {}
        self.category_definitions = {}

    def load_data(self) -> pd.DataFrame:
        """加载品类定义数据"""
        logger.info(f"Loading data from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
        self.df = self.df.fillna('')

        relevant_columns = [
            '序号', '一级类目', '二级类目', '三级类目', '四级类目',
            '类目等级', '定义', '包括', '不包括',
            '代表品牌', '分类', '功能', '口味', '包装类型',
            '是否含糖', '产品类型', '产地', '备注'
        ]

        available_cols = [col for col in relevant_columns if col in self.df.columns]
        self.raw_data = self.df[available_cols].copy()

        logger.info(f"Loaded {len(self.raw_data)} records with columns: {', '.join(available_cols)}")
        return self.raw_data

    def build_hierarchy(self) -> Dict:
        """构建分类层级结构"""
        if self.raw_data is None:
            self.load_data()

        hierarchy = {}
        for _, row in self.raw_data.iterrows():
            level_1 = row.get('一级类目', '')
            level_2 = row.get('二级类目', '')
            level_3 = row.get('三级类目', '')
            level_4 = row.get('四级类目', '')

            if level_1 and level_1 not in hierarchy:
                hierarchy[level_1] = {}
            if level_2 and level_2 not in hierarchy.get(level_1, {}):
                hierarchy[level_1][level_2] = {}
            if level_3 and level_3 not in hierarchy.get(level_1, {}).get(level_2, {}):
                hierarchy[level_1][level_2][level_3] = []
            if level_4 and level_4 not in hierarchy.get(level_1, {}).get(level_2, {}).get(level_3, []):
                hierarchy[level_1][level_2][level_3].append(level_4)

        self.category_hierarchy = hierarchy
        logger.info(f"Built hierarchy with {len(hierarchy)} level-1 categories")
        return hierarchy

    def get_category_definitions(self) -> Dict[str, Dict]:
        """获取所有分类的详细定义"""
        if self.raw_data is None:
            self.load_data()

        df = self.raw_data.copy()

        # 前向填充分类层级
        level_columns = ['序号', '一级类目', '二级类目', '三级类目', '四级类目']
        for col in level_columns:
            if col in df.columns:
                df[col] = df[col].replace('', pd.NA).ffill()

        categories = {}

        for _, row in df.iterrows():
            level_4 = str(row.get('四级类目', ''))
            if not level_4 or level_4 == 'nan':
                continue

            if level_4 not in categories:
                categories[level_4] = {
                    '基本信息': {
                        '序号': str(row.get('序号', '')),
                        '一级类目': row.get('一级类目', ''),
                        '二级类目': row.get('二级类目', ''),
                        '三级类目': row.get('三级类目', ''),
                        '四级类目': level_4,
                        '类目等级': row.get('类目等级', ''),
                    },
                    '定义信息': {
                        '定义': row.get('定义', ''),
                        '包括': row.get('包括', ''),
                        '不包括': row.get('不包括', ''),
                        '备注': row.get('备注', ''),
                    },
                    '特征属性': {
                        '分类': row.get('分类', ''),
                        '功能': row.get('功能', ''),
                        '口味': row.get('口味', ''),
                        '包装类型': row.get('包装类型', ''),
                        '是否含糖': row.get('是否含糖', ''),
                        '产品类型': row.get('产品类型', ''),
                        '产地': row.get('产地', ''),
                    },
                    '代表品牌': [],
                    '全部口味': [],
                    '全部包装': []
                }

            brand = row.get('代表品牌', '').strip()
            if brand and brand not in categories[level_4]['代表品牌']:
                categories[level_4]['代表品牌'].append(brand)

            flavor = row.get('口味', '').strip()
            if flavor and flavor not in categories[level_4]['全部口味']:
                categories[level_4]['全部口味'].append(flavor)

            packaging = row.get('包装类型', '').strip()
            if packaging and packaging not in categories[level_4]['全部包装']:
                categories[level_4]['全部包装'].append(packaging)

        self.category_definitions = categories
        logger.info(f"Extracted {len(categories)} level-4 category definitions")
        return categories

    def create_knowledge_documents(self) -> List[Dict]:
        """创建知识库文档，每个四级分类一个文档"""
        if not self.category_definitions:
            self.get_category_definitions()

        documents = []

        for category, info in self.category_definitions.items():
            basic = info['基本信息']
            definition = info['定义信息']
            features = info['特征属性']

            brands_str = '、'.join(info['代表品牌']) if info['代表品牌'] else ''
            flavors_str = '、'.join(info['全部口味']) if info['全部口味'] else ''
            packaging_str = '、'.join(info['全部包装']) if info['全部包装'] else ''

            doc = {
                'category_id': hashlib.md5(category.encode()).hexdigest()[:8],
                'category_name': category,
                'full_path': f"{basic['一级类目']} > {basic['二级类目']} > {basic['三级类目']} > {category}",
                'text': self._build_category_text(info),
                'metadata': {
                    'level_1': basic['一级类目'],
                    'level_2': basic['二级类目'],
                    'level_3': basic['三级类目'],
                    'level_4': category,
                    'category_level': basic['类目等级'],

                    'definition': definition['定义'],
                    'includes': definition['包括'],
                    'excludes': definition['不包括'],
                    'notes': definition['备注'],

                    'category_type': features['分类'],
                    'function': features['功能'],
                    'primary_flavor': features['口味'],
                    'all_flavors': flavors_str,
                    'primary_packaging': features['包装类型'],
                    'all_packagings': packaging_str,
                    'sugar_content': features['是否含糖'],
                    'product_type': features['产品类型'],
                    'origin': features['产地'],

                    'brands': brands_str,
                    'brand_count': len(info['代表品牌']),
                    'flavor_count': len(info['全部口味']),
                    'packaging_count': len(info['全部包装'])
                }
            }
            documents.append(doc)

        return documents

    def _build_category_text(self, info: Dict) -> str:
        """构建分类文本描述"""
        parts = []

        basic = info['基本信息']
        definition = info['定义信息']
        features = info['特征属性']

        parts.append(f"【分类信息】")
        parts.append(f"分类名称：{basic['四级类目']}")
        parts.append(f"分类路径：{basic['一级类目']} > {basic['二级类目']} > {basic['三级类目']} > {basic['四级类目']}")
        parts.append(f"类目等级：{basic['类目等级']}")

        if definition['定义']:
            parts.append(f"\n【定义】{definition['定义']}")

        if definition['包括']:
            parts.append(f"\n【包括范围】{definition['包括']}")

        if definition['不包括']:
            parts.append(f"\n【不包括范围】{definition['不包括']}")

        if features['功能']:
            parts.append(f"\n【功能特征】{features['功能']}")

        if info['代表品牌']:
            parts.append(f"\n【代表品牌】{'、'.join(info['代表品牌'])}")

        if info['全部口味']:
            parts.append(f"\n【口味类型】{'、'.join(info['全部口味'])}")

        if info['全部包装']:
            parts.append(f"\n【包装类型】{'、'.join(info['全部包装'])}")

        if features['是否含糖']:
            parts.append(f"\n【含糖属性】{features['是否含糖']}")

        if features['产品类型']:
            parts.append(f"\n【产品类型】{features['产品类型']}")

        if features['产地']:
            parts.append(f"\n【产地信息】{features['产地']}")

        if definition['备注']:
            parts.append(f"\n【备注】{definition['备注']}")

        return '\n'.join(parts)

    def create_brand_index(self) -> Dict[str, List[Dict]]:
        """创建品牌索引"""
        if not self.category_definitions:
            self.get_category_definitions()

        brand_index = {}

        for category, info in self.category_definitions.items():
            for brand in info['代表品牌']:
                brand_lower = brand.lower()
                if brand_lower not in brand_index:
                    brand_index[brand_lower] = []

                brand_index[brand_lower].append({
                    'category': category,
                    'full_path': f"{info['基本信息']['一级类目']} > {info['基本信息']['二级类目']} > {info['基本信息']['三级类目']} > {category}",
                    'brands': info['代表品牌'],
                    'flavors': info['全部口味'],
                    'packaging': info['全部包装']
                })

        logger.info(f"Created brand index with {len(brand_index)} brands")
        return brand_index

    def create_flavor_index(self) -> Dict[str, List[Dict]]:
        """创建口味索引"""
        if not self.category_definitions:
            self.get_category_definitions()

        flavor_index = {}

        for category, info in self.category_definitions.items():
            for flavor in info['全部口味']:
                flavor_lower = flavor.lower()
                if flavor_lower not in flavor_index:
                    flavor_index[flavor_lower] = []

                flavor_index[flavor_lower].append({
                    'category': category,
                    'full_path': f"{info['基本信息']['一级类目']} > {info['基本信息']['二级类目']} > {info['基本信息']['三级类目']} > {category}",
                    'flavor': flavor,
                    'brands': info['代表品牌'],
                    'packaging': info['全部包装']
                })

        logger.info(f"Created flavor index with {len(flavor_index)} flavors")
        return flavor_index

    def save_knowledge_base(self, output_dir: Optional[str] = None):
        """保存知识库到文件"""
        output_path = Path(output_dir or config.KNOWLEDGE_BASE_DIR)
        output_path.mkdir(parents=True, exist_ok=True)

        documents = self.create_knowledge_documents()

        with open(output_path / 'knowledge_documents.json', 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        with open(output_path / 'category_hierarchy.json', 'w', encoding='utf-8') as f:
            json.dump(self.category_hierarchy, f, ensure_ascii=False, indent=2)

        brand_index = self.create_brand_index()
        with open(output_path / 'brand_index.json', 'w', encoding='utf-8') as f:
            json.dump(brand_index, f, ensure_ascii=False, indent=2)

        flavor_index = self.create_flavor_index()
        with open(output_path / 'flavor_index.json', 'w', encoding='utf-8') as f:
            json.dump(flavor_index, f, ensure_ascii=False, indent=2)

        stats = {
            'total_categories': len(documents),
            'total_brands': len(brand_index),
            'total_flavors': len(flavor_index),
            'level_1_count': len(self.category_hierarchy),
            'documents_with_brands': sum(1 for d in documents if d['metadata']['brands']),
            'documents_with_flavors': sum(1 for d in documents if d['metadata']['all_flavors']),
            'documents_with_packaging': sum(1 for d in documents if d['metadata']['all_packagings']),
        }

        with open(output_path / 'statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Knowledge base saved to {output_dir}")
        logger.info(f"  - Documents: {stats['total_categories']}")
        logger.info(f"  - Brands: {stats['total_brands']}")
        logger.info(f"  - Flavors: {stats['total_flavors']}")

        return stats
