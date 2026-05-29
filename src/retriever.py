"""
智能检索模块
结合BGE语义检索和品牌加权
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from src.config import config
from src.logger import get_logger
from src.vector_store import BGEVectorStore, CategoryRetriever

logger = get_logger(__name__)


class SmartRetriever:
    """智能检索器：BGE语义检索 + 品牌加权"""

    def __init__(self, vector_store: BGEVectorStore, brand_index_path: Optional[str] = None):
        """
        初始化智能检索器
        Args:
            vector_store: BGE向量存储实例
            brand_index_path: 品牌索引路径（默认从知识库路径加载）
        """
        self.vector_store = vector_store
        self.brand_index = {}

        # 加载品牌索引
        if brand_index_path:
            self._load_brand_index(brand_index_path)
        elif vector_store.brand_index:
            self.brand_index = vector_store.brand_index
            logger.info(f"Loaded brand index from vector store: {len(self.brand_index)} brands")

        self.config = {
            'top_k': config.RETRIEVAL_CANDIDATE_K,
            'brand_boost': config.BRAND_BOOST_SCORE,
            'return_top_k': config.RETRIEVAL_TOP_K,
        }

    def _load_brand_index(self, path: str):
        """加载品牌索引"""
        with open(path, 'r', encoding='utf-8') as f:
            self.brand_index = json.load(f)
        logger.info(f"Loaded brand index: {len(self.brand_index)} brands")

    def extract_brands_from_name(self, product_name: str) -> List[str]:
        """从商品名称中提取已知品牌"""
        matched_brands = []
        product_lower = product_name.lower()

        for brand in self.brand_index.keys():
            if brand in product_lower:
                matched_brands.append(brand)

        return matched_brands

    def get_categories_by_brands(self, brands: List[str]) -> set:
        """根据品牌获取关联的分类"""
        categories = set()

        for brand in brands:
            if brand in self.brand_index:
                for cat_info in self.brand_index[brand]:
                    categories.add(cat_info['category'])

        return categories

    def retrieve(self, product_name: str, top_k: int = None) -> List[Dict]:
        """智能检索"""
        retrieval_top_k = self.config['top_k']
        brand_boost = self.config['brand_boost']
        return_k = top_k or self.config['return_top_k']

        # 1. BGE语义检索
        results = self.vector_store.retrieve(product_name, top_k=retrieval_top_k)

        # 2. 品牌加权
        matched_brands = self.extract_brands_from_name(product_name)

        if matched_brands:
            brand_categories = self.get_categories_by_brands(matched_brands)

            # 对结果加权
            for result in results:
                if result['category_name'] in brand_categories:
                    result['similarity'] += brand_boost
                    result['brand_match'] = matched_brands

            logger.debug(f"Brand matches: {matched_brands}, boosted categories")

        # 3. 重新排序
        results.sort(key=lambda x: x['similarity'], reverse=True)

        # 4. 返回Top-K
        return results[:return_k]

    def set_config(self, top_k: int = None, brand_boost: float = None, return_top_k: int = None):
        """修改配置"""
        if top_k is not None:
            self.config['top_k'] = top_k
        if brand_boost is not None:
            self.config['brand_boost'] = brand_boost
        if return_top_k is not None:
            self.config['return_top_k'] = return_top_k
        logger.info(f"Config updated: {self.config}")
