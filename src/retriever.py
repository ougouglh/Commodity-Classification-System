"""
智能检索模块
结合BGE语义检索和BM75词频匹配的混合检索策略
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import config
from src.logger import get_logger
from src.vector_store import BGEVectorStore

logger = get_logger(__name__)


class HybridRetriever:
    """混合检索器：BGE语义检索 + BM25词频匹配 + 品牌加权"""

    def __init__(self, vector_store: BGEVectorStore, brand_index_path: Optional[str] = None):
        """
        初始化混合检索器
        Args:
            vector_store: BGE向量存储实例
            brand_index_path: 品牌索引路径（默认从知识库路径加载）
        """
        self.vector_store = vector_store
        self.brand_index = {}
        self.bm25_index = None
        self.tokenized_docs = []

        # 加载品牌索引
        if brand_index_path:
            self._load_brand_index(brand_index_path)
        elif vector_store.brand_index:
            self.brand_index = vector_store.brand_index
            logger.info(f"Loaded brand index from vector store: {len(self.brand_index)} brands")

        # 构建BM25索引
        self._build_bm25_index()

        self.config = {
            'top_k': config.RETRIEVAL_CANDIDATE_K,
            'brand_boost': config.BRAND_BOOST_SCORE,
            'return_top_k': config.RETRIEVAL_TOP_K,
            'alpha': config.HYBRID_ALPHA,  # 向量检索权重
        }

    def _load_brand_index(self, path: str):
        """加载品牌索引"""
        with open(path, 'r', encoding='utf-8') as f:
            self.brand_index = json.load(f)
        logger.info(f"Loaded brand index: {len(self.brand_index)} brands")

    def _build_bm25_index(self):
        """构建BM25索引"""
        logger.info("Building BM25 index...")

        self.tokenized_docs = []
        for doc in self.vector_store.documents:
            # 简单分词：按空格和中文标点分割
            text = doc.get('text', '')
            tokens = self._tokenize(text)
            self.tokenized_docs.append(tokens)

        self.bm25_index = BM25Okapi(self.tokenized_docs)
        logger.info(f"BM25 index built with {len(self.tokenized_docs)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """简单的中文分词"""
        import re
        # 按中文字符、英文单词、数字分割
        tokens = re.findall(r'[一-鿿]+|[a-zA-Z0-9]+', text.lower())
        return tokens

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
        """混合检索"""
        retrieval_top_k = self.config['top_k']
        brand_boost = self.config['brand_boost']
        return_k = top_k or self.config['return_top_k']
        alpha = self.config['alpha']

        # 1. BGE语义检索
        vector_results = self.vector_store.retrieve(product_name, top_k=retrieval_top_k)

        # 2. BM25词频检索
        query_tokens = self._tokenize(product_name)
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # 构建BM25结果列表
        bm25_results = []
        for idx, score in enumerate(bm25_scores):
            if score > 0:  # 只保留有匹配的
                bm25_results.append({
                    'index': idx,
                    'bm25_score': score
                })

        # 3. 融合两种检索结果
        # 创建文档索引到分数的映射
        vector_scores = {r['category_name']: r.get('similarity', 0) for r in vector_results}

        fused_results = []
        for doc in self.vector_store.documents:
            category_name = doc.get('category_name', '')
            doc_index = doc.get('category_name', '')

            # 获取向量分数
            vector_score = vector_scores.get(category_name, 0)

            # 获取BM25分数
            bm25_score = 0
            for bm25_r in bm25_results:
                if self.vector_store.documents[bm25_r['index']].get('category_name') == category_name:
                    bm25_score = bm25_r['bm25_score']
                    break

            # 如果两种检索都没有匹配，跳过
            if vector_score == 0 and bm25_score == 0:
                continue

            # 归一化分数（简单最大值归一化）
            max_vector = max([r.get('similarity', 0) for r in vector_results]) if vector_results else 1
            max_bm25 = max([r['bm25_score'] for r in bm25_results]) if bm25_results else 1

            norm_vector = vector_score / max_vector if max_vector > 0 else 0
            norm_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0

            # 加权融合
            fused_score = alpha * norm_vector + (1 - alpha) * norm_bm25

            result = doc.copy()
            result['similarity'] = fused_score
            result['vector_score'] = vector_score
            result['bm25_score'] = bm25_score

            fused_results.append(result)

        # 4. 品牌加权
        matched_brands = self.extract_brands_from_name(product_name)

        if matched_brands:
            brand_categories = self.get_categories_by_brands(matched_brands)

            # 对结果加权
            for result in fused_results:
                if result['category_name'] in brand_categories:
                    result['similarity'] += brand_boost
                    result['brand_match'] = matched_brands

            logger.debug(f"Brand matches: {matched_brands}, boosted categories")

        # 5. 重新排序并返回Top-K
        fused_results.sort(key=lambda x: x['similarity'], reverse=True)
        return fused_results[:return_k]

    def set_config(self, top_k: int = None, brand_boost: float = None,
                   return_top_k: int = None, alpha: float = None):
        """修改配置"""
        if top_k is not None:
            self.config['top_k'] = top_k
        if brand_boost is not None:
            self.config['brand_boost'] = brand_boost
        if return_top_k is not None:
            self.config['return_top_k'] = return_top_k
        if alpha is not None:
            self.config['alpha'] = alpha
        logger.info(f"Config updated: {self.config}")


# 保持向后兼容
SmartRetriever = HybridRetriever
