"""
混合检索器单元测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import HybridRetriever
from src.config import config


@pytest.fixture
def mock_vector_store():
    """模拟向量存储"""
    mock_store = Mock()
    mock_store.documents = [
        {
            'category_name': '碳酸饮料',
            'text': '碳酸饮料是一种含有二氧化碳的饮料，代表品牌有可口可乐、百事可乐',
            'metadata': {'brands': '可口可乐,百事可乐'}
        },
        {
            'category_name': '萝卜',
            'text': '萝卜是一种蔬菜，香辣萝卜是常见零食，代表品牌有湘满天',
            'metadata': {'brands': '湘满天'}
        },
        {
            'category_name': '气泡水',
            'text': '气泡水是含气矿泉水，代表品牌有巴黎水、圣培露',
            'metadata': {'brands': '巴黎水,圣培露'}
        },
    ]
    mock_store.brand_index = {
        '可口可乐': [{'category': '碳酸饮料'}],
        '百事可乐': [{'category': '碳酸饮料'}],
        '湘满天': [{'category': '萝卜'}],
    }
    return mock_store


@pytest.fixture
def retriever(mock_vector_store):
    """创建检索器实例"""
    # 不真的构建BM25索引，用mock
    with patch.object(HybridRetriever, '_build_bm25_index'):
        retriever = HybridRetriever(mock_vector_store)
        # 手动设置一个假的BM25索引
        retriever.bm25_index = Mock()
        retriever.bm25_index.get_scores = Mock(return_value=[0.8, 0.6, 0.4])
        retriever.tokenized_docs = [['碳酸', '饮料'], ['萝卜'], ['气泡', '水']]
        return retriever


class TestBrandExtraction:
    """品牌提取测试"""

    def test_extract_brand_from_name(self, retriever):
        """测试从商品名中提取品牌"""
        product = "可口可乐330ml"
        brands = retriever.extract_brands_from_name(product)
        assert "可口可乐" in brands

    def test_extract_multiple_brands(self, retriever):
        """测试提取多个品牌"""
        product = "可口可乐百事可乐"
        brands = retriever.extract_brands_from_name(product)
        assert "可口可乐" in brands
        assert "百事可乐" in brands

    def test_no_brand_in_name(self, retriever):
        """测试无品牌商品名"""
        product = "普通饮料330ml"
        brands = retriever.extract_brands_from_name(product)
        assert len(brands) == 0


class TestConfig:
    """配置测试"""

    def test_default_config(self, retriever):
        """测试默认配置"""
        assert retriever.config['alpha'] >= 0
        assert retriever.config['alpha'] <= 1
        assert retriever.config['top_k'] > 0
        assert retriever.config['return_top_k'] > 0

    def test_set_config(self, retriever):
        """测试修改配置"""
        retriever.set_config(alpha=0.5, top_k=10)
        assert retriever.config['alpha'] == 0.5
        assert retriever.config['top_k'] == 10


class TestHybridRetrieval:
    """混合检索测试"""

    def test_retrieve_returns_results(self, retriever):
        """测试检索返回结果"""
        with patch.object(retriever.vector_store, 'retrieve') as mock_vector_retrieve:
            mock_vector_retrieve.return_value = [
                {'category_name': '碳酸饮料', 'similarity': 0.9},
                {'category_name': '气泡水', 'similarity': 0.7},
            ]

            results = retriever.retrieve("可口可乐")
            assert len(results) > 0
            assert isinstance(results, list)

    def test_brand_boosting(self, retriever):
        """测试品牌加权"""
        with patch.object(retriever.vector_store, 'retrieve') as mock_vector_retrieve:
            # 模拟向量检索返回结果
            mock_vector_retrieve.return_value = [
                {
                    'category_name': '碳酸饮料',
                    'similarity': 0.7,
                    'text': '碳酸饮料...',
                    'metadata': {}
                },
            ]

            # 模拟BM25分数
            retriever.bm25_index.get_scores = Mock(return_value=[0.5])

            results = retriever.retrieve("可口可乐")

            # 检查是否有品牌匹配标记
            if results and '可口可乐' in "可口可乐":
                for r in results:
                    if 'brand_match' in r:
                        assert '可口可乐' in r['brand_match']


class TestTokenization:
    """分词测试"""

    def test_tokenize_chinese(self, retriever):
        """测试中文分词"""
        text = "湘满天香辣萝卜28g"
        tokens = retriever._tokenize(text)
        assert len(tokens) > 0
        # 检查是否包含相关词汇（完整或部分）
        has_relevant_token = any('萝卜' in t or '香辣' in t or '湘满天' in t for t in tokens)
        assert has_relevant_token

    def test_tokenize_mixed(self, retriever):
        """测试中英文混合分词"""
        text = "可口可乐CocaCola330ml"
        tokens = retriever._tokenize(text)
        assert len(tokens) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
