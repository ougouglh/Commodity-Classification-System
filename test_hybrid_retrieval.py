"""
混合检索测试脚本
对比纯向量检索 vs 混合检索（BGE + BM25）
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.logger import setup_logger
from src.knowledge_base import CategoryKnowledgeBase
from src.vector_store import BGEVectorStore
from src.retriever import HybridRetriever

logger = setup_logger('test_hybrid')


def test_hybrid_retrieval():
    """测试混合检索效果"""

    # 测试商品
    test_products = [
        "湘满天香辣萝卜香辣味28g",
        "可口可乐330ml罐装",
        "农夫山泉550ml瓶装",
        "康师傅红烧牛肉面",
    ]

    print("=" * 80)
    print("混合检索测试")
    print("=" * 80)

    # 加载系统
    print("\n[1/3] 加载向量索引...")
    vector_store = BGEVectorStore()
    vector_store.load_index()
    print(f"     已加载 {len(vector_store.documents)} 个文档")

    print("\n[2/3] 初始化混合检索器...")
    retriever = HybridRetriever(vector_store)
    print(f"     配置: alpha={retriever.config['alpha']:.1f}")
    print(f"          top_k={retriever.config['return_top_k']}")

    print("\n[3/3] 测试检索效果...")

    for product in test_products:
        print("\n" + "-" * 80)
        print(f"商品: {product}")
        print("-" * 80)

        # 混合检索
        results = retriever.retrieve(product)

        print(f"\n{'排名':<4} {'相似度':<8} {'向量分':<8} {'BM25分':<8} {'四级分类'}")
        print("-" * 80)

        for i, r in enumerate(results, 1):
            category = r.get('category_name', '')
            vector_score = r.get('vector_score', 0)
            bm25_score = r.get('bm25_score', 0)
            similarity = r.get('similarity', 0)

            print(f"{i:<4} {similarity:<8.3f} {vector_score:<8.3f} {bm25_score:<8.3f} {category}")

            # 显示品牌匹配信息
            if 'brand_match' in r:
                print(f"      ↳ 品牌匹配: {r['brand_match']}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

    # 对比测试：不同 alpha 值
    print("\n\n[对比测试] 不同 alpha 值的影响")
    print("=" * 80)

    product = "湘满天香辣萝卜香辣味28g"
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

    for alpha in alphas:
        retriever.set_config(alpha=alpha)
        results = retriever.retrieve(product)
        top1 = results[0]['category_name'] if results else '无结果'
        print(f"alpha={alpha:.1f}: {top1}")

    print("\n说明:")
    print("  alpha=0.0: 纯 BM25（词频匹配）")
    print("  alpha=0.5: 各占 50%")
    print("  alpha=0.7: 70% 向量 + 30% BM25（默认）")
    print("  alpha=1.0: 纯向量检索")


if __name__ == '__main__':
    test_hybrid_retrieval()
