"""
智能检索效果测试
对比普通BGE检索 vs 智能检索（BGE + 品牌加权）
"""

import json
from bge_vector_store import BGEVectorStore
from smart_retriever import SmartRetriever


def load_brand_index():
    """加载品牌索引"""
    with open(
        r'c:\Users\wangjinhao\pythonProject\llmrag\enhanced_knowledge_base\brand_index.json',
        encoding='utf-8'
    ) as f:
        return json.load(f)


def test_comparison():
    """对比测试"""
    print("="*80)
    print("普通BGE检索 vs 智能检索（BGE + 品牌加权）")
    print("="*80)
    
    # 初始化向量存储
    vector_store = BGEVectorStore(
        r'c:\Users\wangjinhao\pythonProject\llmrag\knowledge_base'
    )
    vector_store.load_index(
        r'c:\Users\wangjinhao\pythonProject\llmrag\bge_vector_index'
    )
    
    # 初始化智能检索器
    brand_index_path = r'c:\Users\wangjinhao\pythonProject\llmrag\enhanced_knowledge_base\brand_index.json'
    smart_retriever = SmartRetriever(vector_store, brand_index_path)
    
    # 测试商品
    test_products = [
        ("可口可乐330ml罐装", "可口可乐"),
        ("农夫山泉NFC橙汁1L", "农夫山泉"),
        ("元气森林白桃味气泡水500ml", "元气森林"),
        ("百事可乐500ml塑料瓶", "百事可乐"),
        ("王老吉凉茶310ml", "王老吉"),
        ("乐事薯片原味150g", "乐事"),
        ("伊利纯牛奶250ml", "伊利"),
        ("蒙牛酸酸乳原味", "蒙牛"),
    ]
    
    print("\n" + "="*80)
    print("检索效果对比")
    print("="*80)
    
    for product_name, expected_brand in test_products:
        print(f"\n商品: {product_name}")
        print(f"预期品牌: {expected_brand}")
        print("-"*80)
        
        # 普通BGE检索（top_k=8）
        normal_results = vector_store.retrieve(product_name, top_k=8)
        
        # 智能检索（BGE + 品牌加权）
        smart_results = smart_retriever.retrieve(product_name)
        
        print("\n[普通BGE检索 - Top5]")
        for i, result in enumerate(normal_results[:5], 1):
            marker = "✅" if expected_brand in result['metadata'].get('brands', '') else "  "
            print(f"  {marker}{i}. {result['category_name']} ({result['similarity']:.4f})")
        
        print("\n[智能检索 - Top3]")
        for i, result in enumerate(smart_results, 1):
            marker = "✅" if expected_brand in result['metadata'].get('brands', '') else "  "
            brand_info = ""
            if 'brand_match' in result:
                brand_info = f" [品牌: {', '.join(result['brand_match'])}]"
            print(f"  {marker}{i}. {result['category_name']} ({result['similarity']:.4f}){brand_info}")
        
        # 检查正确答案位置
        correct_in_normal_top5 = any(
            expected_brand in r['metadata'].get('brands', '')
            for r in normal_results[:5]
        )
        correct_in_smart_top3 = any(
            expected_brand in r['metadata'].get('brands', '')
            for r in smart_results
        )
        
        print(f"\n正确答案在前5/3中: ", end="")
        print(f"普通={correct_in_normal_top5}, 智能={correct_in_smart_top3}")
    
    print("\n" + "="*80)
    print("配置信息")
    print("="*80)
    print(f"top_k: {smart_retriever.config['top_k']}")
    print(f"brand_boost: +{smart_retriever.config['brand_boost']}")
    print(f"return_top_k: {smart_retriever.config['return_top_k']}")


if __name__ == '__main__':
    test_comparison()
