"""
项目清理脚本
运行此脚本删除旧的代码文件
"""

import os
from pathlib import Path

# 需要删除的旧文件
OLD_FILES = [
    'rag_knowledge_base.py', 'rag_vector_store.py', 'rag_classifier.py', 'rag_main.py',
    'bge_main.py', 'bge_vector_store.py', 'bge_llm_classifier.py',
    'enhanced_main.py', 'enhanced_knowledge_base.py', 'enhanced_bge_vector_store.py',
    'enhanced_main_v2.py', 'hf_embedding_store.py', 'retrieval_config.py', 'smart_retriever.py',
    'compare_retrieval.py', 'compare_knowledge_base.py', 'demo_index_usage.py', 'demo_product_profile.py',
    'simple_brand_test.py', 'review_scripts.py',
    'test_bge_loading.py', 'test_llm_api.py', 'test_llm_api_simple.py',
    'test_llm_independent_judgment.py', 'test_retrieval.py', 'test_smart_retriever.py', 'test_smart_vs_normal.py',
    'split_excel_to_csv.py', 'analyze_csv_structure.py'
]

def cleanup():
    """删除旧文件"""
    deleted = []
    not_found = []

    for filename in OLD_FILES:
        filepath = Path(filename)
        if filepath.exists():
            filepath.unlink()
            deleted.append(filename)
            print(f"[DELETE] {filename}")
        else:
            not_found.append(filename)

    print("\n" + "=" * 60)
    print(f"Deleted {len(deleted)} files")
    print(f"Not found: {len(not_found)} files")
    print("=" * 60)

    return deleted

if __name__ == '__main__':
    cleanup()
