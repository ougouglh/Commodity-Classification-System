"""
RAG商品分类系统 - 主入口程序
BGE embedding + LLM API
"""

import sys
from pathlib import Path

from src.config import config
from src.logger import setup_logger, get_logger
from src.knowledge_base import CategoryKnowledgeBase
from src.vector_store import BGEVectorStore
from src.llm_classifier import LLMAPIClient, ProductClassifier
from src.retriever import SmartRetriever
from src.evaluator import run_evaluation, load_test_data

# 设置日志
logger = setup_logger('rag_system')


class RAGClassificationSystem:
    """RAG商品分类系统"""

    def __init__(self):
        """初始化系统"""
        self.vector_store = None
        self.classifier = None
        self.smart_retriever = None
        logger.info("RAG Classification System initialized")

    def build_knowledge_base(self):
        """构建知识库和向量索引"""
        print("=" * 60)
        print("Building Knowledge Base and Vector Index")
        print("=" * 60)

        # 构建知识库
        kb = CategoryKnowledgeBase()
        kb.load_data()
        kb.build_hierarchy()
        kb.get_category_definitions()
        stats = kb.save_knowledge_base()

        # 构建向量索引
        print("\n" + "=" * 60)
        print("Building BGE Vector Index")
        print("=" * 60)

        self.vector_store = BGEVectorStore()
        self.vector_store.load_knowledge_base()
        self.vector_store.build_index()
        self.vector_store.save_index()

        print("\n" + "=" * 60)
        print("Build completed!")
        print(f"  Categories: {stats['total_categories']}")
        print(f"  Brands: {stats['total_brands']}")
        print(f"  Flavors: {stats['total_flavors']}")
        print("=" * 60)

    def load_system(self):
        """加载已构建的系统"""
        print("=" * 60)
        print("Loading RAG Classification System")
        print("=" * 60)

        config.validate()

        # 加载向量存储
        self.vector_store = BGEVectorStore()
        self.vector_store.load_index()
        print(f"[OK] Loaded {len(self.vector_store.documents)} documents")

        # 初始化智能检索器
        self.smart_retriever = SmartRetriever(self.vector_store)
        print(f"[OK] Initialized Smart Retriever")

        # 初始化LLM客户端
        print(f"\n[DEBUG] Loading LLM client with:")
        print(f"  Provider: {config.LLM_PROVIDER}")
        print(f"  Model: {config.LLM_MODEL}")
        print(f"  Base URL: {config.API_BASE_URL}")
        print(f"  API Key: {config.OPENAI_API_KEY[:20]}...")

        llm_client = LLMAPIClient()
        print(f"[OK] Initialized LLM client: {config.LLM_PROVIDER}/{config.LLM_MODEL}")

        # 初始化分类器
        self.classifier = ProductClassifier(self.vector_store, llm_client)
        print(f"[OK] Initialized Product Classifier")

        print("\n" + "=" * 60)
        print("System loaded successfully!")
        print("=" * 60)
        print(f"\nRetrieval config:")
        print(f"  top_k: {config.RETRIEVAL_TOP_K}")
        print(f"  candidate_k: {config.RETRIEVAL_CANDIDATE_K}")
        print(f"  brand_boost: +{config.BRAND_BOOST_SCORE}")
        print(f"  cache_enabled: {config.ENABLE_CACHE}")

    def classify_single(self, product_name: str) -> dict:
        """分类单个商品"""
        logger.info(f"Classifying: {product_name}")

        result = self.classifier.classify_product(product_name)

        print("\n" + "-" * 60)
        print("Classification Result:")
        print("-" * 60)
        print(f"  Product: {result['product_name']}")
        print(f"  Category: {result['level_1']} > {result['level_2']} > {result['level_3']} > {result['level_4']}")
        print(f"  Flavor: {result['flavor']}")
        print(f"  Packaging: {result['packaging']}")
        print(f"  Sugar: {result['sugar_content']}")
        print(f"  Brand: {result['brand']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        if result['reasoning']:
            print(f"  Reasoning: {result['reasoning'][:100]}...")

        return result

    def classify_batch(self, input_file: str, output_file: str):
        """批量分类"""
        import pandas as pd

        print("=" * 60)
        print(f"Batch Classification: {input_file} -> {output_file}")
        print("=" * 60)

        df = pd.read_csv(input_file)
        if '商品名称' not in df.columns:
            print("[ERROR] CSV must contain '商品名称' column")
            return

        results = []
        total = len(df)

        for idx, row in df.iterrows():
            product_name = row['商品名称']
            try:
                result = self.classifier.classify_product(product_name)
                results.append({
                    '商品名称': product_name,
                    '一级分类': result['level_1'],
                    '二级分类': result['level_2'],
                    '三级分类': result['level_3'],
                    '四级分类': result['level_4'],
                    '口味': result['flavor'],
                    '包装': result['packaging'],
                    '含糖属性': result['sugar_content'],
                    '品牌': result['brand'],
                    '功能': result['function'],
                    '产地': result['origin'],
                    '置信度': result['confidence'],
                    '推理': result['reasoning']
                })
                print(f"[{idx+1}/{total}] {product_name} -> {result['level_4']}")
            except Exception as e:
                logger.error(f"Failed to classify {product_name}: {e}")
                results.append({'商品名称': product_name, 'error': str(e)})

        # 保存结果
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {output_file}")

    def interactive_mode(self):
        """交互式模式"""
        print("=" * 60)
        print("Interactive Mode")
        print("Enter product names to classify, or 'quit' to exit")
        print("=" * 60)

        while True:
            try:
                product_name = input("\n请输入商品名称: ").strip()

                if product_name.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if product_name:
                    self.classify_single(product_name)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


def main():
    """主入口"""
    system = RAGClassificationSystem()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py build         - Build knowledge base and index")
        print("  python main.py interactive   - Interactive classification mode")
        print("  python main.py classify <input.csv> <output.csv> - Batch classification")
        print("  python main.py evaluate <test.csv> [level] - Evaluate system performance")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'build':
        system.build_knowledge_base()

    elif command == 'interactive':
        system.load_system()
        system.interactive_mode()

    elif command == 'classify':
        if len(sys.argv) < 4:
            print("[ERROR] Please specify input and output files")
            sys.exit(1)
        system.load_system()
        system.classify_batch(sys.argv[2], sys.argv[3])

    elif command == 'evaluate':
        if len(sys.argv) < 3:
            print("[ERROR] Please specify test CSV file")
            sys.exit(1)

        level = int(sys.argv[3]) if len(sys.argv) > 3 else 4

        system.load_system()

        # 加载测试数据
        test_data = load_test_data(sys.argv[2])
        print(f"\n[INFO] 加载了 {len(test_data)} 个测试样本")

        # 运行评估
        metrics, evaluator, predictions = run_evaluation(system, test_data, level)

        # 打印报告
        report = evaluator.generate_report(metrics)
        print("\n" + report)

        # 保存错误案例
        error_file = config.OUTPUT_DIR / 'evaluation_errors.csv'
        evaluator.save_errors(str(error_file))

    else:
        print(f"[ERROR] Unknown command: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
