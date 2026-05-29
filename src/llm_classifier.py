"""
LLM分类器模块
使用大语言模型API进行商品分类
"""

import json
import os
from typing import List, Dict, Optional
from functools import lru_cache

from openai import OpenAI

from src.config import config
from src.logger import get_logger
from src.vector_store import BGEVectorStore, CategoryRetriever

logger = get_logger(__name__)


class LLMAPIClient:
    """LLM API客户端"""

    def __init__(self,
                 provider: str = None,
                 api_key: str = None,
                 base_url: str = None,
                 model: str = None):
        """
        初始化LLM API客户端
        Args:
            provider: 提供商 (openai/yi/custom)
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
        """
        self.provider = provider or config.LLM_PROVIDER
        self.api_key = api_key or config.OPENAI_API_KEY
        self.base_url = base_url or config.API_BASE_URL
        self.model = model or config.LLM_MODEL

        self.client = self._initialize_client()
        logger.info(f"LLM client initialized: {self.provider}/{self.model}")

    def _initialize_client(self) -> OpenAI:
        """初始化OpenAI客户端"""
        logger.info(f"Initializing LLM client:")
        logger.info(f"  Provider: {self.provider}")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(f"  API Key: {self.api_key[:20]}...")

        if self.provider == 'openai':
            return OpenAI(api_key=self.api_key)
        elif self.base_url:
            return OpenAI(api_key=self.api_key or 'not-needed', base_url=self.base_url)
        else:
            return OpenAI(api_key=self.api_key or 'not-needed')

    def chat(self, messages: List[Dict], temperature: float = None, max_tokens: int = None) -> str:
        """发送对话请求"""
        temperature = temperature or config.LLM_TEMPERATURE
        max_tokens = max_tokens or config.LLM_MAX_TOKENS

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=config.LLM_TIMEOUT
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def classify_product(self, product_name: str, context: str) -> Dict:
        """使用LLM进行商品分类"""
        prompt = self._build_product_profile_prompt(product_name, context)

        messages = [
            {"role": "system", "content": "你是一个专业的商品分析专家，能够根据商品名称和分类知识库信息，输出完整的商品画像。"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.chat(messages)
            return self._parse_product_profile_result(response)
        except Exception as e:
            logger.error(f"Classification failed for {product_name}: {e}")
            return self._error_result(product_name, str(e))

    def _build_product_profile_prompt(self, product_name: str, context: str) -> str:
        """构建商品画像提示词"""
        return f"""你是一个专业的商品分析专家。请根据商品名称和提供的知识库信息，输出商品画像。

重要原则：
- **只输出知识库中明确提供的信息**
- **对于知识库中没有的信息，必须输出"未知"，禁止编造**
- **任何字段都不能随意填写，必须有据可查**

商品名称：{product_name}

{context}

分析步骤：
1. 首先识别商品名称中明确包含的信息（品牌、口味、规格等）
2. 从知识库中检索该分类的完整信息
3. 严格按照以下规则填写每个字段：

字段填写规则：
- **分类信息**：必须根据知识库中的定义判断，知识库没有明确定义的分类不得输出
- **口味类型**：仅当商品名称中包含口味描述，或知识库中该分类只有一种口味时方可填写
- **包装类型**：仅当商品名称中包含包装描述，或知识库中该分类只有一种包装时方可填写
- **含糖属性**：仅当知识库明确标注含糖/无糖信息时方可填写
- **代表品牌**：仅当商品名称中包含品牌，或知识库中该分类的品牌数量≤3时方可填写
- **功能特征**：仅当知识库中明确描述功能时方可填写
- **产地信息**：仅当知识库中明确标注产地时方可填写

请按以下JSON格式返回商品画像：
{{
    "product_name": "{product_name}",
    "level_1": "一级分类（知识库中有则填写，无则留空）",
    "level_2": "二级分类（知识库中有则填写，无则留空）",
    "level_3": "三级分类（知识库中有则填写，无则留空）",
    "level_4": "四级分类（知识库中有则填写，无则留空）",
    "flavor": "口味（商品名称或知识库中明确有则填写，否则填写"未知"）",
    "packaging": "包装（商品名称或知识库中明确有则填写，否则填写"未知"）",
    "sugar_content": "含糖属性（知识库中明确有则填写含糖/无糖/低糖，否则填写"未知"）",
    "brand": "品牌（商品名称中明确包含则填写，否则填写"未知"）",
    "function": "功能特征（知识库中明确描述则填写，否则填写"未知"）",
    "origin": "产地（知识库中明确标注则填写，否则填写"未知"）",
    "confidence": 0.0到1.0之间的置信度（根据知识库信息完整度评估）,
    "reasoning": "分析理由（说明每个判断的依据，以及哪些信息是知识库提供的，哪些是未知的）"
}}

警告：
- 绝对不能编造任何信息
- "全国"、"常见"、"通常"等模糊词汇不允许出现
- 所有信息必须有明确的知识库依据或商品名称依据"""

    def _parse_product_profile_result(self, result_str: str) -> Dict:
        """解析商品画像结果"""
        try:
            result_str = result_str.strip()

            if result_str.startswith('```'):
                lines = result_str.split('\n')
                lines = [line for line in lines if not line.startswith('```')]
                result_str = '\n'.join(lines)

            result = json.loads(result_str)

            return {
                'product_name': result.get('product_name', ''),
                'level_1': result.get('level_1', ''),
                'level_2': result.get('level_2', ''),
                'level_3': result.get('level_3', ''),
                'level_4': result.get('level_4', ''),
                'flavor': result.get('flavor', ''),
                'packaging': result.get('packaging', ''),
                'sugar_content': result.get('sugar_content', ''),
                'brand': result.get('brand', ''),
                'function': result.get('function', ''),
                'origin': result.get('origin', ''),
                'confidence': float(result.get('confidence', 0.0)),
                'reasoning': result.get('reasoning', '')
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.debug(f"Raw result: {result_str}")
            return self._error_result('', f'JSON解析失败: {str(e)}')

    def _error_result(self, product_name: str, error_msg: str) -> Dict:
        """返回错误结果"""
        return {
            'product_name': product_name,
            'level_1': '', 'level_2': '', 'level_3': '', 'level_4': '',
            'flavor': '', 'packaging': '', 'sugar_content': '',
            'brand': '', 'function': '', 'origin': '',
            'confidence': 0.0,
            'reasoning': f'Error: {error_msg}'
        }


class ProductClassifier:
    """商品分类器"""

    def __init__(self, vector_store: BGEVectorStore, llm_client: LLMAPIClient):
        """
        初始化分类器
        Args:
            vector_store: 向量存储实例
            llm_client: LLM API客户端实例
        """
        self.vector_store = vector_store
        self.llm = llm_client
        self.retriever = CategoryRetriever(vector_store)
        self._cache_enabled = config.ENABLE_CACHE
        self._cache = {}

    def classify_product(self, product_name: str, top_k: int = None) -> Dict:
        """对单个商品进行分类"""
        top_k = top_k or config.RETRIEVAL_TOP_K

        # 检查缓存
        if self._cache_enabled and product_name in self._cache:
            logger.debug(f"Cache hit for: {product_name}")
            return self._cache[product_name]

        retrieved_categories = self.retriever.retrieve_for_product(product_name, top_k=top_k)
        context = self.retriever.format_retrieved_context(retrieved_categories)

        classification = self.llm.classify_product(product_name, context)

        result = {
            'product_name': product_name,
            'level_1': classification['level_1'],
            'level_2': classification['level_2'],
            'level_3': classification['level_3'],
            'level_4': classification['level_4'],
            'flavor': classification['flavor'],
            'packaging': classification['packaging'],
            'sugar_content': classification['sugar_content'],
            'brand': classification['brand'],
            'function': classification['function'],
            'origin': classification['origin'],
            'confidence': classification['confidence'],
            'reasoning': classification['reasoning'],
            'retrieved_categories': retrieved_categories
        }

        # 缓存结果
        if self._cache_enabled:
            if len(self._cache) >= config.CACHE_SIZE:
                # 简单的LRU：删除第一个
                self._cache.pop(next(iter(self._cache)))
            self._cache[product_name] = result

        return result

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        return {
            'enabled': self._cache_enabled,
            'size': len(self._cache),
            'max_size': config.CACHE_SIZE
        }
