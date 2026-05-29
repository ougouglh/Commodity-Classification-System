"""
向量存储模块
使用BGE模型进行文本编码和相似度检索
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class BGEVectorStore:
    """基于BGE模型的向量存储"""

    def __init__(self, knowledge_base_path: Optional[str] = None, model_name: Optional[str] = None):
        """
        初始化向量存储
        Args:
            knowledge_base_path: 知识库路径
            model_name: BGE模型名称
        """
        self.knowledge_base_path = Path(knowledge_base_path or config.KNOWLEDGE_BASE_DIR)
        self.model_name = model_name or config.BGE_MODEL_NAME
        self.documents = []
        self.model = None
        self.tokenizer = None
        self.document_embeddings = None

        self.brand_index = {}
        self.flavor_index = {}

    def load_model(self):
        """加载BGE模型"""
        if self.model is None:
            logger.info(f"Loading BGE model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            except Exception as e:
                logger.warning(f"Safetensors loading failed, trying regular loading: {e}")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )

            self.model.eval()
            logger.info("BGE model loaded successfully")

    def mean_pooling(self, model_output, attention_mask):
        """平均池化"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_texts(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """编码文本为向量"""
        if self.model is None:
            self.load_model()

        encoded_input = self.tokenizer(texts, padding=True, truncation=True,
                                      max_length=config.BGE_MAX_LENGTH, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.numpy()

    def load_knowledge_base(self):
        """加载知识库文档"""
        with open(self.knowledge_base_path / 'knowledge_documents.json', 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        logger.info(f"Loaded {len(self.documents)} knowledge documents")

        if (self.knowledge_base_path / 'brand_index.json').exists():
            with open(self.knowledge_base_path / 'brand_index.json', 'r', encoding='utf-8') as f:
                self.brand_index = json.load(f)
            logger.info(f"Loaded {len(self.brand_index)} brand index entries")

        if (self.knowledge_base_path / 'flavor_index.json').exists():
            with open(self.knowledge_base_path / 'flavor_index.json', 'r', encoding='utf-8') as f:
                self.flavor_index = json.load(f)
            logger.info(f"Loaded {len(self.flavor_index)} flavor index entries")

    def build_index(self):
        """构建向量索引"""
        if not self.documents:
            self.load_knowledge_base()

        self.load_model()

        texts = [doc['text'] for doc in self.documents]
        logger.info(f"Encoding {len(texts)} documents...")

        batch_size = config.BGE_BATCH_SIZE
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.encode_texts(batch)
            all_embeddings.append(batch_embeddings)
            if (i // batch_size + 1) % 10 == 0:
                logger.debug(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} documents")

        self.document_embeddings = np.vstack(all_embeddings)
        logger.info(f"Vector index built with shape: {self.document_embeddings.shape}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索最相关的文档"""
        if self.document_embeddings is None:
            self.build_index()

        query_embedding = self.encode_texts([query])[0]
        similarities = np.dot(self.document_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['similarity'] = float(similarities[idx])

            # 展开metadata到顶层
            if 'metadata' in doc:
                metadata = doc.pop('metadata')
                for key, value in metadata.items():
                    if key not in doc:
                        doc[key] = value

            results.append(doc)

        return results

    def save_index(self, output_path: Optional[str] = None):
        """保存索引"""
        output_path = Path(output_path or config.VECTOR_INDEX_DIR)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.document_embeddings is not None:
            np.save(output_path / 'document_embeddings.npy', self.document_embeddings)

        with open(output_path / 'documents.json', 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

        with open(output_path / 'model_info.json', 'w', encoding='utf-8') as f:
            json.dump({'model_name': self.model_name}, f, ensure_ascii=False)

        logger.info(f"Index saved to {output_path}")

    def load_index(self, index_path: Optional[str] = None):
        """加载索引"""
        index_path = Path(index_path or config.VECTOR_INDEX_DIR)

        self.document_embeddings = np.load(index_path / 'document_embeddings.npy')

        with open(index_path / 'documents.json', 'r', encoding='utf-8') as f:
            self.documents = json.load(f)

        # 展开metadata
        for doc in self.documents:
            if 'metadata' in doc:
                metadata = doc.pop('metadata')
                for key, value in metadata.items():
                    if key not in doc:
                        doc[key] = value

        with open(index_path / 'model_info.json', 'r', encoding='utf-8') as f:
            model_info = json.load(f)
            self.model_name = model_info.get('model_name', self.model_name)

        if (index_path / 'brand_index.json').exists():
            with open(index_path / 'brand_index.json', 'r', encoding='utf-8') as f:
                self.brand_index = json.load(f)

        if (index_path / 'flavor_index.json').exists():
            with open(index_path / 'flavor_index.json', 'r', encoding='utf-8') as f:
                self.flavor_index = json.load(f)

        logger.info(f"Index loaded: {len(self.documents)} documents")


class CategoryRetriever:
    """分类检索器"""

    def __init__(self, vector_store: BGEVectorStore):
        self.vector_store = vector_store

    def retrieve_for_product(self, product_name: str, top_k: int = None) -> List[Dict]:
        """为商品检索最相关的分类"""
        top_k = top_k or config.RETRIEVAL_TOP_K
        results = self.vector_store.retrieve(product_name, top_k=top_k)

        retrieved_categories = []
        for result in results:
            # 兼容处理metadata
            if 'metadata' in result:
                metadata = result['metadata']
                brands = metadata.get('brands', '')
                brand_count = metadata.get('brand_count', 0)
                all_flavors = metadata.get('all_flavors', '')
                flavor_count = metadata.get('flavor_count', 0)
                all_packagings = metadata.get('all_packagings', '')
                packaging_count = metadata.get('packaging_count', 0)
                level_1 = metadata.get('level_1', '')
                level_2 = metadata.get('level_2', '')
                level_3 = metadata.get('level_3', '')
                level_4 = metadata.get('level_4', '')
            else:
                brands = result.get('brands', '')
                brand_count = result.get('brand_count', 0)
                all_flavors = result.get('all_flavors', '')
                flavor_count = result.get('flavor_count', 0)
                all_packagings = result.get('all_packagings', '')
                packaging_count = result.get('packaging_count', 0)
                level_1 = result.get('level_1', '')
                level_2 = result.get('level_2', '')
                level_3 = result.get('level_3', '')
                level_4 = result.get('level_4', '')

            category_info = {
                'category_name': result['category_name'],
                'full_path': result['full_path'],
                'similarity': result['similarity'],
                'definition': result.get('definition', ''),
                'includes': result.get('includes', ''),
                'excludes': result.get('excludes', ''),
                'notes': result.get('notes', ''),
                'brands': brands,
                'brand_count': brand_count,
                'primary_flavor': result.get('primary_flavor', ''),
                'all_flavors': all_flavors,
                'flavor_count': flavor_count,
                'primary_packaging': result.get('primary_packaging', ''),
                'all_packagings': all_packagings,
                'packaging_count': packaging_count,
                'sugar_content': result.get('sugar_content', ''),
                'product_type': result.get('product_type', ''),
                'origin': result.get('origin', ''),
                'function': result.get('function', ''),
                'level_1': level_1,
                'level_2': level_2,
                'level_3': level_3,
                'level_4': level_4
            }

            retrieved_categories.append(category_info)

        return retrieved_categories

    def format_retrieved_context(self, retrieved_categories: List[Dict]) -> str:
        """格式化检索结果为上下文"""
        context_parts = ["以下是相关的商品分类知识：\n"]

        for i, cat in enumerate(retrieved_categories, 1):
            context_parts.append(f"\n参考分类 {i}（相似度: {cat['similarity']:.4f}）：")
            context_parts.append(f"  分类名称：{cat['category_name']}")
            context_parts.append(f"  分类路径：{cat['full_path']}")

            if cat['definition']:
                context_parts.append(f"  定义：{cat['definition'][:300]}...")

            if cat['brands']:
                context_parts.append(f"  代表品牌（共{cat['brand_count']}个）：{cat['brands']}")

            if cat['all_flavors']:
                context_parts.append(f"  口味类型（共{cat['flavor_count']}种）：{cat['all_flavors']}")

            if cat['all_packagings']:
                context_parts.append(f"  包装类型（共{cat['packaging_count']}种）：{cat['all_packagings']}")

        return '\n'.join(context_parts)
