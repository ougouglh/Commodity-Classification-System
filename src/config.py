"""
配置管理模块
集中管理所有配置项，支持环境变量覆盖
"""

import os
from pathlib import Path
from typing import Optional
import dotenv


class Config:
    """项目配置类 - 使用属性延迟加载环境变量"""

    # 项目路径
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    KNOWLEDGE_BASE_DIR = BASE_DIR / 'knowledge_base'
    VECTOR_INDEX_DIR = BASE_DIR / 'bge_vector_index'
    OUTPUT_DIR = BASE_DIR / 'output'
    LOGS_DIR = BASE_DIR / 'logs'

    # 日志配置
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = LOGS_DIR / 'app.log'

    @classmethod
    def _get_env(cls, key: str, default=''):
        """获取环境变量，自动加载.env文件"""
        dotenv.load_dotenv(cls.BASE_DIR / '.env', override=True)
        return os.getenv(key, default)

    # 使用属性动态获取配置
    @property
    def BGE_MODEL_NAME(self):
        return self._get_env('BGE_MODEL_NAME', 'BAAI/bge-base-zh-v1.5')

    @property
    def BGE_DEVICE(self):
        return self._get_env('BGE_DEVICE', 'cpu')

    @property
    def BGE_BATCH_SIZE(self):
        return int(self._get_env('BGE_BATCH_SIZE', '32'))

    @property
    def BGE_MAX_LENGTH(self):
        return int(self._get_env('BGE_MAX_LENGTH', '512'))

    @property
    def LLM_PROVIDER(self):
        return self._get_env('LLM_PROVIDER', 'openai')

    @property
    def LLM_MODEL(self):
        return self._get_env('LLM_MODEL', 'gpt-3.5-turbo')

    @property
    def OPENAI_API_KEY(self):
        return self._get_env('OPENAI_API_KEY', '')

    @property
    def API_BASE_URL(self):
        return self._get_env('API_BASE_URL', '')

    @property
    def LLM_TEMPERATURE(self):
        return float(self._get_env('LLM_TEMPERATURE', '0.3'))

    @property
    def LLM_MAX_TOKENS(self):
        return int(self._get_env('LLM_MAX_TOKENS', '1000'))

    @property
    def LLM_TIMEOUT(self):
        return int(self._get_env('LLM_TIMEOUT', '30'))

    @property
    def RETRIEVAL_TOP_K(self):
        return int(self._get_env('RETRIEVAL_TOP_K', '3'))

    @property
    def RETRIEVAL_CANDIDATE_K(self):
        return int(self._get_env('RETRIEVAL_CANDIDATE_K', '8'))

    @property
    def BRAND_BOOST_SCORE(self):
        return float(self._get_env('BRAND_BOOST_SCORE', '0.15'))

    @property
    def ENABLE_CACHE(self):
        return self._get_env('ENABLE_CACHE', 'true').lower() == 'true'

    @property
    def CACHE_SIZE(self):
        return int(self._get_env('CACHE_SIZE', '1000'))

    @property
    def LOG_LEVEL(self):
        return self._get_env('LOG_LEVEL', 'INFO')

    @property
    def LOG_MAX_BYTES(self):
        return int(self._get_env('LOG_MAX_BYTES', '10485760'))

    @property
    def LOG_BACKUP_COUNT(self):
        return int(self._get_env('LOG_BACKUP_COUNT', '5'))

    @property
    def BATCH_SIZE(self):
        return int(self._get_env('BATCH_SIZE', '10'))

    @property
    def MAX_WORKERS(self):
        return int(self._get_env('MAX_WORKERS', '4'))

    @property
    def USE_FAISS(self):
        return self._get_env('USE_FAISS', 'false').lower() == 'true'

    @property
    def FAISS_INDEX_TYPE(self):
        return self._get_env('FAISS_INDEX_TYPE', 'flat')

    @property
    def HYBRID_ALPHA(self):
        """混合检索中向量检索的权重 (0-1)，BM25权重为 1-alpha"""
        return float(self._get_env('HYBRID_ALPHA', '0.7'))

    def ensure_dirs(self):
        """确保所有必需的目录存在"""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """验证配置是否有效"""
        errors = []

        if self.LLM_PROVIDER in ['openai', 'yi', 'custom'] and not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")

        if self.LLM_PROVIDER == 'custom' and not self.API_BASE_URL:
            errors.append("API_BASE_URL is required when LLM_PROVIDER is 'custom'")

        if errors:
            print("配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True


# 全局配置实例
config = Config()
config.ensure_dirs()
