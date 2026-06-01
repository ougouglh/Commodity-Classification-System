"""
配置管理单元测试
"""

import pytest
import os
from pathlib import Path
import sys
from unittest.mock import patch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, config


class TestConfigDefaults:
    """测试默认配置"""

    def test_bge_model_default(self):
        """测试BGE模型默认值"""
        assert config.BGE_MODEL_NAME == 'BAAI/bge-base-zh-v1.5'

    def test_llm_provider_default(self):
        """测试LLM提供商默认值"""
        assert config.LLM_PROVIDER in ['openai', 'yi', 'custom']

    def test_retrieval_defaults(self):
        """测试检索配置默认值"""
        assert config.RETRIEVAL_TOP_K > 0
        assert config.RETRIEVAL_CANDIDATE_K >= config.RETRIEVAL_TOP_K
        assert 0 <= config.BRAND_BOOST_SCORE <= 1

    def test_hybrid_alpha_default(self):
        """测试混合检索alpha默认值"""
        assert 0 <= config.HYBRID_ALPHA <= 1
        # 默认应该是0.7左右
        assert 0.5 <= config.HYBRID_ALPHA <= 0.9

    def test_cache_defaults(self):
        """测试缓存配置默认值"""
        assert isinstance(config.ENABLE_CACHE, bool)
        assert config.CACHE_SIZE > 0


class TestConfigPaths:
    """测试路径配置"""

    def test_directories_exist(self):
        """测试目录配置"""
        assert config.BASE_DIR.exists()
        assert isinstance(config.DATA_DIR, Path)
        assert isinstance(config.VECTOR_INDEX_DIR, Path)
        assert isinstance(config.OUTPUT_DIR, Path)

    def test_log_file_path(self):
        """测试日志文件路径"""
        assert isinstance(config.LOG_FILE, Path)
        assert 'log' in str(config.LOG_FILE).lower()


class TestConfigValidation:
    """测试配置验证"""

    def test_validate_without_api_key(self, monkeypatch):
        """测试无API密钥时的验证"""
        # Mock _get_env方法返回空API密钥
        def mock_get_env(key, default=''):
            if key == 'OPENAI_API_KEY':
                return ''  # 返回空值
            elif key == 'LLM_PROVIDER':
                return 'openai'
            return default

        monkeypatch.setattr('src.config.os.getenv', lambda k, d=None: mock_get_env(k, d))

        test_config = Config()
        result = test_config.validate()
        assert result == False  # 应该验证失败

    def test_validate_with_api_key(self):
        """测试有API密钥时的验证"""
        # 设置测试API密钥
        os.environ['OPENAI_API_KEY'] = 'test-key-12345'
        test_config = Config()

        result = test_config.validate()
        assert result == True  # 应该验证成功


class TestConfigEnsureDirs:
    """测试目录创建"""

    def test_ensure_dirs_creates_directories(self, tmp_path):
        """测试ensure_dirs创建目录"""
        test_config = Config()
        test_config.OUTPUT_DIR = tmp_path / "output"
        test_config.LOGS_DIR = tmp_path / "logs"

        test_config.ensure_dirs()

        assert test_config.OUTPUT_DIR.exists()
        assert test_config.LOGS_DIR.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
