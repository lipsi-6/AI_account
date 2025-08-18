"""
配置管理器测试
"""

import pytest
import pytest_asyncio
import tempfile
import os
from pathlib import Path
import yaml
import asyncio

from modules.config_manager import ConfigManager, APIKeysConfig, LLMSettingsConfig


class TestConfigManager:
    """配置管理器测试类"""
    
    @pytest_asyncio.fixture
    async def config_file(self):
        """创建临时配置文件"""
        config_data = {
            'api_keys': {
                'openai': 'test-openai-key',
                'anthropic': 'test-anthropic-key'
            },
            'llm_settings': {
                'provider': 'openai',
                'primary_model': 'gpt-4o',
                'temperature': 0.7,
                'max_tokens': 4000
            },
            'embedding_settings': {
                'provider': 'openai',
                'model': 'text-embedding-3-large'
            },
            'paths': {
                'drafts_folder': './test_output/drafts',
                'logs_folder': './test_logs',
                'watch_folder': './test_input',
                'temp_downloads_folder': './test_temp',
                'prompt_templates_folder': './test_prompts'
            },
            'performance': {
                'max_concurrent_downloads': 3,
                'max_concurrent_llm_calls': 2,
                'api_retry_attempts': 3,
                'api_retry_delay_seconds': 1
            },
            'logging': {
                'level': 'DEBUG',
                'file_path': './test_logs/test.log'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # 清理
        os.unlink(temp_path)
    
    def test_constructor_no_side_effects(self):
        """测试构造函数无副作用"""
        config = ConfigManager()
        
        # 检查所有配置项都是None
        assert config.api_keys is None
        assert config.llm_settings is None
        assert config.embedding_settings is None
        assert config.paths is None
        assert not config.is_initialized()
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, config_file):
        """测试成功初始化"""
        config = ConfigManager()
        await config.initialize(config_file)
        
        assert config.is_initialized()
        assert config.api_keys is not None
        assert config.api_keys.openai == 'test-openai-key'
        assert config.llm_settings is not None
        assert config.llm_settings.provider == 'openai'
    
    @pytest.mark.asyncio
    async def test_initialization_file_not_found(self):
        """测试配置文件不存在"""
        config = ConfigManager()
        
        with pytest.raises(FileNotFoundError):
            await config.initialize('nonexistent.yaml')
    
    @pytest.mark.asyncio
    async def test_environment_variables(self, config_file):
        """测试环境变量处理"""
        # 设置环境变量
        os.environ['OPENAI_API_KEY'] = 'env-openai-key'
        
        try:
            config = ConfigManager()
            await config.initialize(config_file)
            
            # 环境变量应该覆盖配置文件中的值
            assert config.api_keys.openai == 'test-openai-key'  # 配置文件优先
            
        finally:
            # 清理环境变量
            if 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
    
    @pytest.mark.asyncio
    async def test_get_api_key(self, config_file):
        """测试API密钥获取"""
        config = ConfigManager()
        await config.initialize(config_file)
        
        assert config.get_api_key('openai') == 'test-openai-key'
        assert config.get_api_key('anthropic') == 'test-anthropic-key'
        assert config.get_api_key('nonexistent') is None
    
    @pytest.mark.asyncio
    async def test_directory_creation(self, config_file):
        """测试目录创建"""
        config = ConfigManager()
        await config.initialize(config_file)
        
        # 检查目录是否被创建
        assert Path(config.paths.drafts_folder).exists()
        assert Path(config.paths.logs_folder).exists()
        
        # 清理测试目录
        import shutil
        for path in [config.paths.drafts_folder, config.paths.logs_folder, 
                    config.paths.watch_folder, config.paths.temp_downloads_folder,
                    config.paths.prompt_templates_folder]:
            if Path(path).exists():
                shutil.rmtree(path, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_validation_missing_api_key(self):
        """测试缺少API密钥的验证"""
        config_data = {
            'llm_settings': {'provider': 'openai'},
            'embedding_settings': {'provider': 'openai'},
            'paths': {'drafts_folder': './test'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = ConfigManager()
            with pytest.raises(ValueError, match="At least one LLM API key must be provided"):
                await config.initialize(temp_path)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])
