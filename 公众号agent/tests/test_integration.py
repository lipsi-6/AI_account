"""
集成测试
"""

import pytest
import pytest_asyncio
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from modules.config_manager import ConfigManager
from modules.llm_provider import LLMProviderService, LLMResponse
from modules.embedding_provider import EmbeddingProviderService, EmbeddingResponse


@pytest.mark.integration
class TestSystemIntegration:
    """系统集成测试"""
    
    @pytest_asyncio.fixture
    async def full_config(self):
        """完整配置"""
        config_data = {
            'api_keys': {
                'openai': 'test-openai-key'
            },
            'llm_settings': {
                'provider': 'openai',
                'primary_model': 'gpt-4o'
            },
            'embedding_settings': {
                'provider': 'openai',
                'model': 'text-embedding-3-large'
            },
            'vector_db': {
                'provider': 'chromadb',
                'path': './test_data/chroma_db'
            },
            'knowledge_graph': {
                'provider': 'networkx',
                'file_path': './test_data/kg.graphml'
            },
            'paths': {
                'drafts_folder': './test_output/drafts',
                'logs_folder': './test_logs',
                'watch_folder': './test_input',
                'temp_downloads_folder': './test_temp',
                'prompt_templates_folder': './prompts'
            },
            'performance': {
                'max_concurrent_downloads': 2,
                'max_concurrent_llm_calls': 2,
                'api_retry_attempts': 2,
                'api_retry_delay_seconds': 0.1
            },
            'logging': {
                'level': 'INFO',
                'file_path': './test_logs/test.log'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        config = ConfigManager()
        await config.initialize(config_path)
        
        yield config
        
        # 清理
        Path(config_path).unlink(missing_ok=True)
        import shutil
        for path in ['./test_data', './test_output', './test_logs', './test_temp', './test_input']:
            shutil.rmtree(path, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_config_llm_integration(self, full_config):
        """测试配置和LLM服务集成"""
        # 模拟成功的API响应
        mock_response_data = {
            "choices": [{
                "message": {"content": "Test response"},
                "finish_reason": "stop"
            }],
            "usage": {"total_tokens": 50}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # 初始化LLM服务
            llm_service = LLMProviderService(full_config)
            
            # 测试生成
            response = await llm_service.generate(
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            assert response.content == "Test response"
            assert response.provider == "openai"
            
            await llm_service.close()
    
    @pytest.mark.asyncio
    async def test_config_embedding_integration(self, full_config):
        """测试配置和嵌入服务集成"""
        # 模拟成功的嵌入响应
        mock_response_data = {
            "data": [{"embedding": [0.1, 0.2] * 1536}],  # 3072维
            "usage": {"total_tokens": 10}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # 初始化嵌入服务
            embedding_service = EmbeddingProviderService(full_config)
            
            # 测试嵌入生成
            embedding = await embedding_service.embed_text("Test text")
            
            assert len(embedding) == 3072
            
            await embedding_service.close()
    
    @pytest.mark.asyncio
    async def test_llm_embedding_coordination(self, full_config):
        """测试LLM和嵌入服务协调工作"""
        # 模拟LLM响应
        llm_mock_data = {
            "choices": [{
                "message": {"content": "This is about machine learning"},
                "finish_reason": "stop"
            }],
            "usage": {"total_tokens": 20}
        }
        
        # 模拟嵌入响应
        embedding_mock_data = {
            "data": [{"embedding": [0.1] * 3072}],
            "usage": {"total_tokens": 5}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # 设置不同的响应
            responses = [llm_mock_data, embedding_mock_data]
            response_iter = iter(responses)
            
            async def mock_json():
                return next(response_iter)
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = mock_json
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # 初始化服务
            llm_service = LLMProviderService(full_config)
            embedding_service = EmbeddingProviderService(full_config)
            
            # 测试协调工作流
            # 1. 生成文本
            llm_response = await llm_service.generate(
                messages=[{"role": "user", "content": "Describe machine learning"}]
            )
            
            # 2. 对生成的文本进行嵌入
            embedding = await embedding_service.embed_text(llm_response.content)
            
            assert llm_response.content == "This is about machine learning"
            assert len(embedding) == 3072
            
            await llm_service.close()
            await embedding_service.close()
    
    @pytest.mark.asyncio
    async def test_error_propagation(self, full_config):
        """测试错误传播"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # 模拟API错误
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.json = AsyncMock(return_value={
                "error": {"message": "Invalid API key"}
            })
            mock_post.return_value.__aenter__.return_value = mock_response
            
            llm_service = LLMProviderService(full_config)
            
            # 错误应该正确传播
            with pytest.raises(Exception):
                await llm_service.generate(
                    messages=[{"role": "user", "content": "Test"}]
                )
            
            await llm_service.close()


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """端到端工作流测试"""
    
    @pytest.mark.asyncio
    async def test_paper_processing_workflow(self, sample_processed_paper, sample_global_analysis_map):
        """测试论文处理工作流"""
        # 这是一个简化的端到端测试
        # 在实际环境中，这会涉及所有组件的协调
        
        # 1. 模拟论文摄取完成
        processed_paper = sample_processed_paper
        assert processed_paper.metadata.title == "Attention Is All You Need"
        assert len(processed_paper.sections) > 0
        
        # 2. 模拟分析完成
        global_map = sample_global_analysis_map
        assert global_map.research_domain == "Natural Language Processing"
        assert len(global_map.key_contributions) > 0
        
        # 3. 模拟合成完成
        # 在真实场景中，这会调用SynthesisEngine
        article_sections = {
            "title": "深入解析：Attention Is All You Need",
            "introduction": "本文介绍了Transformer架构...",
            "key_insights": ["提出了自注意力机制", "摒弃了循环和卷积结构"],
            "deep_thinking": "Transformer的提出标志着NLP领域的重要进展..."
        }
        
        assert "title" in article_sections
        assert "introduction" in article_sections
        assert len(article_sections["key_insights"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
