"""
LLM提供商测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import aiohttp
import json

from modules.llm_provider import (
    LLMProviderService, OpenAIClient, LLMRequest, LLMResponse,
    LLMAPIError, LLMRateLimitError, LLMAuthenticationError
)
from modules.config_manager import ConfigManager


class TestOpenAIClient:
    """OpenAI客户端测试"""
    
    @pytest.fixture
    def client(self):
        return OpenAIClient("test-api-key", "gpt-4o")
    
    @pytest.mark.asyncio
    async def test_successful_generation(self, client):
        """测试成功生成"""
        mock_response_data = {
            "choices": [{
                "message": {"content": "Test response"},
                "finish_reason": "stop"
            }],
            "usage": {
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_post.return_value.__aenter__.return_value = mock_response
            
            request = LLMRequest(
                messages=[{"role": "user", "content": "Test message"}],
                temperature=0.7
            )
            
            response = await client.generate(request)
            
            assert isinstance(response, LLMResponse)
            assert response.content == "Test response"
            assert response.model == "gpt-4o"
            assert response.provider == "openai"
            assert response.tokens_used == 100
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, client):
        """测试API错误处理"""
        mock_error_response = {
            "error": {
                "message": "Invalid API key",
                "code": "invalid_api_key"
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.json = AsyncMock(return_value=mock_error_response)
            mock_post.return_value.__aenter__.return_value = mock_response
            
            request = LLMRequest(
                messages=[{"role": "user", "content": "Test message"}]
            )
            
            with pytest.raises(LLMAuthenticationError):
                await client.generate(request)
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, client):
        """测试限流处理"""
        mock_error_response = {
            "error": {
                "message": "Rate limit exceeded",
                "code": "rate_limit_exceeded"
            },
            "retry_after": 30
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_response.json = AsyncMock(return_value=mock_error_response)
            mock_post.return_value.__aenter__.return_value = mock_response
            
            request = LLMRequest(
                messages=[{"role": "user", "content": "Test message"}]
            )
            
            with pytest.raises(LLMRateLimitError) as exc_info:
                await client.generate(request)
            
            assert exc_info.value.retry_after == 30
    
    def test_cost_calculation(self, client):
        """测试成本计算"""
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500
        }
        
        cost = client._calculate_cost("gpt-4o", usage)
        
        # gpt-4o: input=0.005, output=0.015 per 1K tokens
        expected_cost = (1000/1000 * 0.005) + (500/1000 * 0.015)
        assert abs(cost - expected_cost) < 0.0001


class TestLLMProviderService:
    """LLM提供商服务测试"""
    
    @pytest.fixture
    def mock_config(self):
        """创建模拟配置"""
        config = Mock(spec=ConfigManager)
        config.is_initialized.return_value = True
        config.llm_settings = Mock()
        config.llm_settings.provider = "openai"
        config.llm_settings.primary_model = "gpt-4o"
        config.llm_settings.insight_model = "gpt-4o"
        config.performance = Mock()
        config.performance.api_retry_attempts = 3
        config.performance.api_retry_delay_seconds = 1
        config.get_api_key.return_value = "test-api-key"
        
        return config
    
    def test_initialization_success(self, mock_config):
        """测试成功初始化"""
        service = LLMProviderService(mock_config)
        
        assert service.primary_client is not None
        assert service.insight_client is not None
        assert service.primary_client.model == "gpt-4o"
    
    def test_initialization_no_api_key(self, mock_config):
        """测试没有API密钥的初始化"""
        mock_config.get_api_key.return_value = None
        
        with pytest.raises(ValueError):
            LLMProviderService(mock_config)
    
    @pytest.mark.asyncio
    async def test_generate_with_retry_success(self, mock_config):
        """测试带重试的生成成功"""
        service = LLMProviderService(mock_config)
        
        # 模拟客户端
        mock_client = AsyncMock()
        mock_response = LLMResponse(
            content="Test response",
            model="gpt-4o",
            provider="openai"
        )
        mock_client.generate.return_value = mock_response
        service.primary_client = mock_client
        
        response = await service.generate(
            messages=[{"role": "user", "content": "Test"}]
        )
        
        assert response.content == "Test response"
        mock_client.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_with_retry_failure(self, mock_config):
        """测试带重试的生成失败"""
        service = LLMProviderService(mock_config)
        
        # 模拟客户端总是失败
        mock_client = AsyncMock()
        mock_client.generate.side_effect = LLMAPIError("API Error")
        service.primary_client = mock_client
        
        with pytest.raises(Exception):
            await service.generate(
                messages=[{"role": "user", "content": "Test"}]
            )
        
        # 应该重试指定次数
        assert mock_client.generate.call_count == mock_config.performance.api_retry_attempts + 1
    
    @pytest.mark.asyncio
    async def test_close(self, mock_config):
        """测试关闭服务"""
        service = LLMProviderService(mock_config)
        
        # 模拟客户端
        mock_client = AsyncMock()
        service.primary_client = mock_client
        service.clients = {"openai": mock_client}
        
        await service.close()
        
        mock_client.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
