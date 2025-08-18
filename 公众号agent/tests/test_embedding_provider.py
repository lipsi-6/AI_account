"""
嵌入提供商测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from modules.embedding_provider import (
    EmbeddingProviderService, OpenAIEmbeddingClient, EmbeddingRequest, EmbeddingResponse,
    EmbeddingAPIError, EmbeddingRateLimitError
)
from modules.config_manager import ConfigManager


class TestOpenAIEmbeddingClient:
    """OpenAI嵌入客户端测试"""
    
    @pytest.fixture
    def client(self):
        return OpenAIEmbeddingClient("test-api-key", "text-embedding-3-large")
    
    @pytest.mark.asyncio
    async def test_successful_embedding(self, client):
        """测试成功生成嵌入"""
        mock_response_data = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3] * 1024}  # 3072维向量
            ],
            "usage": {
                "total_tokens": 10
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_post.return_value.__aenter__.return_value = mock_response
            
            request = EmbeddingRequest(texts="Test text", normalize=True)
            response = await client.embed(request)
            
            assert isinstance(response, EmbeddingResponse)
            assert len(response.embeddings) == 1
            assert len(response.embeddings[0]) == 3072
            assert response.provider == "openai"
            assert response.tokens_used == 10
    
    @pytest.mark.asyncio
    async def test_batch_embedding(self, client):
        """测试批量嵌入"""
        mock_response_data = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3] * 1024},
                {"embedding": [0.4, 0.5, 0.6] * 1024}
            ],
            "usage": {
                "total_tokens": 20
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_post.return_value.__aenter__.return_value = mock_response
            
            request = EmbeddingRequest(texts=["Text 1", "Text 2"], normalize=True)
            response = await client.embed(request)
            
            assert len(response.embeddings) == 2
            assert response.tokens_used == 20
    
    def test_normalize_embeddings(self, client):
        """测试向量归一化"""
        embeddings = [[3.0, 4.0], [1.0, 0.0]]
        normalized = client._normalize_embeddings(embeddings)
        
        # 检查归一化后的向量长度为1
        for vec in normalized:
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 1e-6
    
    def test_cost_calculation(self, client):
        """测试成本计算"""
        cost = client._calculate_cost("text-embedding-3-large", 1000)
        expected_cost = 1000 * (0.00013 / 1000)
        assert abs(cost - expected_cost) < 1e-8
    
    def test_get_dimensions(self, client):
        """测试获取维度"""
        assert client.get_dimensions() == 3072
    
    def test_unsupported_model(self):
        """测试不支持的模型"""
        with pytest.raises(ValueError):
            OpenAIEmbeddingClient("test-key", "unsupported-model")


class TestEmbeddingProviderService:
    """嵌入提供商服务测试"""
    
    @pytest.fixture
    def mock_config(self):
        """创建模拟配置"""
        config = Mock(spec=ConfigManager)
        config.is_initialized.return_value = True
        config.embedding_settings = Mock()
        config.embedding_settings.provider = "openai"
        config.embedding_settings.model = "text-embedding-3-large"
        config.performance = Mock()
        config.performance.api_retry_attempts = 3
        config.performance.api_retry_delay_seconds = 1
        config.get_api_key.return_value = "test-api-key"
        
        return config
    
    def test_initialization_success(self, mock_config):
        """测试成功初始化"""
        service = EmbeddingProviderService(mock_config)
        
        assert service.client is not None
        assert service.client.model == "text-embedding-3-large"
    
    def test_initialization_no_api_key(self, mock_config):
        """测试没有API密钥的初始化"""
        mock_config.get_api_key.return_value = None
        
        with pytest.raises(ValueError):
            EmbeddingProviderService(mock_config)
    
    @pytest.mark.asyncio
    async def test_embed_text_success(self, mock_config):
        """测试单文本嵌入成功"""
        service = EmbeddingProviderService(mock_config)
        
        # 模拟客户端
        mock_client = AsyncMock()
        mock_response = EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            model="text-embedding-3-large",
            provider="openai",
            dimensions=3
        )
        mock_client.embed.return_value = mock_response
        service.client = mock_client
        
        embedding = await service.embed_text("Test text")
        
        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embed_texts_batch(self, mock_config):
        """测试批量文本嵌入"""
        service = EmbeddingProviderService(mock_config)
        
        # 模拟客户端
        mock_client = AsyncMock()
        mock_response = EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model="text-embedding-3-large",
            provider="openai",
            dimensions=3
        )
        mock_client.embed.return_value = mock_response
        service.client = mock_client
        
        embeddings = await service.embed_texts(["Text 1", "Text 2"])
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
    
    @pytest.mark.asyncio
    async def test_similarity_calculation(self, mock_config):
        """测试相似度计算"""
        service = EmbeddingProviderService(mock_config)
        
        # 模拟客户端返回单位向量
        mock_client = AsyncMock()
        mock_response = EmbeddingResponse(
            embeddings=[[1.0, 0.0], [0.0, 1.0]],  # 垂直向量
            model="text-embedding-3-large",
            provider="openai",
            dimensions=2
        )
        mock_client.embed.return_value = mock_response
        service.client = mock_client
        
        similarity = await service.similarity("Text 1", "Text 2")
        
        # 垂直向量的余弦相似度是0，映射后是0.5
        assert abs(similarity - 0.5) < 1e-6
    
    @pytest.mark.asyncio
    async def test_find_most_similar(self, mock_config):
        """测试查找最相似文本"""
        service = EmbeddingProviderService(mock_config)
        
        # 模拟客户端
        mock_client = AsyncMock()
        mock_response = EmbeddingResponse(
            embeddings=[
                [1.0, 0.0],  # 查询向量
                [1.0, 0.0],  # 完全相同
                [0.0, 1.0],  # 垂直
                [-1.0, 0.0]  # 相反
            ],
            model="text-embedding-3-large",
            provider="openai",
            dimensions=2
        )
        mock_client.embed.return_value = mock_response
        service.client = mock_client
        
        results = await service.find_most_similar(
            "query", ["candidate1", "candidate2", "candidate3"], top_k=2
        )
        
        assert len(results) == 2
        # 第一个应该是最相似的
        assert results[0][0] == "candidate1"
        assert results[0][1] > results[1][1]  # 相似度更高
    
    def test_cache_functionality(self, mock_config):
        """测试缓存功能"""
        service = EmbeddingProviderService(mock_config)
        
        # 添加缓存项
        service._embedding_cache["test_key"] = [0.1, 0.2, 0.3]
        
        stats = service.get_cache_stats()
        assert stats["cache_size"] == 1
        
        service.clear_cache()
        stats = service.get_cache_stats()
        assert stats["cache_size"] == 0
    
    def test_get_provider_info(self, mock_config):
        """测试获取提供商信息"""
        service = EmbeddingProviderService(mock_config)
        
        info = service.get_provider_info()
        
        assert info["provider"] == "openai"
        assert info["model"] == "text-embedding-3-large"
        assert "dimensions" in info


if __name__ == "__main__":
    pytest.main([__file__])
