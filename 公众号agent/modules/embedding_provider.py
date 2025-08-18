"""
嵌入服务抽象层

提供统一的、生产级质量的文本嵌入服务，严禁TF-IDF或任何形式的伪嵌入。
支持多种SOTA嵌入模型，实现健壮的重试机制和错误处理。
"""

import asyncio
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Tuple
from enum import Enum
import aiohttp
import numpy as np
import json

from .config_manager import ConfigManager
from .logger import get_logger, log_api_call, log_performance


class EmbeddingProvider(Enum):
    """嵌入提供商枚举"""
    OPENAI = "openai"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"  # 本地模型
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # 本地模型


@dataclass
class EmbeddingRequest:
    """嵌入请求数据类"""
    texts: Union[str, List[str]]
    model: Optional[str] = None
    normalize: bool = True
    batch_size: Optional[int] = None


@dataclass
class EmbeddingResponse:
    """嵌入响应数据类"""
    embeddings: List[List[float]]
    model: str
    provider: str
    dimensions: int
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    raw_response: Optional[Dict[str, Any]] = None


class EmbeddingError(Exception):
    """嵌入相关错误基类"""
    pass


class EmbeddingAPIError(EmbeddingError):
    """API调用错误"""
    def __init__(self, message: str, status_code: Optional[int] = None, provider: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider


class EmbeddingRateLimitError(EmbeddingAPIError):
    """API限流错误"""
    def __init__(self, message: str, retry_after: Optional[int] = None, provider: Optional[str] = None):
        super().__init__(message, 429, provider)
        self.retry_after = retry_after


class BaseEmbeddingClient(ABC):
    """嵌入客户端基类"""
    
    def __init__(self, model: str):
        self.model = model
        self.logger = get_logger(f"embedding_client_{self.__class__.__name__.lower()}")
    
    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """生成嵌入向量"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """获取提供商名称"""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """获取嵌入维度"""
        pass


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI嵌入客户端（支持自定义 base_url 以对接 OpenAI 兼容中转服务）"""
    
    # 模型维度映射
    MODEL_DIMENSIONS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536
    }
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-large", base_url: Optional[str] = None):
        super().__init__(model)
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        
        if model not in self.MODEL_DIMENSIONS:
            raise ValueError(f"Unsupported OpenAI embedding model: {model}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=aiohttp.ClientTimeout(total=300)
            )
        return self.session
    
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """生成嵌入向量"""
        start_time = time.time()
        
        # 标准化输入
        texts = request.texts if isinstance(request.texts, list) else [request.texts]
        
        # 分批处理大量文本
        batch_size = request.batch_size or 100  # OpenAI推荐的批量大小
        all_embeddings = []
        total_tokens = 0
        total_cost = 0.0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_response = await self._embed_batch(batch_texts, request.model or self.model)
            
            all_embeddings.extend(batch_response["embeddings"])
            total_tokens += batch_response["tokens"]
            total_cost += batch_response["cost"]
        
        # 归一化嵌入向量
        if request.normalize:
            all_embeddings = self._normalize_embeddings(all_embeddings)
        
        duration = time.time() - start_time
        log_api_call(
            provider="openai",
            model=request.model or self.model,
            tokens_used=total_tokens,
            cost=total_cost,
            success=True
        )
        log_performance("openai_embedding", duration, 
                       texts_count=len(texts), tokens=total_tokens)
        
        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=request.model or self.model,
            provider="openai",
            dimensions=self.get_dimensions(),
            tokens_used=total_tokens,
            cost_usd=total_cost
        )
    
    async def _embed_batch(self, texts: List[str], model: str) -> Dict[str, Any]:
        """批量生成嵌入"""
        payload = {
            "model": model,
            "input": texts,
            "encoding_format": "float"
        }
        
        session = await self._get_session()
        
        async with session.post(f"{self.base_url}/embeddings", json=payload) as response:
            response_data = await response.json()
            
            if response.status != 200:
                await self._handle_error_response(response.status, response_data)
            
            # 提取嵌入向量
            embeddings = [item["embedding"] for item in response_data["data"]]
            
            # 计算token使用量和成本（兼容不同字段）
            usage = response_data.get("usage", {}) or {}
            tokens_used = 0
            try:
                if "total_tokens" in usage:
                    tokens_used = int(usage.get("total_tokens", 0))
                else:
                    prompt_tokens = int(usage.get("prompt_tokens", 0))
                    completion_tokens = int(usage.get("completion_tokens", 0))
                    tokens_used = prompt_tokens + completion_tokens
            except Exception:
                tokens_used = 0
            cost_usd = self._calculate_cost(model, tokens_used)
            
            return {
                "embeddings": embeddings,
                "tokens": tokens_used,
                "cost": cost_usd
            }
    
    async def _handle_error_response(self, status_code: int, response_data: Dict[str, Any]) -> None:
        """处理错误响应"""
        error_info = response_data.get("error", {})
        error_message = error_info.get("message", "Unknown error")
        
        if status_code == 401:
            raise EmbeddingAPIError(f"OpenAI authentication failed: {error_message}", status_code, "openai")
        elif status_code == 429:
            retry_after = int(response_data.get("retry_after", 60))
            raise EmbeddingRateLimitError(f"OpenAI rate limit exceeded: {error_message}", retry_after, "openai")
        else:
            raise EmbeddingAPIError(f"OpenAI API error: {error_message}", status_code, "openai")
    
    def _calculate_cost(self, model: str, tokens: int) -> float:
        """计算调用成本（美元）"""
        # OpenAI嵌入定价（每1000 tokens）
        pricing = {
            "text-embedding-3-large": 0.00013,
            "text-embedding-3-small": 0.00002,
            "text-embedding-ada-002": 0.0001
        }
        
        price_per_token = pricing.get(model, 0.0) / 1000
        return tokens * price_per_token
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """归一化嵌入向量"""
        normalized = []
        for embedding in embeddings:
            vec = np.array(embedding)
            norm = np.linalg.norm(vec)
            if norm > 0:
                normalized.append((vec / norm).tolist())
            else:
                normalized.append(vec.tolist())
        return normalized
    
    def get_provider_name(self) -> str:
        return "openai"
    
    def get_dimensions(self) -> int:
        return self.MODEL_DIMENSIONS[self.model]
    
    async def close(self):
        """关闭会话"""
        if self.session and not self.session.closed:
            await self.session.close()

    def __del__(self):
        try:
            if getattr(self, "session", None) and not self.session.closed:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.session.close())
                    else:
                        loop.run_until_complete(self.session.close())
                except Exception:
                    try:
                        asyncio.run(self.session.close())
                    except Exception:
                        pass
        except Exception:
            pass


class SentenceTransformersEmbeddingClient(BaseEmbeddingClient):
    """Sentence Transformers本地嵌入客户端"""
    
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        super().__init__(model)
        self._model_instance = None
        self._dimensions = None
        
        # 延迟加载模型：在首次 embed 调用时惰性加载，避免无事件循环情形
    
    async def _load_model(self):
        """异步加载模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.logger.info(f"Loading Sentence Transformers model: {self.model}")
            # 在后台线程加载重型模型，避免阻塞事件循环
            self._model_instance = await asyncio.to_thread(SentenceTransformer, self.model)
            self._dimensions = await asyncio.to_thread(self._model_instance.get_sentence_embedding_dimension)
            self.logger.info(f"Model loaded successfully, dimensions: {self._dimensions}")
            
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers library not installed. "
                "Please install it with: pip install sentence-transformers"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load Sentence Transformers model: {e}")
    
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """生成嵌入向量"""
        if self._model_instance is None:
            await self._load_model()
        
        start_time = time.time()
        
        # 标准化输入
        texts = request.texts if isinstance(request.texts, list) else [request.texts]
        
        try:
            # 生成嵌入向量
            embeddings = self._model_instance.encode(
                texts,
                normalize_embeddings=request.normalize,
                show_progress_bar=len(texts) > 100
            )
            
            # 转换为列表格式
            if len(embeddings.shape) == 1:
                embeddings = [embeddings.tolist()]
            else:
                embeddings = embeddings.tolist()
            
            duration = time.time() - start_time
            log_performance("sentence_transformers_embedding", duration, 
                           texts_count=len(texts))
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=self.model,
                provider="sentence_transformers",
                dimensions=self.get_dimensions(),
                tokens_used=None,  # 本地模型不计算token
                cost_usd=0.0  # 本地模型无成本
            )
            
        except Exception as e:
            raise EmbeddingError(f"Sentence Transformers embedding failed: {e}")
    
    def get_provider_name(self) -> str:
        return "sentence_transformers"
    
    def get_dimensions(self) -> int:
        if self._dimensions is None:
            raise EmbeddingError("Model not loaded yet")
        return self._dimensions
    
    async def close(self):
        """清理资源"""
        if self._model_instance is not None:
            del self._model_instance
            self._model_instance = None


class GoogleEmbeddingClient(BaseEmbeddingClient):
    """Google Generative AI 嵌入客户端（text-embedding-004）。"""

    # 常见模型维度（根据公开资料，text-embedding-004 输出 768 维）
    MODEL_DIMENSIONS = {
        "text-embedding-004": 768,
        # 历史/兼容别名
        "models/text-embedding-004": 768,
        "textembedding-gecko-002": 768,
        "models/embedding-001": 768,
    }

    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        super().__init__(model)
        self.api_key = api_key
        self._configured = False
        self._dimensions = self.MODEL_DIMENSIONS.get(model, 768)

    def _ensure_client(self) -> None:
        if not self._configured:
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise EmbeddingAPIError(
                    "google-generativeai library not installed. Please install: pip install google-generativeai",
                    provider="google",
                ) from e
            genai.configure(api_key=self.api_key)
            self._genai = genai
            self._configured = True

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self._ensure_client()
        texts = request.texts if isinstance(request.texts, list) else [request.texts]

        embeddings: List[List[float]] = []
        tokens_used = None
        try:
            # 逐条生成，避免旧版 SDK 无 batch 接口的问题
            for text in texts:
                def _embed_one():
                    # genai.embed_content 返回 { 'embedding': { 'values': [...] } }
                    model_name = self.model
                    if not (model_name.startswith("models/") or model_name.startswith("tunedModels/")):
                        model_name = f"models/{model_name}"
                    # 为不同长度选择合适的 task_type，提升兼容性
                    task_type = "retrieval_document" if len(text or "") > 200 else "retrieval_query"
                    return self._genai.embed_content(model=model_name, content=text, task_type=task_type)

                resp = await asyncio.to_thread(_embed_one)
                vec: List[float] = []
                try:
                    # 兼容多种返回结构：
                    # 1) {'embedding': {'values': [...]}}
                    # 2) {'embedding': [...]}（部分版本）
                    # 3) resp.embedding.values / resp.embedding
                    if isinstance(resp, dict):
                        emb = resp.get("embedding")
                        if isinstance(emb, list):
                            vec = [float(x) for x in emb]
                        elif isinstance(emb, dict):
                            vals = emb.get("values") or emb.get("embedding") or []
                            if isinstance(vals, list):
                                vec = [float(x) for x in vals]
                    else:
                        emb_attr = getattr(resp, "embedding", None)
                        if isinstance(emb_attr, list):
                            vec = [float(x) for x in emb_attr]
                        else:
                            vals = getattr(emb_attr, "values", None)
                            if isinstance(vals, list):
                                vec = [float(x) for x in vals]
                except Exception:
                    # 忽略解析错误，进入回退逻辑
                    vec = []
                if not vec:
                    # 回退到兼容模型 models/embedding-001
                    def _embed_one_fallback():
                        return self._genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")
                    fb = await asyncio.to_thread(_embed_one_fallback)
                    emb = None
                    if isinstance(fb, dict):
                        emb = fb.get("embedding")
                        if isinstance(emb, list):
                            vec = [float(x) for x in emb]
                        elif isinstance(emb, dict):
                            vals = emb.get("values") or emb.get("embedding") or []
                            if isinstance(vals, list):
                                vec = [float(x) for x in vals]
                    else:
                        emb_attr = getattr(fb, "embedding", None)
                        if isinstance(emb_attr, list):
                            vec = [float(x) for x in emb_attr]
                        else:
                            vals = getattr(emb_attr, "values", None)
                            if isinstance(vals, list):
                                vec = [float(x) for x in vals]
                    if not vec:
                        raise EmbeddingAPIError("Empty embedding returned from Google", provider="google")
                embeddings.append(vec)

            # 归一化
            if request.normalize and embeddings:
                embeddings = self._normalize_embeddings(embeddings)

            # 维度更新
            try:
                self._dimensions = len(embeddings[0])
            except Exception:
                pass

            return EmbeddingResponse(
                embeddings=embeddings,
                model=request.model or self.model,
                provider="google",
                dimensions=self.get_dimensions(),
                tokens_used=tokens_used,
                cost_usd=None,
            )
        except EmbeddingAPIError:
            raise
        except Exception as e:
            raise EmbeddingAPIError(f"Google embedding failed: {e}", provider="google")

    def get_provider_name(self) -> str:
        return "google"

    def get_dimensions(self) -> int:
        return int(self._dimensions or self.MODEL_DIMENSIONS.get(self.model, 768))
    
    async def close(self):
        return


class EmbeddingProviderService:
    """
    嵌入提供商服务
    
    统一的嵌入接口，支持多种SOTA嵌入模型，实现健壮的重试机制和缓存。
    严禁TF-IDF或任何形式的伪嵌入作为备用方案。
    """
    
    def __init__(self, config: ConfigManager):
        """
        初始化嵌入提供商服务
        
        Args:
            config: 配置管理器
        """
        if not config.is_initialized():
            raise ValueError("ConfigManager must be initialized")
        
        self.config = config
        self.logger = get_logger("embedding_provider_service")
        self.client: Optional[BaseEmbeddingClient] = None
        self._embedding_cache: Dict[str, List[float]] = {}
        # 可选：基于 Cross-Encoder 的重排序器（仅在检索时使用，不影响向量生成）
        self._reranker = None
        self._reranker_model_name: Optional[str] = None
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """初始化嵌入客户端"""
        embedding_config = self.config.embedding_settings
        
        provider = embedding_config.provider.lower()
        model = embedding_config.model
        
        if provider == "openai":
            api_key = self.config.get_api_key("openai") or self.config.get_api_key("gemini") or self.config.get_api_key("google_gemini")
            if not api_key:
                raise ValueError("OpenAI API key is required for OpenAI embedding provider")
            base_url = None
            try:
                base_url = getattr(getattr(self.config, "embedding_settings", None), "base_url", None) or getattr(getattr(self.config, "llm_settings", None), "base_url", None)
            except Exception:
                base_url = None
            self.client = OpenAIEmbeddingClient(api_key, model, base_url=base_url)
        elif provider == "google":
            api_key = self.config.get_api_key("google") or self.config.get_api_key("google_gemini")
            if not api_key:
                raise ValueError("Google Gemini API key is required for Google embedding provider")
            self.client = GoogleEmbeddingClient(api_key, model)
            
        elif provider == "sentence_transformers":
            self.client = SentenceTransformersEmbeddingClient(model)
            
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        
        if not self.client:
            raise ValueError(f"Failed to initialize embedding client for provider: {provider}")
        
        self.logger.info(f"Initialized embedding service with provider: {provider}, model: {model}")
        # 初始化可选重排序器（不在构造函数期间执行耗时加载，仅记录模型名）
        try:
            es = self.config.embedding_settings
            if getattr(es, "enable_reranker", False):
                # 记录模型名，真正加载时惰性初始化
                self._reranker_model_name = (
                    es.reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                self.logger.info(
                    f"Reranker enabled (lazy): {self._reranker_model_name}"
                )
        except Exception as e:
            self.logger.warning(f"Reranker configuration ignored: {e}")
    
    async def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """
        嵌入单个文本
        
        Args:
            text: 要嵌入的文本
            use_cache: 是否使用缓存
            
        Returns:
            嵌入向量
        """
        # 检查缓存
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                self.logger.debug(f"Cache hit for text embedding")
                return self._embedding_cache[cache_key]
        
        # 生成嵌入
        request = EmbeddingRequest(texts=text, normalize=True)
        response = await self._embed_with_retry(request)
        
        embedding = response.embeddings[0]
        
        # 缓存结果
        if use_cache:
            cache_key = self._get_cache_key(text)
            self._embedding_cache[cache_key] = embedding
        
        return embedding

    async def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        使用 Cross-Encoder 对候选文档进行重排序。

        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 可选，返回前K个（默认返回全部）

        Returns:
            列表 [(index_in_candidates, score)]，按分数降序
        """
        if not candidates:
            return []
        # 未启用时，直接返回等权分数，保持外部流程稳定
        try:
            es = self.config.embedding_settings
            if not getattr(es, "enable_reranker", False):
                return [(i, 0.0) for i in range(len(candidates))][: (top_k or len(candidates))]
        except Exception:
            return [(i, 0.0) for i in range(len(candidates))][: (top_k or len(candidates))]

        try:
            # 惰性加载重排序模型
            if self._reranker is None:
                await self._load_reranker()
            # 组织 (query, doc) 成对输入
            pairs = [(query, doc) for doc in candidates]
            scores = await asyncio.to_thread(self._reranker.predict, pairs)
            # 打包索引和分数
            indexed = list(enumerate([float(s) for s in scores]))
            indexed.sort(key=lambda x: x[1], reverse=True)
            if top_k is not None:
                indexed = indexed[: top_k]
            return indexed
        except Exception as e:
            self.logger.warning(f"Rerank failed, fallback to identity order: {e}")
            return [(i, 0.0) for i in range(len(candidates))][: (top_k or len(candidates))]

    async def _load_reranker(self) -> None:
        """惰性加载 Cross-Encoder 模型（在后台线程，避免阻塞事件循环）。"""
        if self._reranker is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            model_name = self._reranker_model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.logger.info(f"Loading cross-encoder reranker: {model_name}")
            self._reranker = await asyncio.to_thread(CrossEncoder, model_name)
            self.logger.info("Reranker loaded successfully")
        except ImportError as e:
            self.logger.warning(
                "sentence-transformers not installed; reranker disabled."
            )
            self._reranker = None
        except Exception as e:
            self.logger.warning(f"Failed to load reranker: {e}")
            self._reranker = None
    
    async def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        批量嵌入文本
        
        Args:
            texts: 要嵌入的文本列表
            batch_size: 批处理大小
            
        Returns:
            嵌入向量列表
        """
        if not texts:
            return []
        
        request = EmbeddingRequest(
            texts=texts,
            normalize=True,
            batch_size=batch_size
        )
        
        response = await self._embed_with_retry(request)
        return response.embeddings
    
    async def _embed_with_retry(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        带重试机制的嵌入生成
        
        Args:
            request: 嵌入请求
            
        Returns:
            嵌入响应
        """
        if not self.client:
            raise EmbeddingError("No embedding client available")
        
        max_retries = self.config.performance.api_retry_attempts
        base_delay = self.config.performance.api_retry_delay_seconds
        
        for attempt in range(max_retries + 1):
            try:
                return await self.client.embed(request)
                
            except EmbeddingRateLimitError as e:
                if attempt == max_retries:
                    raise
                
                # 使用API返回的重试时间或指数退避
                delay = e.retry_after or (base_delay * (2 ** attempt))
                self.logger.warning(f"Rate limit hit, retrying in {delay} seconds (attempt {attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(delay)
                
            except (EmbeddingAPIError, Exception) as e:
                if attempt == max_retries:
                    raise EmbeddingError(f"Embedding generation failed after {max_retries + 1} attempts: {e}")
                
                # 指数退避
                delay = base_delay * (2 ** attempt)
                self.logger.warning(f"Embedding error, retrying in {delay} seconds (attempt {attempt + 1}/{max_retries + 1}): {e}")
                await asyncio.sleep(delay)
        
        raise EmbeddingError("Max retries exceeded")
    
    def _get_cache_key(self, text: str) -> str:
        """
        生成缓存键
        
        Args:
            text: 文本
            
        Returns:
            缓存键
        """
        # 使用文本内容和模型信息生成唯一键
        content = f"{self.client.model}:{text}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get_dimensions(self) -> int:
        """获取嵌入维度"""
        if not self.client:
            raise EmbeddingError("No embedding client available")
        return self.client.get_dimensions()
    
    def get_provider_info(self) -> Dict[str, Any]:
        """获取提供商信息"""
        if not self.client:
            return {}
        
        info: Dict[str, Any] = {
            "provider": self.client.get_provider_name(),
            "model": self.client.model,
        }
        try:
            info["dimensions"] = self.client.get_dimensions()
        except Exception as e:
            # 维度未知（例如本地模型尚未加载），以 None 表示
            info["dimensions"] = None
        return info
    
    def clear_cache(self) -> None:
        """清除嵌入缓存"""
        self._embedding_cache.clear()
        self.logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "cache_size": len(self._embedding_cache),
            "cache_keys": list(self._embedding_cache.keys())[:10]  # 只显示前10个键
        }
    
    async def similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            余弦相似度 (0-1之间)
        """
        embeddings = await self.embed_texts([text1, text2])
        
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        
        # 计算余弦相似度
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # 将相似度从[-1, 1]映射到[0, 1]
        return (cosine_sim + 1) / 2
    
    async def find_most_similar(self, query: str, candidates: List[str], top_k: int = 5) -> List[tuple]:
        """
        找到最相似的候选文本
        
        Args:
            query: 查询文本
            candidates: 候选文本列表
            top_k: 返回前k个最相似的
            
        Returns:
            (候选文本, 相似度分数) 的列表，按相似度降序排列
        """
        if not candidates:
            return []
        
        # 生成所有文本的嵌入
        all_texts = [query] + candidates
        embeddings = await self.embed_texts(all_texts)
        
        query_embedding = np.array(embeddings[0])
        candidate_embeddings = np.array(embeddings[1:])
        
        # 计算相似度
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            cosine_sim = np.dot(query_embedding, candidate_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
            )
            # 将相似度从[-1, 1]映射到[0, 1]
            similarity = (cosine_sim + 1) / 2
            similarities.append((candidates[i], similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    async def close(self) -> None:
        """关闭嵌入服务"""
        if self.client:
            await self.client.close()
        
        self.clear_cache()
        self.logger.info("Embedding service closed")
