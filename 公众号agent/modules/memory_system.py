"""
SOTA级别混合记忆系统

实现基于ChromaDB和NetworkX的混合记忆架构：
- 向量数据库 (Episodic Memory) - 支持语义检索
- 知识图谱 (Conceptual Memory) - 支持关系推理

严格遵循依赖注入原则，无副作用构造函数，显式异步初始化。
实现真正的语义检索，绝不使用伪实现。
"""

import asyncio
import json
import time
import hashlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from enum import Enum
import uuid

import numpy as np
import networkx as nx
from collections import defaultdict

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# 可选：Qdrant 远程向量后端
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance as QDistance,
        VectorParams,
        PointStruct,
        Filter as QFilter,
        FieldCondition,
        MatchValue,
        ExtendedPointId,
        PayloadSchemaType,
    )
    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False

from .config_manager import ConfigManager
from .logger import get_logger, log_memory_operation, log_performance
from .embedding_provider import EmbeddingProviderService


class MemoryType(Enum):
    """记忆类型枚举"""
    EPISODIC = "episodic"  # 情节记忆（向量数据库）
    CONCEPTUAL = "conceptual"  # 概念记忆（知识图谱）
    SEMANTIC = "semantic"  # 语义记忆（混合）


class NodeType(Enum):
    """知识图谱节点类型"""
    PAPER = "paper"
    AUTHOR = "author"
    CONCEPT = "concept"
    TOPIC = "topic"
    VENUE = "venue"
    ORGANIZATION = "organization"
    KEYWORD = "keyword"


class EdgeType(Enum):
    """知识图谱边类型"""
    AUTHORED = "authored"
    CITED = "cited"
    RELATED_TO = "related_to"
    CONTAINS = "contains"
    SIMILAR_TO = "similar_to"
    INFLUENCES = "influences"
    PUBLISHED_IN = "published_in"
    WORKS_AT = "works_at"


@dataclass
class MemoryEntry:
    """记忆条目基类"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: Optional[List[float]] = None
    importance_score: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class EpisodicMemoryEntry(MemoryEntry):
    """情节记忆条目"""
    context: Optional[str] = None
    source: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ConceptualMemoryEntry:
    """概念记忆条目（知识图谱节点）"""
    id: str
    node_type: NodeType
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    importance_score: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None


@dataclass
class MemorySearchQuery:
    """记忆搜索查询"""
    text: str
    memory_types: List[MemoryType] = field(default_factory=lambda: [MemoryType.EPISODIC, MemoryType.CONCEPTUAL])
    limit: int = 10
    similarity_threshold: float = 0.7
    filters: Optional[Dict[str, Any]] = None
    include_related: bool = True
    search_depth: int = 2


@dataclass
class MemorySearchResult:
    """记忆搜索结果"""
    entry: Union[EpisodicMemoryEntry, ConceptualMemoryEntry]
    similarity_score: float
    memory_type: MemoryType
    explanation: Optional[str] = None
    related_entries: List["MemorySearchResult"] = field(default_factory=list)


class MemoryError(Exception):
    """记忆系统错误基类"""
    pass


class MemoryInitializationError(MemoryError):
    """记忆系统初始化错误"""
    pass


class MemoryStorageError(MemoryError):
    """记忆存储错误"""
    pass


class MemoryRetrievalError(MemoryError):
    """记忆检索错误"""
    pass


class BaseMemoryBackend(ABC):
    """记忆后端基类"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化后端"""
        pass
    
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """存储记忆条目"""
        pass
    
    @abstractmethod
    async def retrieve(self, query: MemorySearchQuery) -> List[MemorySearchResult]:
        """检索记忆条目"""
        pass
    
    @abstractmethod
    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """更新记忆条目"""
        pass
    
    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """删除记忆条目"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """关闭后端"""
        pass


class ChromaDBEpisodicBackend(BaseMemoryBackend):
    """ChromaDB情节记忆后端"""
    
    def __init__(self, config: ConfigManager, embedding_service: EmbeddingProviderService):
        """
        初始化ChromaDB后端
        
        Args:
            config: 配置管理器
            embedding_service: 嵌入服务
        """
        if not CHROMADB_AVAILABLE:
            raise MemoryInitializationError(
                "ChromaDB not available. Please install: pip install chromadb"
            )
        
        self.config = config
        self.embedding_service = embedding_service
        self.logger = get_logger("chromadb_episodic_backend")
        
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化ChromaDB"""
        if self._initialized:
            return
        
        try:
            vector_db_config = self.config.vector_db
            db_path = Path(vector_db_config.path).resolve()
            db_path.mkdir(parents=True, exist_ok=True)
            
            # 初始化ChromaDB客户端
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 获取或创建集合（固定使用 cosine 度量空间，稳定距离-相似度映射）
            collection_name = vector_db_config.collection_name

            try:
                self.collection = self.client.get_collection(name=collection_name)
                self.logger.info(f"Connected to existing ChromaDB collection: {collection_name}")
            except ValueError:
                # 集合不存在，创建新集合
                try:
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        metadata={
                            "description": "Deep Scholar AI Episodic Memory",
                            "hnsw:space": "cosine"
                        }
                    )
                except TypeError:
                    # 兼容旧版本 chroma 不支持在 metadata 中设置 hnsw:space
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"description": "Deep Scholar AI Episodic Memory"}
                    )
                self.logger.info(f"Created new ChromaDB collection: {collection_name}")
            
            self._initialized = True
            self.logger.info("ChromaDB episodic backend initialized successfully")
            
        except Exception as e:
            raise MemoryInitializationError(f"Failed to initialize ChromaDB backend: {e}")
    
    async def store(self, entry: EpisodicMemoryEntry) -> str:
        """存储情节记忆条目"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # 生成嵌入向量
            if entry.embedding is None:
                entry.embedding = await self.embedding_service.embed_text(entry.content)
            
            # 准备元数据
            metadata = {
                "timestamp": entry.timestamp.isoformat(),
                "importance_score": entry.importance_score,
                "access_count": entry.access_count,
                "source": entry.source or "",
                "context": entry.context or "",
                "tags": json.dumps(entry.tags) if entry.tags else "[]",
                **entry.metadata
            }
            # 注入嵌入版本信息（模型/维度/版本）
            try:
                provider_info = self.embedding_service.get_provider_info() or {}
                if provider_info:
                    if provider_info.get("model"):
                        metadata["embedding_model"] = provider_info.get("model")
                        # 以模型名作为默认版本标识，后续可改为显式版本
                        metadata["embedding_version"] = str(provider_info.get("model"))
                    if provider_info.get("dimensions") is not None:
                        metadata["embedding_dim"] = int(provider_info.get("dimensions") or 0)
            except Exception:
                pass
            
            # 幂等写入：若ID已存在，则只更新元数据（避免重复文档与embedding）
            try:
                existing = self.collection.get(ids=[entry.id], include=["metadatas", "documents"])  # type: ignore[arg-type]
            except Exception:
                existing = {"ids": [[]]}

            exists = bool(existing.get("ids") and existing["ids"][0])

            if exists:
                # 合并元数据并更新
                try:
                    current_meta = (existing.get("metadatas") or [{}])[0] or {}
                    # 保护性合并：新值覆盖旧值
                    new_meta = {**current_meta, **metadata}
                    new_meta["updated_at"] = datetime.now(timezone.utc).isoformat()
                    # 重要性与访问计数取最大/累加更合理，这里采用覆盖重要性、保留原访问计数增长
                    if "access_count" in current_meta:
                        try:
                            new_meta["access_count"] = int(current_meta.get("access_count", 0))
                        except Exception:
                            pass
                    self.collection.update(ids=[entry.id], metadatas=[new_meta])
                except Exception as _e:
                    # 如果更新失败，回退为完整替换
                    self.collection.update(
                        ids=[entry.id],
                        metadatas=[metadata],
                    )
            else:
                # 首次写入
                self.collection.add(
                    ids=[entry.id],
                    embeddings=[entry.embedding],
                    documents=[entry.content],
                    metadatas=[metadata]
                )
            
            duration = time.time() - start_time
            log_memory_operation(
                operation="store",
                memory_type="episodic",
                success=True,
                entry_id=entry.id,
                duration=duration
            )
            log_performance("chromadb_store", duration)
            
            self.logger.debug(f"Stored episodic memory entry: {entry.id}")
            return entry.id
            
        except Exception as e:
            log_memory_operation(
                operation="store",
                memory_type="episodic",
                success=False,
                error_message=str(e),
                entry_id=entry.id
            )
            raise MemoryStorageError(f"Failed to store episodic memory: {e}")
    
    async def retrieve(self, query: MemorySearchQuery) -> List[MemorySearchResult]:
        """检索情节记忆"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # 生成查询嵌入
            query_embedding = await self.embedding_service.embed_text(query.text)
            
            # 构建过滤条件
            where_filter = {}
            if query.filters:
                for key, value in query.filters.items():
                    if key in ["source", "context"]:
                        where_filter[key] = value
                    elif key == "tags":
                        # Chroma 对列表的直接包含查询支持有限，这里不在 where 端做不可靠的匹配
                        # 将在检索后进行客户端侧过滤（见下游 post_filter）
                        where_filter["__post_filter_tags"] = value if isinstance(value, list) else [value]
            
            # 执行相似度搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=query.limit,
                # 仅传递真实可用的 where 条件
                where={k: v for k, v in where_filter.items() if k not in ("tags", "__post_filter_tags")} or None,
                include=["documents", "metadatas", "distances"]
            )
            
            # 转换结果
            memory_results = []
            if results["ids"] and results["ids"][0]:
                for i, entry_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    # 将余弦距离 d ∈ [0,2] 线性映射到相似度 s ∈ [0,1]
                    # s = 1 - 0.5 * d
                    try:
                        similarity = 1.0 - 0.5 * float(distance)
                    except Exception:
                        similarity = 0.0
                    similarity = max(0.0, min(1.0, similarity))
                    
                    if similarity >= query.similarity_threshold:
                        metadata = results["metadatas"][0][i]
                        document = results["documents"][0][i]
                        
                        # 客户端侧 tags 二次过滤（如果请求中包含 tags 条件）
                        requested_tags = where_filter.get("__post_filter_tags")
                        if requested_tags:
                            try:
                                entry_tags = json.loads(metadata.get("tags", "[]"))
                            except Exception:
                                entry_tags = []
                            # 只要包含任意一个请求标签即可通过
                            if not any(t in entry_tags for t in requested_tags):
                                continue

                        # 重建EpisodicMemoryEntry
                        entry = EpisodicMemoryEntry(
                            id=entry_id,
                            content=document,
                            metadata={k: v for k, v in metadata.items() 
                                    if k not in ["timestamp", "importance_score", "access_count", "source", "context", "tags"]},
                            timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now(timezone.utc).isoformat())),
                            importance_score=metadata.get("importance_score", 1.0),
                            access_count=metadata.get("access_count", 0),
                            source=metadata.get("source"),
                            context=metadata.get("context"),
                            tags=json.loads(metadata.get("tags", "[]"))
                        )
                        
                        # 更新访问统计
                        await self._update_access_stats(entry_id)
                        
                        memory_results.append(MemorySearchResult(
                            entry=entry,
                            similarity_score=similarity,
                            memory_type=MemoryType.EPISODIC,
                            explanation=f"Semantic similarity: {similarity:.3f}"
                        ))
            
            duration = time.time() - start_time
            log_memory_operation(
                operation="retrieve",
                memory_type="episodic",
                success=True,
                query=query.text,
                results_count=len(memory_results),
                duration=duration
            )
            log_performance("chromadb_retrieve", duration, results_count=len(memory_results))
            
            return memory_results
            
        except Exception as e:
            log_memory_operation(
                operation="retrieve",
                memory_type="episodic",
                success=False,
                error_message=str(e),
                query=query.text
            )
            raise MemoryRetrievalError(f"Failed to retrieve episodic memory: {e}")

    async def list_entries(self, max_items: int = 5000) -> List[EpisodicMemoryEntry]:
        """列出部分或全部情节记忆（用于巩固/压缩作业）。
        为控制资源，默认最多返回5000条；实际按集合规模裁剪。
        """
        if not self._initialized:
            await self.initialize()

        try:
            count = 0
            total = 0
            try:
                total = int(self.collection.count())
            except Exception:
                total = max_items

            target = min(max_items, total if total > 0 else max_items)
            page_size = 1000
            offset = 0
            entries: List[EpisodicMemoryEntry] = []

            while offset < target:
                limit = min(page_size, target - offset)
                try:
                    batch = self.collection.get(
                        include=["metadatas", "documents"],
                        limit=limit,
                        offset=offset,
                    )
                except TypeError:
                    # 旧版 chroma 不支持 offset/limit -> 退化为一次性获取
                    batch = self.collection.get(include=["metadatas", "documents"])  # type: ignore[assignment]
                ids = batch.get("ids") or []
                docs = batch.get("documents") or []
                metas = batch.get("metadatas") or []
                if not ids:
                    break
                # 对齐批次字段
                for i, entry_id in enumerate(ids):
                    try:
                        metadata = metas[i] if i < len(metas) else {}
                        document = docs[i] if i < len(docs) else ""
                        entry = EpisodicMemoryEntry(
                            id=entry_id,
                            content=document,
                            metadata={k: v for k, v in (metadata or {}).items()
                                      if k not in ["timestamp", "importance_score", "access_count", "source", "context", "tags"]},
                            timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now(timezone.utc).isoformat())) if isinstance(metadata, dict) else datetime.now(timezone.utc),
                            importance_score=float((metadata or {}).get("importance_score", 1.0) or 1.0),
                            access_count=int((metadata or {}).get("access_count", 0) or 0),
                            source=(metadata or {}).get("source"),
                            context=(metadata or {}).get("context"),
                            tags=json.loads((metadata or {}).get("tags", "[]")) if isinstance((metadata or {}).get("tags", "[]"), str) else ((metadata or {}).get("tags") or []),
                        )
                        entries.append(entry)
                        count += 1
                        if count >= target:
                            break
                    except Exception:
                        continue
                if count >= target:
                    break
                offset += len(ids)

            return entries
        except Exception as e:
            self.logger.warning(f"Failed to list episodic entries: {e}")
            return []

    async def bump_importance(self, entry_id: str, delta: float) -> bool:
        """根据反馈调整情节记忆的重要性分数。"""
        if not self._initialized:
            await self.initialize()
        try:
            result = self.collection.get(ids=[entry_id], include=["metadatas"])
            if not result.get("ids") or not result["ids"][0]:
                return False
            meta = (result.get("metadatas") or [{}])[0] or {}
            cur = float(meta.get("importance_score", 1.0) or 1.0)
            new_val = float(max(0.0, min(1.0, cur + delta)))
            meta["importance_score"] = new_val
            meta["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.collection.update(ids=[entry_id], metadatas=[meta])
            return True
        except Exception as e:
            self.logger.warning(f"bump_importance failed for {entry_id}: {e}")
            return False
    
    async def _update_access_stats(self, entry_id: str) -> None:
        """更新访问统计"""
        try:
            # 获取当前条目
            result = self.collection.get(ids=[entry_id], include=["metadatas"])
            if result["metadatas"] and result["metadatas"][0]:
                metadata = result["metadatas"][0].copy()
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                metadata["last_accessed"] = datetime.now(timezone.utc).isoformat()
                
                # 更新
                self.collection.update(
                    ids=[entry_id],
                    metadatas=[metadata]
                )
        except Exception as e:
            self.logger.warning(f"Failed to update access stats for {entry_id}: {e}")
    
    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """更新记忆条目"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # 获取当前条目
            result = self.collection.get(ids=[entry_id], include=["metadatas", "documents"])
            if not result.get("ids") or not result["ids"][0] or entry_id not in result["ids"][0]:
                return False
            
            metadata = result["metadatas"][0].copy()
            document = result["documents"][0]
            
            # 应用更新
            new_content = updates.get("content", document)
            new_embedding = None
            
            # 优先使用调用方传入的预计算向量，避免重复计算
            if "embedding" in updates and isinstance(updates.get("embedding"), list):
                new_embedding = updates.get("embedding")
            elif ("content" in updates and updates["content"] != document) or bool(updates.get("_reembed", False)):
                # 内容变化或显式要求重嵌入时再计算
                new_embedding = await self.embedding_service.embed_text(new_content)
            
            # 更新元数据
            for key, value in updates.items():
                if key == "content":
                    continue
                elif key == "tags" and isinstance(value, list):
                    metadata[key] = json.dumps(value)
                elif key == "embedding":
                    # 跳过非元数据字段
                    continue
                else:
                    metadata[key] = value
            
            metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            # 更新ChromaDB
            update_params = {
                "ids": [entry_id],
                "metadatas": [metadata],
                "documents": [new_content]
            }
            
            if new_embedding:
                update_params["embeddings"] = [new_embedding]
            
            self.collection.update(**update_params)
            
            log_memory_operation(
                operation="update",
                memory_type="episodic",
                success=True,
                entry_id=entry_id
            )
            
            return True
            
        except Exception as e:
            log_memory_operation(
                operation="update",
                memory_type="episodic",
                success=False,
                error_message=str(e),
                entry_id=entry_id
            )
            return False
    
    async def delete(self, entry_id: str) -> bool:
        """删除记忆条目"""
        if not self._initialized:
            await self.initialize()
        
        try:
            self.collection.delete(ids=[entry_id])
            
            log_memory_operation(
                operation="delete",
                memory_type="episodic",
                success=True,
                entry_id=entry_id
            )
            
            return True
            
        except Exception as e:
            log_memory_operation(
                operation="delete",
                memory_type="episodic",
                success=False,
                error_message=str(e),
                entry_id=entry_id
            )
            return False
    
    async def close(self) -> None:
        """关闭后端"""
        if self.client:
            # ChromaDB客户端会自动持久化，无需显式关闭
            pass
        self._initialized = False
        self.logger.info("ChromaDB episodic backend closed")


class QdrantEpisodicBackend(BaseMemoryBackend):
    """Qdrant 情节记忆后端（生产可用，可通过配置切换）。"""

    def __init__(self, config: ConfigManager, embedding_service: EmbeddingProviderService):
        if not QDRANT_AVAILABLE:
            raise MemoryInitializationError(
                "Qdrant client not available. Please install: pip install qdrant-client"
            )

        self.config = config
        self.embedding_service = embedding_service
        self.logger = get_logger("qdrant_episodic_backend")

        self.client: Optional[QdrantClient] = None
        self.collection_name: Optional[str] = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            vcfg = self.config.vector_db
            endpoint = getattr(vcfg, "endpoint", None)
            api_key = getattr(vcfg, "api_key", None)
            if not endpoint:
                raise MemoryInitializationError("Qdrant endpoint is required when provider='qdrant'")

            # 连接客户端
            self.client = QdrantClient(url=endpoint, api_key=api_key)
            self.collection_name = vcfg.collection_name

            # 确保集合存在
            dim = 0
            try:
                dim = int(self.embedding_service.get_dimensions())
            except Exception:
                dim = 1024
            try:
                collections = self.client.get_collections()
                names = [c.name for c in getattr(collections, "collections", [])]
            except Exception:
                names = []
            if self.collection_name not in names:
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=dim, distance=QDistance.COSINE),
                )

            self._initialized = True
            self.logger.info(
                f"Qdrant backend initialized: {self.collection_name} @ {endpoint} dim={dim}"
            )
        except Exception as e:
            raise MemoryInitializationError(f"Failed to initialize Qdrant backend: {e}")

    async def store(self, entry: EpisodicMemoryEntry) -> str:
        if not self._initialized:
            await self.initialize()
        try:
            if entry.embedding is None:
                entry.embedding = await self.embedding_service.embed_text(entry.content)

            payload = {
                "content": entry.content,
                "timestamp": entry.timestamp.isoformat(),
                "importance_score": entry.importance_score,
                "access_count": entry.access_count,
                "source": entry.source or "",
                "context": entry.context or "",
                "tags": entry.tags or [],
                **(entry.metadata or {}),
            }
            # 注入嵌入信息
            try:
                info = self.embedding_service.get_provider_info() or {}
                if info.get("model"):
                    payload["embedding_model"] = info.get("model")
                    payload["embedding_version"] = str(info.get("model"))
                if info.get("dimensions") is not None:
                    payload["embedding_dim"] = int(info.get("dimensions") or 0)
            except Exception:
                pass

            # Upsert（幂等）
            pt = PointStruct(
                id=ExtendedPointId(entry.id),
                vector=entry.embedding,
                payload=payload,
            )
            self.client.upsert(collection_name=self.collection_name, points=[pt])
            log_memory_operation("store", "episodic", True, entry_id=entry.id)
            return entry.id
        except Exception as e:
            log_memory_operation("store", "episodic", False, error_message=str(e), entry_id=entry.id)
            raise MemoryStorageError(f"Qdrant store failed: {e}")

    async def retrieve(self, query: MemorySearchQuery) -> List[MemorySearchResult]:
        if not self._initialized:
            await self.initialize()
        try:
            qvec = await self.embedding_service.embed_text(query.text)

            # 基本过滤（仅 source/context，tags 客户端后过滤）
            qfilter: Optional[QFilter] = None
            if query.filters:
                must: List[FieldCondition] = []
                if "source" in query.filters:
                    must.append(FieldCondition(key="source", match=MatchValue(value=str(query.filters["source"]))))
                if "context" in query.filters:
                    must.append(FieldCondition(key="context", match=MatchValue(value=str(query.filters["context"]))))
                if must:
                    qfilter = QFilter(must=must)

            # 在后台线程执行以避免阻塞事件循环
            def _search():
                return self.client.search(
                    collection_name=self.collection_name,
                    query_vector=qvec,
                    limit=query.limit,
                    query_filter=qfilter,
                    with_payload=True,
                )
            results = await asyncio.to_thread(_search)

            out: List[MemorySearchResult] = []
            for r in results:
                score = float(r.score or 0.0)
                # Qdrant 返回的是相似度分数，通常已在 0..1 内；保底规约
                sim = max(0.0, min(1.0, score))
                if sim < query.similarity_threshold:
                    continue
                payload = r.payload or {}
                # 处理 tags 过滤（客户端侧）
                if query.filters and "tags" in query.filters:
                    req = query.filters["tags"]
                    req_list = req if isinstance(req, list) else [req]
                    entry_tags = payload.get("tags") or []
                    if not any(t in entry_tags for t in req_list):
                        continue

                entry = EpisodicMemoryEntry(
                    id=str(r.id),
                    content=str(payload.get("content", "")),
                    metadata={k: v for k, v in payload.items() if k not in {"content", "tags", "source", "context", "timestamp", "importance_score", "access_count"}},
                    timestamp=datetime.fromisoformat(payload.get("timestamp", datetime.now(timezone.utc).isoformat())) if isinstance(payload.get("timestamp"), str) else datetime.now(timezone.utc),
                    importance_score=float(payload.get("importance_score", 1.0) or 1.0),
                    access_count=int(payload.get("access_count", 0) or 0),
                    source=payload.get("source"),
                    context=payload.get("context"),
                    tags=list(payload.get("tags") or []),
                )
                out.append(MemorySearchResult(entry=entry, similarity_score=sim, memory_type=MemoryType.EPISODIC, explanation="vector_search"))
            # 访问计数（可选：批量 set_payload 增加 access_count）略
            return out
        except Exception as e:
            raise MemoryRetrievalError(f"Qdrant retrieve failed: {e}")

    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        if not self._initialized:
            await self.initialize()
        try:
            # 读取现有点
            recs = await asyncio.to_thread(
                self.client.retrieve, self.collection_name, [ExtendedPointId(entry_id)], True, True
            )
            point = recs[0] if recs else None
            payload = dict(getattr(point, "payload", {}) or {})
            vector = getattr(point, "vector", None)

            new_content = updates.get("content", payload.get("content", ""))
            new_embedding = None
            if ("content" in updates and updates["content"] != payload.get("content")) or bool(updates.get("_reembed", False)):
                new_embedding = await self.embedding_service.embed_text(new_content)

            for k, v in updates.items():
                if k == "content" or k == "_reembed":
                    continue
                payload[k] = v
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            if "content" in updates:
                payload["content"] = new_content

            # set_payload（不会更新向量）
            await asyncio.to_thread(self.client.set_payload, self.collection_name, payload, [ExtendedPointId(entry_id)])
            # 如果需要，更新向量
            if new_embedding is not None:
                pt = PointStruct(id=ExtendedPointId(entry_id), vector=new_embedding, payload={})
                await asyncio.to_thread(self.client.upsert, self.collection_name, [pt])
            return True
        except Exception as e:
            self.logger.warning(f"Qdrant update failed: {e}")
            return False

    async def delete(self, entry_id: str) -> bool:
        if not self._initialized:
            await self.initialize()
        try:
            from qdrant_client.models import PointIdsList
            await asyncio.to_thread(self.client.delete, self.collection_name, PointIdsList(points=[ExtendedPointId(entry_id)]))
            return True
        except Exception as e:
            self.logger.warning(f"Qdrant delete failed: {e}")
            return False

    async def list_entries(self, max_items: int = 5000) -> List[EpisodicMemoryEntry]:
        if not self._initialized:
            await self.initialize()
        try:
            entries: List[EpisodicMemoryEntry] = []
            next_page = None
            fetched = 0
            while fetched < max_items:
                res = await asyncio.to_thread(
                    self.client.scroll,
                    self.collection_name,
                    True,
                    min(1000, max_items - fetched),
                    next_page,
                )
                pts = getattr(res, "points", [])
                next_page = getattr(res, "next_page_offset", None)
                if not pts:
                    break
                for p in pts:
                    payload = p.payload or {}
                    entry = EpisodicMemoryEntry(
                        id=str(p.id),
                        content=str(payload.get("content", "")),
                        metadata={k: v for k, v in payload.items() if k not in {"content", "tags", "source", "context", "timestamp", "importance_score", "access_count"}},
                        timestamp=datetime.fromisoformat(payload.get("timestamp", datetime.now(timezone.utc).isoformat())) if isinstance(payload.get("timestamp"), str) else datetime.now(timezone.utc),
                        importance_score=float(payload.get("importance_score", 1.0) or 1.0),
                        access_count=int(payload.get("access_count", 0) or 0),
                        source=payload.get("source"),
                        context=payload.get("context"),
                        tags=list(payload.get("tags") or []),
                    )
                    entries.append(entry)
                    fetched += 1
                    if fetched >= max_items:
                        break
                if not next_page:
                    break
            return entries
        except Exception as e:
            self.logger.warning(f"Qdrant list_entries failed: {e}")
            return []

    async def close(self) -> None:
        self._initialized = False
        self.logger.info("Qdrant episodic backend closed")

class NetworkXConceptualBackend(BaseMemoryBackend):
    """NetworkX知识图谱概念记忆后端"""
    
    def __init__(self, config: ConfigManager, embedding_service: EmbeddingProviderService):
        """
        初始化NetworkX后端
        
        Args:
            config: 配置管理器
            embedding_service: 嵌入服务
        """
        self.config = config
        self.embedding_service = embedding_service
        self.logger = get_logger("networkx_conceptual_backend")
        
        self.graph: Optional[nx.MultiDiGraph] = None
        self.graph_path: Optional[Path] = None
        self._node_embeddings: Dict[str, List[float]] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化知识图谱"""
        if self._initialized:
            return
        
        try:
            kg_config = self.config.knowledge_graph
            self.graph_path = Path(kg_config.file_path).resolve()
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 初始化图
            self.graph = nx.MultiDiGraph()
            
            # 优先从 GPickle 加载；若不存在或失败则回退 GraphML
            gpickle_path = self.graph_path.with_suffix('.gpickle')
            if gpickle_path.exists():
                try:
                    self.graph = await asyncio.to_thread(nx.read_gpickle, str(gpickle_path))
                    if not isinstance(self.graph, nx.MultiDiGraph):
                        self.graph = nx.MultiDiGraph(self.graph)
                    await self._load_node_embeddings()
                    self.logger.info(
                        f"Loaded knowledge graph from GPickle: nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to load GPickle graph, will try GraphML: {e}")
                    self.graph = nx.MultiDiGraph()
            
            if self.graph.number_of_nodes() == 0 and self.graph_path.exists():
                try:
                    self.graph = await asyncio.to_thread(nx.read_graphml, str(self.graph_path))
                    # 确保图是MultiDiGraph类型
                    if not isinstance(self.graph, nx.MultiDiGraph):
                        self.graph = nx.MultiDiGraph(self.graph)
                    # 加载节点嵌入
                    await self._load_node_embeddings()
                    self.logger.info(
                        f"Loaded knowledge graph from GraphML: nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to load GraphML; starting with empty graph: {e}")
                    self.graph = nx.MultiDiGraph()
            
            self._initialized = True
            self.logger.info("NetworkX conceptual backend initialized successfully")
            
        except Exception as e:
            raise MemoryInitializationError(f"Failed to initialize NetworkX backend: {e}")
    
    async def store(self, entry: ConceptualMemoryEntry) -> str:
        """存储概念记忆条目（知识图谱节点）"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # 生成嵌入向量
            if entry.embedding is None:
                embedding_text = f"{entry.name} {entry.node_type.value}"
                if entry.attributes:
                    attr_text = " ".join(str(v) for v in entry.attributes.values() if isinstance(v, (str, int, float)))
                    embedding_text += f" {attr_text}"
                entry.embedding = await self.embedding_service.embed_text(embedding_text)
            
            # 添加节点到图
            node_attrs = {
                "node_type": entry.node_type.value,
                "name": entry.name,
                "importance_score": entry.importance_score,
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
                **entry.attributes
            }
            
            self.graph.add_node(entry.id, **node_attrs)
            
            # 缓存嵌入
            self._node_embeddings[entry.id] = entry.embedding
            
            # 保存图
            try:
                await self._save_graph()
            except Exception:
                pass
            
            duration = time.time() - start_time
            log_memory_operation(
                operation="store",
                memory_type="conceptual",
                success=True,
                entry_id=entry.id,
                node_type=entry.node_type.value,
                duration=duration
            )
            log_performance("networkx_store", duration)
            
            self.logger.debug(f"Stored conceptual memory node: {entry.id} ({entry.node_type.value})")
            return entry.id
            
        except Exception as e:
            log_memory_operation(
                operation="store",
                memory_type="conceptual",
                success=False,
                error_message=str(e),
                entry_id=entry.id
            )
            raise MemoryStorageError(f"Failed to store conceptual memory: {e}")
    
    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """添加节点间的关系"""
        if not self._initialized:
            await self.initialize()
        
        try:
            if source_id not in self.graph or target_id not in self.graph:
                raise ValueError(f"Source or target node not found: {source_id}, {target_id}")
            
            edge_attrs = {
                "edge_type": edge_type.value,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **(attributes or {})
            }
            
            self.graph.add_edge(source_id, target_id, **edge_attrs)
            try:
                await self._save_graph()
            except Exception:
                pass
            
            log_memory_operation(
                operation="add_relationship",
                memory_type="conceptual",
                success=True,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type.value
            )
            
            return True
            
        except Exception as e:
            log_memory_operation(
                operation="add_relationship",
                memory_type="conceptual",
                success=False,
                error_message=str(e),
                source_id=source_id,
                target_id=target_id
            )
            return False
    
    async def retrieve(self, query: MemorySearchQuery) -> List[MemorySearchResult]:
        """检索概念记忆"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # 生成查询嵌入
            query_embedding = await self.embedding_service.embed_text(query.text)
            
            # 计算与所有节点的相似度
            similarities = []
            for node_id in self.graph.nodes():
                if node_id in self._node_embeddings:
                    node_embedding = self._node_embeddings[node_id]
                    similarity = self._calculate_cosine_similarity(query_embedding, node_embedding)
                    
                    if similarity >= query.similarity_threshold:
                        similarities.append((node_id, similarity))
            
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            similarities = similarities[:query.limit]
            
            # 构建结果
            memory_results = []
            for node_id, similarity in similarities:
                node_data = self.graph.nodes[node_id]
                
                entry = ConceptualMemoryEntry(
                    id=node_id,
                    node_type=NodeType(node_data["node_type"]),
                    name=node_data["name"],
                    attributes={k: v for k, v in node_data.items() 
                              if k not in ["node_type", "name", "importance_score", "created_at", "updated_at"]},
                    importance_score=node_data.get("importance_score", 1.0),
                    created_at=datetime.fromisoformat(node_data["created_at"]),
                    updated_at=datetime.fromisoformat(node_data["updated_at"]) if node_data.get("updated_at") else None
                )
                
                result = MemorySearchResult(
                    entry=entry,
                    similarity_score=similarity,
                    memory_type=MemoryType.CONCEPTUAL,
                    explanation=f"Conceptual similarity: {similarity:.3f}"
                )
                
                # 添加相关节点（如果启用）
                if query.include_related:
                    related_results = await self._find_related_nodes(node_id, query.search_depth)
                    result.related_entries = related_results
                
                memory_results.append(result)
            
            duration = time.time() - start_time
            log_memory_operation(
                operation="retrieve",
                memory_type="conceptual",
                success=True,
                query=query.text,
                results_count=len(memory_results),
                duration=duration
            )
            log_performance("networkx_retrieve", duration, results_count=len(memory_results))
            
            return memory_results
            
        except Exception as e:
            log_memory_operation(
                operation="retrieve",
                memory_type="conceptual",
                success=False,
                error_message=str(e),
                query=query.text
            )
            raise MemoryRetrievalError(f"Failed to retrieve conceptual memory: {e}")
    
    async def _find_related_nodes(self, node_id: str, depth: int) -> List[MemorySearchResult]:
        """查找相关节点"""
        if depth <= 0:
            return []
        
        related_results = []
        
        try:
            # 获取直接邻居
            neighbors = set()
            
            # 出边邻居
            for neighbor in self.graph.successors(node_id):
                neighbors.add(neighbor)
            
            # 入边邻居
            for neighbor in self.graph.predecessors(node_id):
                neighbors.add(neighbor)
            
            # 构建相关节点结果
            for neighbor_id in list(neighbors)[:5]:  # 限制相关节点数量
                node_data = self.graph.nodes[neighbor_id]
                
                entry = ConceptualMemoryEntry(
                    id=neighbor_id,
                    node_type=NodeType(node_data["node_type"]),
                    name=node_data["name"],
                    attributes={k: v for k, v in node_data.items() 
                              if k not in ["node_type", "name", "importance_score", "created_at", "updated_at"]},
                    importance_score=node_data.get("importance_score", 1.0),
                    created_at=datetime.fromisoformat(node_data["created_at"]),
                    updated_at=datetime.fromisoformat(node_data["updated_at"]) if node_data.get("updated_at") else None
                )
                
                # 计算关系强度（基于两节点之间双向边的数量）
                try:
                    edge_count = self.graph.number_of_edges(node_id, neighbor_id) + self.graph.number_of_edges(neighbor_id, node_id)
                except Exception:
                    edge_count = 0
                relationship_strength = min(1.0, float(edge_count) * 0.3)
                
                related_results.append(MemorySearchResult(
                    entry=entry,
                    similarity_score=relationship_strength,
                    memory_type=MemoryType.CONCEPTUAL,
                    explanation=f"Graph relationship strength: {relationship_strength:.3f}"
                ))
            
        except Exception as e:
            self.logger.warning(f"Failed to find related nodes for {node_id}: {e}")
        
        return related_results
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            a = np.array(vec1)
            b = np.array(vec2)
            cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            return max(0.0, (cosine_sim + 1) / 2)  # 归一化到[0,1]
        except:
            return 0.0
    
    async def _load_node_embeddings(self) -> None:
        """加载节点嵌入"""
        try:
            embedding_path = self.graph_path.with_suffix('.embeddings.pkl')
            if embedding_path.exists():
                with open(embedding_path, 'rb') as f:
                    self._node_embeddings = pickle.load(f)
                self.logger.info(f"Loaded {len(self._node_embeddings)} node embeddings")
        except Exception as e:
            self.logger.warning(f"Failed to load node embeddings: {e}")
    
    async def _save_graph(self) -> None:
        """保存知识图谱和嵌入。

        GraphML 导出仅用于互操作/可视化，若不需要可通过配置关闭，避免对多重边的工程性降级。
        """
        try:
            # 1) 首选 GPickle 完整保存（保留 MultiDiGraph 多重边与属性）
            gpickle_path = self.graph_path.with_suffix('.gpickle')
            try:
                write_gpickle = getattr(nx, "write_gpickle", None)
                if write_gpickle:
                    await asyncio.to_thread(write_gpickle, self.graph, str(gpickle_path))
                else:
                    # 回退：直接使用 pickle.dump 序列化图
                    def _dump_graph_pickle(path: Path, graph):
                        with open(path, 'wb') as f:
                            pickle.dump(graph, f)
                    await asyncio.to_thread(_dump_graph_pickle, gpickle_path, self.graph)
            except Exception:
                # 强制回退到 pickle.dump
                def _dump_graph_pickle(path: Path, graph):
                    with open(path, 'wb') as f:
                        pickle.dump(graph, f)
                await asyncio.to_thread(_dump_graph_pickle, gpickle_path, self.graph)

            # 2) 可选 GraphML 导出（用于可视化/互操作）。默认开启；若需严格无降级可在配置关闭。
            export_graphml = True
            try:
                kg_cfg = getattr(self.config, 'knowledge_graph', None)
                # 若配置中存在 export_graphml=False，则跳过
                if kg_cfg and hasattr(kg_cfg, 'export_graphml'):
                    export_graphml = bool(getattr(kg_cfg, 'export_graphml'))
            except Exception:
                export_graphml = True

            if export_graphml:
                safe_graph = self.graph
                try:
                    await asyncio.to_thread(nx.write_graphml, safe_graph, str(self.graph_path))
                except Exception:
                    # 回退：将不可序列化的复杂属性转换为字符串后再写入
                    safe_graph = nx.MultiDiGraph()
                    for n, attrs in self.graph.nodes(data=True):
                        safe_attrs = {}
                        for k, v in attrs.items():
                            try:
                                _ = str(v) if not isinstance(v, (str, int, float)) else v
                                safe_attrs[k] = _
                            except Exception:
                                safe_attrs[k] = ""
                        safe_graph.add_node(n, **safe_attrs)
                    for u, v, attrs in self.graph.edges(data=True):
                        safe_attrs = {}
                        for k, val in attrs.items():
                            try:
                                _ = str(val) if not isinstance(val, (str, int, float)) else val
                                safe_attrs[k] = _
                            except Exception:
                                safe_attrs[k] = ""
                        safe_graph.add_edge(u, v, **safe_attrs)
                    await asyncio.to_thread(nx.write_graphml, safe_graph, str(self.graph_path))
            
            # 保存嵌入
            embedding_path = self.graph_path.with_suffix('.embeddings.pkl')
            def _dump_embeddings(path: Path, data: dict):
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
            await asyncio.to_thread(_dump_embeddings, embedding_path, self._node_embeddings)
                
        except Exception as e:
            self.logger.error(f"Failed to save graph: {e}")
            raise
    
    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """更新知识图谱节点"""
        if not self._initialized:
            await self.initialize()
        
        try:
            if entry_id not in self.graph:
                return False
            
            # 更新节点属性
            node_data = self.graph.nodes[entry_id]
            for key, value in updates.items():
                if key != "id":  # 不允许更新ID
                    node_data[key] = value
            
            node_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            # 如果更新了核心信息，重新生成嵌入
            if any(key in updates for key in ["name", "node_type", "attributes"]):
                embedding_text = f"{node_data['name']} {node_data['node_type']}"
                if "attributes" in node_data:
                    attr_text = " ".join(str(v) for v in node_data["attributes"].values() 
                                       if isinstance(v, (str, int, float)))
                    embedding_text += f" {attr_text}"
                
                new_embedding = await self.embedding_service.embed_text(embedding_text)
                self._node_embeddings[entry_id] = new_embedding
            
            try:
                await self._save_graph()
            except Exception:
                pass
            
            log_memory_operation(
                operation="update",
                memory_type="conceptual",
                success=True,
                entry_id=entry_id
            )
            
            return True
            
        except Exception as e:
            log_memory_operation(
                operation="update",
                memory_type="conceptual",
                success=False,
                error_message=str(e),
                entry_id=entry_id
            )
            return False
    
    async def delete(self, entry_id: str) -> bool:
        """删除知识图谱节点"""
        if not self._initialized:
            await self.initialize()
        
        try:
            if entry_id not in self.graph:
                return False
            
            self.graph.remove_node(entry_id)
            
            if entry_id in self._node_embeddings:
                del self._node_embeddings[entry_id]
            
            try:
                await self._save_graph()
            except Exception:
                pass
            
            log_memory_operation(
                operation="delete",
                memory_type="conceptual",
                success=True,
                entry_id=entry_id
            )
            
            return True
            
        except Exception as e:
            log_memory_operation(
                operation="delete",
                memory_type="conceptual",
                success=False,
                error_message=str(e),
                entry_id=entry_id
            )
            return False

    async def bump_importance(self, node_id: str, delta: float) -> bool:
        """根据反馈调整概念节点的重要性分数。"""
        if not self._initialized:
            await self.initialize()
        try:
            if node_id not in self.graph:
                return False
            node = self.graph.nodes[node_id]
            cur = float(node.get("importance_score", 1.0) or 1.0)
            node["importance_score"] = float(max(0.0, min(1.0, cur + delta)))
            node["updated_at"] = datetime.now(timezone.utc).isoformat()
            await self._save_graph()
            return True
        except Exception as e:
            self.logger.warning(f"bump_importance failed for node {node_id}: {e}")
            return False
    
    async def close(self) -> None:
        """关闭后端"""
        try:
            if self.graph and self.graph_path:
                try:
                    await self._save_graph()
                except Exception:
                    pass
        except Exception as e:
            self.logger.error(f"Error saving graph during close: {e}")
        
        self._initialized = False
        self.logger.info("NetworkX conceptual backend closed")


class HybridMemorySystem:
    """
    SOTA级别混合记忆系统
    
    整合情节记忆（向量数据库）和概念记忆（知识图谱），
    提供统一的记忆接口和高级搜索功能。
    """
    
    def __init__(self, config: ConfigManager, embedding_service: EmbeddingProviderService):
        """
        无副作用构造函数
        
        Args:
            config: 配置管理器
            embedding_service: 嵌入服务
        """
        if not config.is_initialized():
            raise ValueError("ConfigManager must be initialized")
        
        self.config = config
        self.embedding_service = embedding_service
        self.logger = get_logger("hybrid_memory_system")
        
        # 后端存储
        self.episodic_backend: Optional[ChromaDBEpisodicBackend] = None
        self.conceptual_backend: Optional[NetworkXConceptualBackend] = None
        
        # 系统状态
        self._initialized = False
        self._memory_stats = {
            "episodic_count": 0,
            "conceptual_count": 0,
            "total_searches": 0,
            "cache_hits": 0
        }
    
    async def initialize(self) -> None:
        """
        异步初始化记忆系统
        
        Raises:
            MemoryInitializationError: 初始化失败
        """
        if self._initialized:
            return
        
        try:
            self.logger.info("Initializing hybrid memory system...")
            
            # 初始化后端（按配置选择提供商）
            provider = (self.config.vector_db.provider or "chromadb").lower() if self.config.vector_db else "chromadb"
            if provider == "qdrant":
                try:
                    self.episodic_backend = QdrantEpisodicBackend(self.config, self.embedding_service)
                except Exception as e:
                    self.logger.warning(f"Qdrant backend unavailable, fallback to ChromaDB: {e}")
                    self.episodic_backend = ChromaDBEpisodicBackend(self.config, self.embedding_service)
            else:
                self.episodic_backend = ChromaDBEpisodicBackend(self.config, self.embedding_service)
            self.conceptual_backend = NetworkXConceptualBackend(self.config, self.embedding_service)
            
            # 并行初始化
            await asyncio.gather(
                self.episodic_backend.initialize(),
                self.conceptual_backend.initialize()
            )
            
            # 更新统计信息
            await self._update_stats()
            
            self._initialized = True
            self.logger.info(
                f"Hybrid memory system initialized successfully. "
                f"Episodic entries: {self._memory_stats['episodic_count']}, "
                f"Conceptual entries: {self._memory_stats['conceptual_count']}"
            )
            
        except Exception as e:
            raise MemoryInitializationError(f"Failed to initialize hybrid memory system: {e}")

    def _get_embedding_info(self) -> Dict[str, Any]:
        """安全获取当前嵌入模型信息。"""
        try:
            info = self.embedding_service.get_provider_info() or {}
            return {
                "model": info.get("model"),
                "dimensions": info.get("dimensions"),
            }
        except Exception:
            return {"model": None, "dimensions": None}

    def _make_stable_id(
        self,
        content: str,
        source: Optional[str],
        context: Optional[str],
        tags: List[str],
        metadata: Dict[str, Any],
    ) -> str:
        """生成稳定ID：基于模型名+内容+来源+上下文+规范化元数据与标签。

        注意：排除易变字段（timestamp/updated_at/last_accessed/access_count）。
        """
        try:
            emb = self._get_embedding_info()
            model_name = str(emb.get("model") or "")
        except Exception:
            model_name = ""

        # 过滤易变字段
        volatile_keys = {"timestamp", "updated_at", "last_accessed", "access_count"}
        stable_meta_items: List[Tuple[str, str]] = []
        try:
            for k in sorted((metadata or {}).keys()):
                if k in volatile_keys:
                    continue
                v = metadata.get(k)
                # 统一转字符串
                stable_meta_items.append((str(k), str(v)))
        except Exception:
            pass

        stable_tags = []
        try:
            stable_tags = sorted([str(t) for t in (tags or [])])
        except Exception:
            stable_tags = []

        content_hash = hashlib.sha256((content or "").encode("utf-8")).hexdigest()
        payload = {
            "model": model_name,
            "content": content_hash,  # 避免对超长文本直接哈希两次
            "source": str(source or ""),
            "context": str(context or ""),
            "tags": stable_tags,
            "metadata": stable_meta_items,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    
    async def store_episodic_memory(
        self,
        content: str,
        source: Optional[str] = None,
        context: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance_score: float = 1.0
    ) -> str:
        """
        存储情节记忆
        
        Args:
            content: 记忆内容
            source: 来源
            context: 上下文
            tags: 标签
            metadata: 额外元数据
            importance_score: 重要性评分
            
        Returns:
            记忆条目ID
        """
        if not self._initialized:
            await self.initialize()
        
        # 规范化元数据，确保值为 str/int/float/bool，None → ""
        clean_meta: Dict[str, Any] = {}
        try:
            for k, v in (metadata or {}).items():
                if v is None:
                    clean_meta[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    # 尽量用字符串表达，避免后端类型限制
                    clean_meta[k] = str(v)
        except Exception:
            clean_meta = {}

        # 生成稳定ID（幂等）
        entry_id = self._make_stable_id(content, source, context, tags or [], clean_meta)
        entry = EpisodicMemoryEntry(
            id=entry_id,
            content=content,
            source=source,
            context=context,
            tags=tags or [],
            metadata=clean_meta,
            importance_score=importance_score
        )
        
        result_id = await self.episodic_backend.store(entry)
        # 刷新统计，避免幂等写入导致的统计偏差
        await self._update_stats()
        
        self.logger.debug(f"Stored episodic memory: {result_id}")
        return result_id
    
    async def store_conceptual_memory(
        self,
        name: str,
        node_type: NodeType,
        attributes: Optional[Dict[str, Any]] = None,
        importance_score: float = 1.0
    ) -> str:
        """
        存储概念记忆（知识图谱节点）
        
        Args:
            name: 概念名称
            node_type: 节点类型
            attributes: 节点属性
            importance_score: 重要性评分
            
        Returns:
            节点ID
        """
        if not self._initialized:
            await self.initialize()
        
        node_id = str(uuid.uuid4())
        entry = ConceptualMemoryEntry(
            id=node_id,
            node_type=node_type,
            name=name,
            attributes=attributes or {},
            importance_score=importance_score
        )
        
        result_id = await self.conceptual_backend.store(entry)
        # 刷新统计，确保与后端真实条目数一致
        await self._update_stats()
        
        self.logger.debug(f"Stored conceptual memory: {result_id} ({node_type.value})")
        return result_id
    
    async def add_concept_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: EdgeType,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加概念间关系
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            relationship_type: 关系类型
            attributes: 关系属性
            
        Returns:
            是否成功
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.conceptual_backend.add_relationship(
            source_id, target_id, relationship_type, attributes
        )
    
    async def search_memory(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        include_related: bool = True,
        search_depth: int = 2,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemorySearchResult]:
        """
        混合记忆搜索
        
        Args:
            query: 搜索查询
            memory_types: 搜索的记忆类型
            limit: 结果限制
            similarity_threshold: 相似度阈值
            include_related: 是否包含相关条目
            search_depth: 搜索深度
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        memory_types = memory_types or [MemoryType.EPISODIC, MemoryType.CONCEPTUAL]
        
        search_query = MemorySearchQuery(
            text=query,
            memory_types=memory_types,
            limit=limit,
            similarity_threshold=similarity_threshold,
            filters=filters,
            include_related=include_related,
            search_depth=search_depth
        )
        # =============================
        # 阶段A：查询扩展与多视角召回（粗排）
        # =============================
        expanded_queries: List[str] = [query]
        try:
            if MemoryType.CONCEPTUAL in memory_types and self.conceptual_backend and self.conceptual_backend.graph:
                concept_seed_query = MemorySearchQuery(
                    text=query,
                    memory_types=[MemoryType.CONCEPTUAL],
                    limit=min(5, max(3, limit // 2)),
                    similarity_threshold=max(0.5, similarity_threshold * 0.8),
                    include_related=False,
                    search_depth=1,
                    filters=None,
                )
                concept_seeds = await self.conceptual_backend.retrieve(concept_seed_query)
                existing_lower = {q.lower() for q in expanded_queries}
                for seed in concept_seeds:
                    try:
                        name = getattr(seed.entry, "name", None)
                        if isinstance(name, str) and name and name.lower() not in existing_lower:
                            expanded_queries.append(name)
                            existing_lower.add(name.lower())
                    except Exception:
                        continue
        except Exception as e:
            self.logger.warning(f"Query expansion skipped: {e}")

        if len(expanded_queries) > 5:
            expanded_queries = expanded_queries[:5]

        episodic_tasks: List[asyncio.Task] = []
        conceptual_tasks: List[asyncio.Task] = []

        if MemoryType.EPISODIC in memory_types and self.episodic_backend:
            for q_text in expanded_queries:
                episodic_tasks.append(
                    asyncio.create_task(
                        self.episodic_backend.retrieve(
                            MemorySearchQuery(
                                text=q_text,
                                memory_types=[MemoryType.EPISODIC],
                                limit=max(limit * 3, limit),
                                similarity_threshold=max(0.5, similarity_threshold * 0.8),
                                filters=filters,
                                include_related=False,
                                search_depth=1,
                            )
                        )
                    )
                )

        if MemoryType.CONCEPTUAL in memory_types and self.conceptual_backend:
            conceptual_tasks.append(asyncio.create_task(self.conceptual_backend.retrieve(search_query)))

        recall_results = await asyncio.gather(*episodic_tasks, *conceptual_tasks, return_exceptions=True)

        all_results: List[MemorySearchResult] = []
        seen_keys: Dict[Tuple[str, str], MemorySearchResult] = {}
        for result in recall_results:
            if isinstance(result, Exception):
                self.logger.warning(f"Search error: {result}")
                continue
            for res in result:
                try:
                    entry_id = getattr(res.entry, "id", None)
                    mtype = res.memory_type.value
                    key = (mtype, entry_id)
                    if key not in seen_keys:
                        seen_keys[key] = res
                    else:
                        if float(getattr(res, "similarity_score", 0.0) or 0.0) > float(getattr(seen_keys[key], "similarity_score", 0.0) or 0.0):
                            seen_keys[key] = res
                except Exception:
                    all_results.append(res)
        if not all_results:
            all_results = list(seen_keys.values())

        # 图谱邻域增强：从概念结果中提取邻居名称，追加一轮情节检索（小规模）
        try:
            neighbor_queries: List[str] = []
            if self.episodic_backend and all_results:
                # 基于概念记忆候选采样
                for res in all_results:
                    if res.memory_type == MemoryType.CONCEPTUAL:
                        try:
                            name = getattr(res.entry, "name", None)
                            if isinstance(name, str) and name:
                                neighbor_queries.append(name)
                            for rel in (res.related_entries or [])[:3]:
                                n = getattr(rel.entry, "name", None)
                                if isinstance(n, str) and n:
                                    neighbor_queries.append(n)
                        except Exception:
                            continue
                # 去重并排除已扩展查询
                existing_lower = {q.lower() for q in expanded_queries}
                uniq_neighbors = []
                for q in neighbor_queries:
                    if q.lower() not in existing_lower:
                        uniq_neighbors.append(q)
                        existing_lower.add(q.lower())
                # 限制规模，避免成本过高
                uniq_neighbors = uniq_neighbors[: min(5, limit)]
                if uniq_neighbors:
                    extra_tasks = [
                        asyncio.create_task(
                            self.episodic_backend.retrieve(
                                MemorySearchQuery(
                                    text=qq,
                                    memory_types=[MemoryType.EPISODIC],
                                    limit=max(limit // 2, 5),
                                    similarity_threshold=max(0.5, similarity_threshold * 0.8),
                                    filters=filters,
                                    include_related=False,
                                    search_depth=1,
                                )
                            )
                        )
                        for qq in uniq_neighbors
                    ]
                    extra_results = await asyncio.gather(*extra_tasks, return_exceptions=True)
                    for er in extra_results:
                        if isinstance(er, Exception):
                            continue
                        for r in er:
                            try:
                                key = (r.memory_type.value, getattr(r.entry, "id", ""))
                                if key not in seen_keys:
                                    all_results.append(r)
                                    seen_keys[key] = r
                            except Exception:
                                all_results.append(r)
        except Exception as _e:
            # 邻域扩展失败不影响主链路
            pass
        
        # 先按基础相似度排序
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)

        # 基于融合打分进行精排（BM25 不可用时，采用以下稳健融合）：
        # score = w_sim * sim + w_imp * importance + w_recency * recency + w_graph * graph_rel
        # 其中：
        # - importance: 映射 0..1（元数据 importance_score）
        # - recency:   基于时间衰减，越新越高分
        # - graph_rel: 对于概念记忆，related_entries 的强度均值；情节记忆为 0
        def _normalized_importance(entry: Union[EpisodicMemoryEntry, ConceptualMemoryEntry]) -> float:
            try:
                return float(getattr(entry, "importance_score", 0.0))
            except Exception:
                return 0.0

        def _recency_score(ts: Optional[datetime]) -> float:
            try:
                if not ts:
                    return 0.3
                now = datetime.now(timezone.utc)
                delta_days = max(0.0, (now - ts).total_seconds() / 86400.0)
                # 指数衰减，最近 7 天接近 1，>180 天趋向 0.1
                half_life = 30.0
                score = np.exp(-delta_days / half_life)
                return float(max(0.1, min(1.0, score)))
            except Exception:
                return 0.3

        def _graph_strength(res: MemorySearchResult) -> float:
            try:
                if res.memory_type != MemoryType.CONCEPTUAL:
                    return 0.0
                if not res.related_entries:
                    return 0.0
                vals = [float(getattr(r, "similarity_score", 0.0) or 0.0) for r in res.related_entries]
                if not vals:
                    return 0.0
                return float(sum(vals) / len(vals))
            except Exception:
                return 0.0

        # 权重：以相似度为主，其余为辅
        w_sim, w_imp, w_recency, w_graph = 0.65, 0.15, 0.15, 0.05

        def _fused_score(res: MemorySearchResult) -> float:
            try:
                base = float(getattr(res, "similarity_score", 0.0) or 0.0)
                entry = getattr(res, "entry", None)
                imp = _normalized_importance(entry) if entry is not None else 0.0
                # 时间戳（仅情节记忆含 timestamp；概念记忆用 created_at）
                ts = None
                try:
                    ts = getattr(entry, "timestamp", None) or getattr(entry, "created_at", None)
                except Exception:
                    ts = None
                rec = _recency_score(ts)
                graph = _graph_strength(res)
                fused = w_sim * base + w_imp * imp + w_recency * rec + w_graph * graph
                return float(max(0.0, min(1.0, fused)))
            except Exception:
                return float(getattr(res, "similarity_score", 0.0) or 0.0)

        all_results.sort(key=_fused_score, reverse=True)

        # 可选 Cross-Encoder 精排（仅当启用且候选足够时）。
        # 对情节记忆候选取若干条文本做二次重排，提高相关性精度。
        try:
            es = self.embedding_service
            if es and getattr(es.config.embedding_settings, "enable_reranker", False):
                # 仅对前 3*limit 的情节记忆文本做精排
                pre_k = min(len(all_results), max(limit * 3, limit))
                head = all_results[:pre_k]
                episodic_pairs = []
                episodic_indices = []
                for idx, res in enumerate(head):
                    if res.memory_type == MemoryType.EPISODIC:
                        try:
                            text = getattr(res.entry, "content", None)
                            if isinstance(text, str) and text.strip():
                                episodic_pairs.append(text)
                                episodic_indices.append(idx)
                        except Exception:
                            pass
                if episodic_pairs:
                    # 使用 Cross-Encoder 对情节记忆进行精排，保留相对顺序并混排回 head
                    ranks = await es.rerank(query, episodic_pairs)
                    # 构造映射：在 head 中的索引 -> 精排分数
                    score_map = {episodic_indices[i]: score for i, score in ranks}
                    # 稳健排序键：先按是否有精排分数，再按精排分数，再按融合分数
                    def _head_key(i_res):
                        i, res = i_res
                        has = 1 if i in score_map else 0
                        rerank_score = score_map.get(i, -1e9)
                        return (has, rerank_score, _fused_score(res))
                    head = list(enumerate(head))
                    head.sort(key=_head_key, reverse=True)
                    head = [res for _, res in head]
                    # 合并重排后的 head 与 tail
                    tail = all_results[pre_k:]
                    all_results = head + tail
        except Exception as e:
            self.logger.warning(f"Cross-encoder rerank skipped: {e}")

        # =============================
        # 阶段B：MMR 多样性约束
        # =============================
        final_results = all_results
        try:
            if final_results:
                pre_k_mmr = min(len(final_results), max(limit * 5, limit))
                candidates = final_results[:pre_k_mmr]

                def _candidate_text(res: MemorySearchResult) -> str:
                    try:
                        if res.memory_type == MemoryType.EPISODIC:
                            return str(getattr(res.entry, "content", ""))
                        else:
                            name = str(getattr(res.entry, "name", ""))
                            attrs = getattr(res.entry, "attributes", {}) or {}
                            extra = " ".join(str(v) for v in list(attrs.values())[:3] if isinstance(v, (str, int, float)))
                            return f"{name} {extra}".strip()
                    except Exception:
                        return ""

                texts = [_candidate_text(r) for r in candidates]
                if any(t.strip() for t in texts):
                    q_vec = await self.embedding_service.embed_text(query)
                    cand_vecs = await self.embedding_service.embed_texts(texts)

                    def _cos(a: List[float], b: List[float]) -> float:
                        try:
                            va = np.array(a)
                            vb = np.array(b)
                            sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
                            return max(0.0, (sim + 1.0) / 2.0)
                        except Exception:
                            return 0.0

                    q_sims = [
                        _cos(q_vec, cand_vecs[i])
                        for i in range(len(cand_vecs))
                    ]

                    lambda_weight = 0.7
                    selected_indices: List[int] = []
                    remaining_indices: Set[int] = set(range(len(candidates)))

                    while remaining_indices and len(selected_indices) < limit:
                        best_idx = None
                        best_score = -1e9
                        for i in list(remaining_indices):
                            if selected_indices:
                                div = 0.0
                                for j in selected_indices:
                                    div = max(div, _cos(cand_vecs[i], cand_vecs[j]))
                            else:
                                div = 0.0
                            score = lambda_weight * q_sims[i] - (1 - lambda_weight) * div
                            if score > best_score:
                                best_score = score
                                best_idx = i
                        selected_indices.append(best_idx)  # type: ignore[arg-type]
                        remaining_indices.remove(best_idx)  # type: ignore[arg-type]

                    final_results = [candidates[i] for i in selected_indices]
        except Exception as e:
            self.logger.warning(f"MMR diversity skipped: {e}")
            final_results = all_results[:limit]
        
        # 更新统计
        self._memory_stats["total_searches"] += 1
        
        duration = time.time() - start_time
        log_performance("hybrid_memory_search", duration, 
                       query=query, results_count=len(final_results))
        
        self.logger.debug(f"Memory search completed: {len(final_results)} results for '{query}'")
        return final_results

    async def reembed_all(self, target_model: Optional[str] = None, batch_size: int = 128) -> Dict[str, Any]:
        """对使用旧嵌入版本的情节记忆执行批量重嵌入（幂等、安全）。

        注意：当前实现使用系统已配置的嵌入服务模型。传入的 target_model 仅用于记录与校验；
        若与当前模型不一致，将记录告警但仍按当前模型执行。
        """
        if not self._initialized:
            await self.initialize()

        stats = {"total": 0, "reembedded": 0, "skipped": 0, "errors": 0, "model": None}
        try:
            info = self._get_embedding_info()
            cur_model = str(info.get("model") or "")
            stats["model"] = cur_model
            if target_model and target_model != cur_model:
                self.logger.warning(f"reembed_all target_model={target_model} != current_model={cur_model}; will use current.")

            # 列出条目（限制上限以控制内存）
            entries = await self.episodic_backend.list_entries(max_items=5000) if self.episodic_backend else []
            stats["total"] = len(entries)
            if not entries:
                return stats

            # 选择需要重嵌入的条目
            def needs_reembed(e: EpisodicMemoryEntry) -> bool:
                try:
                    em = (e.metadata or {}).get("embedding_model")
                    ed = (e.metadata or {}).get("embedding_dim")
                    if not em:
                        return True
                    if cur_model and em != cur_model:
                        return True
                    # 维度不一致也需要重嵌入
                    dims = info.get("dimensions")
                    if dims is not None and ed is not None and int(ed) != int(dims):
                        return True
                except Exception:
                    return True
                return False

            to_reembed: List[EpisodicMemoryEntry] = [e for e in entries if needs_reembed(e)]
            if not to_reembed:
                return stats

            # 分批重嵌入（一次计算向量，并传入 update，避免重复计算）
            for i in range(0, len(to_reembed), batch_size):
                batch = to_reembed[i : i + batch_size]
                try:
                    texts = [b.content for b in batch]
                    new_vecs = await self.embedding_service.embed_texts(texts)
                    for idx, e in enumerate(batch):
                        try:
                            await self.episodic_backend.update(
                                e.id,
                                {
                                    "embedding": new_vecs[idx],
                                    "embedding_model": cur_model,
                                    "embedding_version": cur_model,
                                    "embedding_dim": int(info.get("dimensions") or 0),
                                },
                            )
                            stats["reembedded"] += 1
                        except Exception:
                            stats["errors"] += 1
                except Exception:
                    stats["errors"] += len(batch)

            stats["skipped"] = stats["total"] - stats["reembedded"] - stats["errors"]
            return stats
        except Exception as e:
            self.logger.error(f"reembed_all failed: {e}")
            return stats
    
    async def get_memory_graph_insights(self, concept_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        获取概念的图谱洞察
        
        Args:
            concept_id: 概念ID
            depth: 分析深度
            
        Returns:
            图谱洞察信息
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if concept_id not in self.conceptual_backend.graph:
                return {"error": "Concept not found"}
            
            graph = self.conceptual_backend.graph
            insights = {
                "concept_id": concept_id,
                "node_info": dict(graph.nodes[concept_id]),
                "direct_connections": {
                    "successors": list(graph.successors(concept_id)),
                    "predecessors": list(graph.predecessors(concept_id))
                },
                "graph_metrics": {}
            }
            
            # 计算图谱指标
            try:
                insights["graph_metrics"]["degree_centrality"] = nx.degree_centrality(graph).get(concept_id, 0)
                insights["graph_metrics"]["betweenness_centrality"] = nx.betweenness_centrality(graph).get(concept_id, 0)
                
                # 计算局部聚类系数
                undirected_graph = graph.to_undirected()
                insights["graph_metrics"]["clustering_coefficient"] = nx.clustering(undirected_graph).get(concept_id, 0)
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate graph metrics: {e}")
                insights["graph_metrics"]["error"] = str(e)
            
            # 查找路径
            if depth > 1:
                # 查找到其他重要节点的最短路径
                important_nodes = sorted(
                    graph.nodes(data=True),
                    key=lambda x: x[1].get("importance_score", 0),
                    reverse=True
                )[:5]
                
                insights["shortest_paths"] = {}
                for node_id, node_data in important_nodes:
                    if node_id != concept_id:
                        try:
                            path = nx.shortest_path(graph, concept_id, node_id)
                            insights["shortest_paths"][node_id] = {
                                "path": path,
                                "length": len(path) - 1,
                                "target_name": node_data.get("name", node_id)
                            }
                        except nx.NetworkXNoPath:
                            continue
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get memory graph insights: {e}")
            return {"error": str(e)}
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """
        优化记忆系统
        
        Returns:
            优化结果
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        optimization_results = {
            "episodic_optimizations": [],
            "conceptual_optimizations": [],
            "errors": []
        }
        
        try:
            # 清理与结构洞察（保留示例）
            # 可在此扩展更多策略
            
            # 示例：标记孤立节点
            if self.conceptual_backend.graph:
                isolated_nodes = list(nx.isolates(self.conceptual_backend.graph))
                if isolated_nodes:
                    optimization_results["conceptual_optimizations"].append({
                        "type": "isolated_nodes_detected",
                        "count": len(isolated_nodes),
                        "nodes": isolated_nodes[:10]  # 只显示前10个
                    })
            
            # 示例：检测社区结构
            try:
                if self.conceptual_backend.graph.number_of_nodes() > 10:
                    undirected = self.conceptual_backend.graph.to_undirected()
                    communities = list(nx.community.greedy_modularity_communities(undirected))
                    optimization_results["conceptual_optimizations"].append({
                        "type": "communities_detected",
                        "count": len(communities),
                        "modularity": nx.community.modularity(undirected, communities)
                    })
            except Exception as e:
                optimization_results["errors"].append(f"Community detection failed: {e}")

            # 记忆巩固与压缩：将高价值情节记忆归纳为抽象主题（TOPIC）节点，挂入图谱
            try:
                # 1) 取样高价值情节记忆
                def _recency_score(ts: Optional[datetime]) -> float:
                    try:
                        if not ts:
                            return 0.3
                        now = datetime.now(timezone.utc)
                        delta_days = max(0.0, (now - ts).total_seconds() / 86400.0)
                        half_life = 30.0
                        score = np.exp(-delta_days / half_life)
                        return float(max(0.1, min(1.0, score)))
                    except Exception:
                        return 0.3

                episodic_entries = []
                if self.episodic_backend:
                    episodic_entries = await self.episodic_backend.list_entries(max_items=3000)
                # 打分排序
                scored: List[Tuple[EpisodicMemoryEntry, float]] = []
                for e in episodic_entries:
                    val = 0.6 * float(getattr(e, "importance_score", 0.0) or 0.0) 
                    val += 0.25 * _recency_score(getattr(e, "timestamp", None))
                    val += 0.15 * (np.log1p(float(getattr(e, "access_count", 0) or 0)))
                    scored.append((e, float(val)))
                scored.sort(key=lambda x: x[1], reverse=True)
                top = [e for e, _ in scored[: min(500, len(scored))]]

                # 2) 简单聚类（k-means 变体），k 按数量自适应
                if top:
                    # 生成嵌入（如未缓存）
                    texts = [t.content for t in top]
                    vecs = await self.embedding_service.embed_texts(texts)
                    if vecs:
                        import random
                        k = max(3, min(20, len(vecs) // 50))
                        if k > len(vecs):
                            k = max(1, len(vecs))
                        centers = [vecs[i] for i in random.sample(range(len(vecs)), k)]

                        def _cos(a: List[float], b: List[float]) -> float:
                            try:
                                va = np.array(a)
                                vb = np.array(b)
                                sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
                                return max(0.0, (sim + 1.0) / 2.0)
                            except Exception:
                                return 0.0

                        # 迭代若干轮
                        assignments = [0] * len(vecs)
                        for _ in range(8):
                            # 分配
                            for i, v in enumerate(vecs):
                                best_c = 0
                                best_s = -1e9
                                for ci, c in enumerate(centers):
                                    s = _cos(v, c)
                                    if s > best_s:
                                        best_s = s
                                        best_c = ci
                                assignments[i] = best_c
                            # 更新中心
                            new_centers = []
                            for ci in range(len(centers)):
                                members = [vecs[i] for i, a in enumerate(assignments) if a == ci]
                                if members:
                                    new_centers.append((np.mean(np.array(members), axis=0)).tolist())
                                else:
                                    new_centers.append(centers[ci])
                            centers = new_centers

                        # 3) 为每个簇创建主题节点
                        def _keywords(text: str, topk: int = 5) -> List[str]:
                            # 极简关键词提取（频率，去停用与短词）
                            try:
                                stop = {"the","and","for","with","that","from","this","into","over","have","has","are","was","were","been","will","would","shall","you","your","our","their","its","but","not","can","could","may","might","should","of","in","on","to","as","by","is","at","we","it","be","or","an","a"}
                                tokens = [t.strip(".,:;!?()[]{}\'\"`).-_").lower() for t in text.split()]
                                freq: Dict[str, int] = {}
                                for t in tokens:
                                    if len(t) < 3 or t in stop:
                                        continue
                                    freq[t] = freq.get(t, 0) + 1
                                return [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:topk]]
                            except Exception:
                                return []

                        created_topics = 0
                        for ci in range(len(centers)):
                            members_idx = [i for i, a in enumerate(assignments) if a == ci]
                            if len(members_idx) < 5:
                                continue
                            cluster_text = "\n".join([top[i].content for i in members_idx[:50]])
                            kws = _keywords(cluster_text, topk=6)
                            if not kws:
                                continue
                            name = f"Topic: {', '.join(kws[:4])}"
                            imp = float(np.mean([getattr(top[i], "importance_score", 0.5) for i in members_idx]))
                            topic_id = await self.store_conceptual_memory(
                                name=name,
                                node_type=NodeType.TOPIC,
                                attributes={
                                    "consolidated": True,
                                    "cluster_size": len(members_idx),
                                },
                                importance_score=float(max(0.1, min(1.0, imp)))
                            )
                            # 关联到最相近的概念节点（前5）
                            try:
                                cvec = centers[ci]
                                sim_list: List[Tuple[str, float]] = []
                                for nid, nvec in (self.conceptual_backend._node_embeddings or {}).items():
                                    sim_list.append((nid, _cos(cvec, nvec)))
                                sim_list.sort(key=lambda x: x[1], reverse=True)
                                for nid, s in sim_list[:5]:
                                    if s >= 0.6:
                                        await self.add_concept_relationship(topic_id, nid, EdgeType.RELATED_TO, {"strength": s})
                            except Exception:
                                pass
                            created_topics += 1

                        if created_topics:
                            optimization_results["conceptual_optimizations"].append({
                                "type": "topics_consolidated",
                                "count": created_topics,
                            })
            except Exception as e:
                optimization_results["errors"].append(f"Consolidation failed: {e}")
            
            duration = time.time() - start_time
            log_performance("memory_optimization", duration)
            
            self.logger.info(f"Memory optimization completed in {duration:.2f}s")
            return optimization_results
            
        except Exception as e:
            optimization_results["errors"].append(str(e))
            self.logger.error(f"Memory optimization failed: {e}")
            return optimization_results
    
    async def _update_stats(self) -> None:
        """更新统计信息"""
        try:
            if self.episodic_backend and self.episodic_backend.collection:
                self._memory_stats["episodic_count"] = self.episodic_backend.collection.count()
            
            if self.conceptual_backend and self.conceptual_backend.graph:
                self._memory_stats["conceptual_count"] = self.conceptual_backend.graph.number_of_nodes()
                
        except Exception as e:
            self.logger.warning(f"Failed to update memory stats: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        stats = self._memory_stats.copy()
        
        if self.conceptual_backend and self.conceptual_backend.graph:
            stats.update({
                "conceptual_edges": self.conceptual_backend.graph.number_of_edges(),
                "graph_density": nx.density(self.conceptual_backend.graph),
                "embedding_cache_size": len(self.conceptual_backend._node_embeddings)
            })
        
        if self.embedding_service:
            stats.update(self.embedding_service.get_cache_stats())
        
        return stats
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    async def close(self) -> None:
        """关闭记忆系统"""
        if self.episodic_backend:
            await self.episodic_backend.close()
        
        if self.conceptual_backend:
            await self.conceptual_backend.close()
        
        self._initialized = False
        self.logger.info("Hybrid memory system closed")

    async def record_feedback(
        self,
        positive_ids: Optional[List[str]] = None,
        negative_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """根据检索/使用反馈更新重要性与访问统计。
        - positive_ids: 用户点击/使用的候选ID（可混合概念/情节）
        - negative_ids: 明确负反馈的候选ID
        返回更新统计。
        """
        if not self._initialized:
            await self.initialize()
        pos = set(positive_ids or [])
        neg = set(negative_ids or [])
        fb = self.config.feedback
        updated = {"episodic": 0, "conceptual": 0}
        # 情节：调整 importance_score，访问量已在检索中自增
        for eid in pos:
            try:
                if await self.episodic_backend.bump_importance(eid, float(fb.positive_weight_increase)):
                    updated["episodic"] += 1
            except Exception:
                pass
            try:
                if await self.conceptual_backend.bump_importance(eid, float(fb.positive_weight_increase)):
                    updated["conceptual"] += 1
            except Exception:
                pass
        for eid in neg:
            try:
                if await self.episodic_backend.bump_importance(eid, -float(fb.negative_weight_decrease)):
                    updated["episodic"] += 1
            except Exception:
                pass
            try:
                if await self.conceptual_backend.bump_importance(eid, -float(fb.negative_weight_decrease)):
                    updated["conceptual"] += 1
            except Exception:
                pass
        return updated

    async def build_rag_context(
        self,
        query: str,
        token_budget: int = 1600,
        segment_token_size: int = 180,
        dedupe_threshold: float = 0.9,
        limit_candidates: int = 30,
    ) -> List[str]:
        """构建段落级RAG上下文：精排+去冗余+窗口化，严格控制token预算。
        返回段落片段列表，按重要性排序。
        """
        if not self._initialized:
            await self.initialize()

        # 1) 召回候选
        results = await self.search_memory(
            query,
            memory_types=[MemoryType.EPISODIC],
            limit=limit_candidates,
            similarity_threshold=0.6,
            include_related=False,
            search_depth=1,
        )
        texts: List[str] = []
        for r in results:
            try:
                txt = str(getattr(r.entry, "content", ""))
                if txt:
                    texts.append(txt)
            except Exception:
                continue
        if not texts:
            return []

        # 2) 切分成近似 segment_token_size 的段落片段
        def _estimate_tokens(s: str) -> int:
            return max(1, len(s) // 4)

        def _split_segments(s: str, target_tokens: int) -> List[str]:
            # 简化策略：按段落拆分，再合并小段
            paras = [p.strip() for p in s.split("\n\n") if p.strip()]
            segs: List[str] = []
            buf = ""
            for p in paras:
                if not buf:
                    buf = p
                else:
                    if _estimate_tokens(buf) + _estimate_tokens(p) <= target_tokens:
                        buf = buf + "\n\n" + p
                    else:
                        segs.append(buf)
                        buf = p
            if buf:
                segs.append(buf)
            # 若段落过长，再按句子粗切
            refined: List[str] = []
            for seg in segs:
                if _estimate_tokens(seg) <= int(target_tokens * 1.5):
                    refined.append(seg)
                else:
                    sents = [s.strip() for s in seg.replace("\n", " ").split(". ") if s.strip()]
                    cur = ""
                    for ss in sents:
                        if not cur:
                            cur = ss
                        else:
                            if _estimate_tokens(cur) + _estimate_tokens(ss) <= target_tokens:
                                cur = cur + ". " + ss
                            else:
                                refined.append(cur)
                                cur = ss
                    if cur:
                        refined.append(cur)
            return refined

        segments: List[str] = []
        for t in texts:
            segments.extend(_split_segments(t, segment_token_size))
        if not segments:
            return []

        # 3) 片段级精排（可选 Cross-Encoder）
        ranks = []
        try:
            ranks = await self.embedding_service.rerank(query, segments)
            # 返回 [(index, score)] 按降序
        except Exception:
            ranks = []
        if not ranks:
            # 回退：用向量相似度作为分数
            qv = await self.embedding_service.embed_text(query)
            sv = await self.embedding_service.embed_texts(segments)
            def _cos(a: List[float], b: List[float]) -> float:
                try:
                    va = np.array(a); vb = np.array(b)
                    sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
                    return max(0.0, (sim + 1.0) / 2.0)
                except Exception:
                    return 0.0
            ranks = sorted(list(enumerate([_cos(qv, s) for s in sv])), key=lambda x: x[1], reverse=True)

        # 4) 去冗余与窗口化（token预算）
        ordered_indices = [i for i, _ in ranks]
        selected: List[str] = []
        selected_vecs: List[List[float]] = []
        total_tokens = 0
        for idx in ordered_indices:
            seg = segments[idx]
            est = _estimate_tokens(seg)
            if total_tokens + est > token_budget:
                continue
            try:
                # 计算与已选片段的最大相似度，避免重复
                if selected_vecs:
                    v = (await self.embedding_service.embed_texts([seg]))[0]
                    max_sim = 0.0
                    for sv in selected_vecs:
                        va = np.array(v); vb = np.array(sv)
                        sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
                        sim = max(0.0, (sim + 1.0) / 2.0)
                        if sim > max_sim:
                            max_sim = sim
                    if max_sim >= dedupe_threshold:
                        continue
                    selected_vecs.append(v)
                else:
                    selected_vecs.append((await self.embedding_service.embed_texts([seg]))[0])
            except Exception:
                pass
            selected.append(seg)
            total_tokens += est
            if total_tokens >= token_budget:
                break

        return selected

    # =====================
    # 高层论文分析落库接口
    # =====================
    async def store_paper_analysis(
        self,
        paper_metadata: Any,
        analysis_result: Any,
    ) -> Dict[str, str]:
        """
        将一次完整的论文分析结果写入混合记忆系统（严格真实实现，无任何伪实现）。

        - 写入情节记忆：全局概述、关键解释片段
        - 写入概念记忆：关键概念节点与关系

        Args:
            paper_metadata: 来自消化/发现阶段的论文元数据对象，需包含 title/authors/doi/arxiv_id 等字段
            analysis_result: 来自 DeepAnalysisCore 的 AnalysisResult 实例

        Returns:
            包含已写入的关键ID映射，便于上层追踪
        """
        if not self._initialized:
            await self.initialize()

        ids: Dict[str, str] = {}

        try:
            # 1) 全局概述写入情节记忆
            global_content_parts = [
                f"Paper: {getattr(analysis_result.global_map, 'title', '')}",
                f"Summary: {getattr(analysis_result, 'overall_summary', '')}",
                f"Domain: {getattr(analysis_result.global_map, 'research_domain', '')}",
            ]
            global_content = "\n".join([p for p in global_content_parts if p])

            # 规范化元数据，避免 None 进入后端
            _meta = {
                "paper_id": str(getattr(analysis_result, "paper_id", "") or ""),
                "arxiv_id": str(getattr(paper_metadata, "arxiv_id", "") or ""),
                "doi": str(getattr(paper_metadata, "doi", "") or ""),
                "title": str(getattr(analysis_result.global_map, "title", "") or ""),
            }
            episodic_global_id = await self.store_episodic_memory(
                content=global_content,
                source="paper_analysis",
                context=f"paper_id:{getattr(analysis_result, 'paper_id', '')}",
                tags=["paper_analysis", "global_map"],
                metadata=_meta,
                importance_score=float(getattr(analysis_result.global_map, "novelty_score", 0.5) or 0.5),
            )
            ids["episodic_global"] = episodic_global_id

            # 2) 关键概念写入概念记忆
            related_concepts = list(getattr(analysis_result.global_map, "related_concepts", []) or [])
            for concept in related_concepts[:10]:
                cid = await self.store_conceptual_memory(
                    name=str(concept),
                    node_type=NodeType.CONCEPT,
                    attributes={
                        "paper_id": getattr(analysis_result, "paper_id", ""),
                        "domain": getattr(analysis_result.global_map, "research_domain", ""),
                        "paper_title": getattr(analysis_result.global_map, "title", ""),
                    },
                    importance_score=1.0,
                )
                # 建立 Paper -> Concept 关系（需要 Paper 节点）
                ids.setdefault("concept_nodes", []).append(cid)

            # 3) Paper 节点（如果尚未建立，建立之）
            paper_name = getattr(paper_metadata, "title", None) or getattr(analysis_result.global_map, "title", "Unknown Paper")
            paper_node_id = await self.store_conceptual_memory(
                name=paper_name,
                node_type=NodeType.PAPER,
                attributes={
                    "arxiv_id": str(getattr(paper_metadata, "arxiv_id", "") or ""),
                    "doi": str(getattr(paper_metadata, "doi", "") or ""),
                    "venue": str((getattr(paper_metadata, "journal", None) or getattr(paper_metadata, "conference", None) or "")),
                },
                importance_score=float(getattr(analysis_result.global_map, "novelty_score", 0.5) or 0.5),
            )
            ids["paper_node"] = paper_node_id

            # 4) 关联 概念 <-> 论文
            for cid in ids.get("concept_nodes", []):
                await self.add_concept_relationship(cid, paper_node_id, EdgeType.RELATED_TO)

            # 5) 高质量意群解释写入情节记忆
            for exp in getattr(analysis_result, "chunk_explanations", []) or []:
                try:
                    confidence = float(getattr(exp, "confidence_score", 0.0) or 0.0)
                    if confidence < 0.7:
                        continue
                    ctx = f"paper:{paper_name}, chunk_id:{getattr(exp, 'chunk_id', '')}"
                    await self.store_episodic_memory(
                        content=str(getattr(exp, "explanation", "")),
                        source="chunk_explanation",
                        context=ctx,
                        tags=["chunk_explanation"],
                        metadata={
                            "paper_id": str(getattr(analysis_result, "paper_id", "") or ""),
                            "chunk_id": str(getattr(exp, "chunk_id", "") or ""),
                        },
                        importance_score=confidence,
                    )
                except Exception:
                    # 严格不中断批量写入
                    continue

            return ids

        except Exception as e:
            self.logger.error(f"store_paper_analysis failed: {e}")
            raise
