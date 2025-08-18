"""
配置管理模块 - 中心化配置管理

严格遵循依赖注入原则，无副作用构造函数，显式异步初始化。
提供强类型访问，绝不使用字典式配置访问。
"""

import os
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import aiofiles
import logging


@dataclass
class APIKeysConfig:
    """API密钥配置"""
    openai: Optional[str] = None
    google_gemini: Optional[str] = None
    anthropic: Optional[str] = None
    twitter_bearer_token: Optional[str] = None


@dataclass 
class LLMSettingsConfig:
    """LLM设置配置"""
    provider: str = "openai"
    primary_model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4000
    insight_model: str = "gpt-4o"
    # 可选：OpenAI 兼容中转 base_url（或供应商自定义 endpoint）
    base_url: Optional[str] = None


@dataclass
class EmbeddingSettingsConfig:
    """嵌入设置配置"""
    provider: str = "openai"
    model: str = "text-embedding-3-large"
    # 是否启用 Cross-Encoder 重排序（用于检索精排，SOTA 常见做法）
    enable_reranker: bool = False
    # Cross-Encoder 模型名称（为 None 时使用默认推荐模型）
    reranker_model: Optional[str] = None
    # 可选：OpenAI 兼容中转 base_url（或供应商自定义 endpoint）
    base_url: Optional[str] = None


@dataclass
class VectorDBConfig:
    """向量数据库配置"""
    provider: str = "chromadb"
    path: str = "./data/memory/chroma_db"
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    collection_name: str = "deep_scholar_papers"
    chunk_collection_name: str = "deep_scholar_chunks"


@dataclass
class KnowledgeGraphConfig:
    """知识图谱配置"""
    provider: str = "networkx"
    file_path: str = "./data/memory/knowledge_graph.graphml"
    uri: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    # 是否导出 GraphML（用于可视化/互操作）。关闭可避免多重边语义弱化，仅使用 GPickle 主存。
    export_graphml: bool = True


@dataclass
class PathsConfig:
    """路径配置"""
    drafts_folder: str = "./output/drafts"
    logs_folder: str = "./logs"
    watch_folder: str = "./input/papers_to_read"
    temp_downloads_folder: str = "./data/temp_downloads"
    prompt_templates_folder: str = "./prompts"


@dataclass
class PreferencesConfig:
    """偏好配置"""
    initial_keywords: List[str] = field(default_factory=lambda: [
        'Large Language Models',
        'Agentic AI',
        'Mixture of Experts',
        'World Models',
        'Diffusion Models',
        'Reinforcement Learning',
        'Neuroscience',
        'Brain-inspired AI'
    ])
    seed_papers: List[str] = field(default_factory=lambda: [
        "https://arxiv.org/abs/1706.03762",  # Attention Is All You Need
        "https://arxiv.org/abs/2005.14165",  # GPT-3
    ])
    twitter_handles_list: List[str] = field(default_factory=lambda: [
        "karpathy",
        "ylecun", 
        "demishassabis"
    ])
    exploration_epsilon: float = 0.1
    # 每日论文筛选上限（供发现流程使用）
    daily_paper_limit: int = 5


@dataclass
class PDFProcessingConfig:
    """PDF处理配置"""
    nougat_model_path: Optional[str] = None
    nougat_timeout_seconds: int = 300
    pymupdf_timeout_seconds: int = 60


@dataclass
class ConferenceSource:
    """会议源配置"""
    name: str
    url: str
    parser: str


@dataclass
class SourcesConfig:
    """信息源配置"""
    arxiv_categories: List[str] = field(default_factory=lambda: ["cs.AI", "cs.LG", "cs.CL", "cs.CV"])
    top_conference_urls: List[ConferenceSource] = field(default_factory=list)
    social_media_query_interval_hours: int = 6


@dataclass
class OutputConfig:
    """输出配置"""
    max_article_word_count: int = 5000
    min_article_word_count: int = 2000
    markdown_format_version: str = "wechat_md"


@dataclass
class FeedbackConfig:
    """反馈配置"""
    positive_weight_increase: float = 0.2
    negative_weight_decrease: float = 0.05
    explicit_feedback_impact: float = 0.3


@dataclass
class LearningConfig:
    """持续学习与探索配置（里程碑C）。"""
    enable: bool = True
    epsilon_min: float = 0.05
    epsilon_max: float = 0.30
    epsilon_step: float = 0.02
    quality_threshold_good: float = 0.70
    recent_window: int = 100


@dataclass
class PerformanceConfig:
    """性能配置"""
    max_concurrent_downloads: int = 5
    max_concurrent_llm_calls: int = 3
    api_retry_attempts: int = 5
    api_retry_delay_seconds: int = 2


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "./logs/deep_scholar.log"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    enable_structured_logging: bool = True


class ConfigManager:
    """
    中心化配置管理器
    
    遵循严格的依赖注入和无副作用原则：
    - 构造函数绝不执行任何I/O操作
    - 所有初始化必须通过 initialize() 方法显式完成
    - 提供强类型访问，禁止字典式访问
    """
    
    def __init__(self):
        """
        无副作用构造函数
        
        仅初始化空的配置容器，不执行任何I/O操作
        """
        self.api_keys: Optional[APIKeysConfig] = None
        self.llm_settings: Optional[LLMSettingsConfig] = None
        self.embedding_settings: Optional[EmbeddingSettingsConfig] = None
        self.vector_db: Optional[VectorDBConfig] = None
        self.knowledge_graph: Optional[KnowledgeGraphConfig] = None
        self.paths: Optional[PathsConfig] = None
        self.preferences: Optional[PreferencesConfig] = None
        self.pdf_processing: Optional[PDFProcessingConfig] = None
        self.sources: Optional[SourcesConfig] = None
        self.output: Optional[OutputConfig] = None
        self.feedback: Optional[FeedbackConfig] = None
        self.learning: Optional[LearningConfig] = None
        self.performance: Optional[PerformanceConfig] = None
        self.logging: Optional[LoggingConfig] = None
        
        self._initialized = False
        self._logger: Optional[logging.Logger] = None
    
    async def initialize(self, config_file_path: str) -> None:
        """
        异步初始化配置管理器
        
        Args:
            config_file_path: 配置文件路径
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: 配置文件格式错误
            ValueError: 配置验证失败
        """
        if self._initialized:
            return
            
        try:
            # 异步读取配置文件
            config_path = Path(config_file_path).resolve()
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            async with aiofiles.open(config_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                config_data = yaml.safe_load(content)
            
            # 加载各个配置部分
            await self._load_configurations(config_data)
            
            # 处理环境变量
            await self._process_environment_variables()
            
            # 验证配置
            await self._validate_configurations()
            
            # 创建必要的目录
            await self._ensure_directories()
            
            self._initialized = True
            
        except FileNotFoundError:
            # 让文件不存在错误按原样冒泡，符合调用方与测试预期
            raise
        except yaml.YAMLError:
            # YAML 解析错误也直接冒泡
            raise
        except Exception as e:
            # 其他未知错误统一包装为 ValueError，提供一致的错误信息
            raise ValueError(f"Failed to initialize configuration: {e}") from e
    
    async def _load_configurations(self, config_data: Dict[str, Any]) -> None:
        """加载各个配置部分"""
        # API Keys
        api_keys_data = config_data.get('api_keys', {})
        self.api_keys = APIKeysConfig(**api_keys_data)
        
        # LLM Settings
        llm_settings_data = config_data.get('llm_settings', {})
        self.llm_settings = LLMSettingsConfig(**llm_settings_data)
        
        # Embedding Settings
        embedding_settings_data = config_data.get('embedding_settings', {})
        self.embedding_settings = EmbeddingSettingsConfig(**embedding_settings_data)
        
        # Vector DB
        vector_db_data = config_data.get('vector_db', {})
        self.vector_db = VectorDBConfig(**vector_db_data)
        
        # Knowledge Graph
        knowledge_graph_data = config_data.get('knowledge_graph', {})
        self.knowledge_graph = KnowledgeGraphConfig(**knowledge_graph_data)
        
        # Paths
        paths_data = config_data.get('paths', {})
        self.paths = PathsConfig(**paths_data)
        
        # Preferences
        preferences_data = config_data.get('preferences', {})
        self.preferences = PreferencesConfig(**preferences_data)
        
        # PDF Processing
        pdf_processing_data = config_data.get('pdf_processing', {})
        self.pdf_processing = PDFProcessingConfig(**pdf_processing_data)
        
        # Sources
        sources_data = config_data.get('sources', {})
        
        # 处理会议源配置
        conference_urls_data = sources_data.get('top_conference_urls', [])
        conference_sources = []
        for conf in conference_urls_data:
            conference_sources.append(ConferenceSource(
                name=conf['name'],
                url=conf['url'],
                parser=conf['parser']
            ))
        
        sources_data['top_conference_urls'] = conference_sources
        self.sources = SourcesConfig(**sources_data)
        
        # Output
        output_data = config_data.get('output', {})
        self.output = OutputConfig(**output_data)
        
        # Feedback
        feedback_data = config_data.get('feedback', {})
        self.feedback = FeedbackConfig(**feedback_data)

        # Learning
        learning_data = config_data.get('learning', {})
        self.learning = LearningConfig(**learning_data)
        
        # Performance
        performance_data = config_data.get('performance', {})
        self.performance = PerformanceConfig(**performance_data)
        
        # Logging
        logging_data = config_data.get('logging', {})
        self.logging = LoggingConfig(**logging_data)
    
    async def _process_environment_variables(self) -> None:
        """处理环境变量，优先级：配置文件 > 环境变量"""
        if self.api_keys:
            # API Keys优先从环境变量获取敏感信息
            self.api_keys.openai = self.api_keys.openai or os.getenv('OPENAI_API_KEY')
            self.api_keys.google_gemini = self.api_keys.google_gemini or os.getenv('GOOGLE_GEMINI_API_KEY')
            self.api_keys.anthropic = self.api_keys.anthropic or os.getenv('ANTHROPIC_API_KEY')
            self.api_keys.twitter_bearer_token = self.api_keys.twitter_bearer_token or os.getenv('TWITTER_BEARER_TOKEN')
    
    async def _validate_configurations(self) -> None:
        """验证配置的有效性"""
        errors = []
        
        # 验证必需的配置
        if not self.api_keys or not (self.api_keys.openai or self.api_keys.google_gemini or self.api_keys.anthropic):
            errors.append("At least one LLM API key must be provided")
        
        if not self.llm_settings or not self.llm_settings.provider:
            errors.append("LLM provider must be specified")
        
        if not self.embedding_settings or not self.embedding_settings.provider:
            errors.append("Embedding provider must be specified")
        
        # 验证路径配置
        if self.paths:
            for attr_name in ['drafts_folder', 'logs_folder', 'watch_folder', 'temp_downloads_folder', 'prompt_templates_folder']:
                path_value = getattr(self.paths, attr_name)
                if not path_value:
                    errors.append(f"Path configuration '{attr_name}' cannot be empty")
        
        # 验证数值配置
        if self.performance:
            if self.performance.max_concurrent_downloads <= 0:
                errors.append("max_concurrent_downloads must be positive")
            if self.performance.max_concurrent_llm_calls <= 0:
                errors.append("max_concurrent_llm_calls must be positive")

        # 学习参数校验
        if self.learning:
            if self.learning.epsilon_min < 0 or self.learning.epsilon_max > 1 or self.learning.epsilon_min > self.learning.epsilon_max:
                errors.append("learning epsilon ranges must satisfy 0<=min<=max<=1")
            if self.learning.epsilon_step <= 0:
                errors.append("learning.epsilon_step must be positive")
            if not (0 <= self.learning.quality_threshold_good <= 1):
                errors.append("learning.quality_threshold_good must be within [0,1]")
            if self.learning.recent_window <= 0:
                errors.append("learning.recent_window must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    async def _ensure_directories(self) -> None:
        """确保必要的目录存在"""
        if not self.paths:
            return
            
        directories = [
            self.paths.drafts_folder,
            self.paths.logs_folder,
            self.paths.watch_folder,
            self.paths.temp_downloads_folder,
            self.paths.prompt_templates_folder
        ]
        
        for directory in directories:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
        
        # 确保数据目录存在
        if self.vector_db and self.vector_db.path:
            Path(self.vector_db.path).parent.mkdir(parents=True, exist_ok=True)
        
        if self.knowledge_graph and self.knowledge_graph.file_path:
            Path(self.knowledge_graph.file_path).parent.mkdir(parents=True, exist_ok=True)
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        安全获取API密钥
        
        Args:
            provider: API提供商名称
            
        Returns:
            API密钥或None
        """
        if not self.api_keys:
            return None
            
        provider_lower = provider.lower()
        if provider_lower == 'openai':
            return self.api_keys.openai
        elif provider_lower in ['google', 'google_gemini', 'gemini']:
            return self.api_keys.google_gemini
        elif provider_lower == 'anthropic':
            return self.api_keys.anthropic
        elif provider_lower in ['twitter', 'x']:
            return self.api_keys.twitter_bearer_token
        else:
            return None
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """
        获取绝对路径
        
        Args:
            relative_path: 相对路径
            
        Returns:
            绝对路径
        """
        return Path(relative_path).resolve()
