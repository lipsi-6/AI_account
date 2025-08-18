"""
学术论文发现与品鉴引擎

实现多源学术论文发现和基于SOTA嵌入模型的智能品味引擎：
- ArxivSource: 使用arxiv Python库的异步实现
- ConferenceSource: 真正的会议网站爬虫
- SocialSource: Twitter/X官方API v2实现
- ManualIngestion: 文件夹监听机制
- TasteEngine: 基于嵌入向量的持久化学术品味引擎

严格遵循依赖注入原则，无副作用构造函数，显式异步初始化。
"""

import asyncio
import time
import json
import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set, Tuple, AsyncGenerator
from enum import Enum
from urllib.parse import urljoin, urlparse
import uuid
import random
import math

from collections import defaultdict

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

from .config_manager import ConfigManager
from .logger import get_logger, log_api_call, log_performance
from .embedding_provider import EmbeddingProviderService
from .llm_provider import LLMProviderService
from .memory_system import HybridMemorySystem, NodeType, EdgeType
from .learning_loop import LearningLoop


class SourceType(Enum):
    """信息源类型"""
    ARXIV = "arxiv"
    CONFERENCE = "conference"
    SOCIAL = "social"
    MANUAL = "manual"


class PaperQuality(Enum):
    """论文质量评级"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    UNKNOWN = "unknown"


@dataclass
class Author:
    """作者信息"""
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None


@dataclass
class PaperMetadata:
    """论文元数据"""
    title: str
    authors: List[Author] = field(default_factory=list)
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    publication_date: Optional[datetime] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pdf_url: Optional[str] = None
    html_url: Optional[str] = None
    citation_count: Optional[int] = None
    categories: List[str] = field(default_factory=list)
    source_type: Optional[SourceType] = None
    source_url: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveredPaper:
    """发现的论文"""
    id: str
    metadata: PaperMetadata
    content: Optional[str] = None
    discovery_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    quality_score: Optional[float] = None
    quality_rating: PaperQuality = PaperQuality.UNKNOWN
    relevance_score: Optional[float] = None
    taste_feedback: Optional[bool] = None
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class TasteVector:
    """学术品味向量"""
    interests: List[float]  # 兴趣嵌入向量
    weights: Dict[str, float] = field(default_factory=dict)  # 各维度权重
    keywords: List[str] = field(default_factory=list)  # 关键词
    exploration_factor: float = 0.1  # 探索因子
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    positive_examples: List[str] = field(default_factory=list)  # 正例论文ID
    negative_examples: List[str] = field(default_factory=list)  # 负例论文ID


class DiscoveryError(Exception):
    """发现引擎错误基类"""
    pass


class SourceError(DiscoveryError):
    """信息源错误"""
    pass


class TasteEngineError(DiscoveryError):
    """品味引擎错误"""
    pass


class BaseDiscoverySource(ABC):
    """发现源基类"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = get_logger(f"discovery_source_{self.__class__.__name__.lower()}")
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化源"""
        pass
    
    @abstractmethod
    async def discover_papers(self, query: Optional[str] = None, limit: int = 50) -> List[PaperMetadata]:
        """发现论文"""
        pass
    
    @abstractmethod
    def get_source_type(self) -> SourceType:
        """获取源类型"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """关闭源"""
        pass


class ArxivSource(BaseDiscoverySource):
    """Arxiv论文源 - 使用arxiv Python库"""
    
    def __init__(self, config: ConfigManager):
        super().__init__(config)
        if not ARXIV_AVAILABLE:
            raise SourceError("arxiv library not available. Please install: pip install arxiv")
        
        self.client: Optional[arxiv.Client] = None
        self.categories = config.sources.arxiv_categories if config.sources else ["cs.AI", "cs.LG"]
        
    async def initialize(self) -> None:
        """初始化Arxiv客户端"""
        if self._initialized:
            return
        
        try:
            # 创建异步友好的arxiv客户端
            self.client = arxiv.Client(
                page_size=50,
                delay_seconds=3,  # 遵守arxiv速率限制
                num_retries=3
            )
            
            self._initialized = True
            self.logger.info(f"Arxiv source initialized with categories: {self.categories}")
            
        except Exception as e:
            raise SourceError(f"Failed to initialize Arxiv source: {e}")
    
    async def discover_papers(self, query: Optional[str] = None, limit: int = 50) -> List[PaperMetadata]:
        """从Arxiv发现论文"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # 构建搜索查询
            if query:
                # 自定义查询
                search_query = query
            else:
                # 基于配置的类别搜索最新论文
                category_query = " OR ".join([f"cat:{cat}" for cat in self.categories])
                search_query = f"({category_query}) AND submittedDate:[{self._get_recent_date()} TO *]"
            
            # 执行异步搜索
            papers = await self._async_search(search_query, limit)
            
            # 转换为PaperMetadata
            paper_list = []
            for paper in papers:
                metadata = await self._convert_arxiv_paper(paper)
                paper_list.append(metadata)
            
            duration = time.time() - start_time
            log_performance("arxiv_discovery", duration, papers_count=len(paper_list))
            
            self.logger.info(f"Discovered {len(paper_list)} papers from Arxiv in {duration:.2f}s")
            return paper_list
            
        except Exception as e:
            self.logger.error(f"Arxiv discovery failed: {e}")
            raise SourceError(f"Arxiv discovery error: {e}")
    
    async def _async_search(self, query: str, limit: int) -> List[arxiv.Result]:
        """异步执行arxiv搜索"""
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        # 在线程池中执行同步搜索
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: list(self.client.results(search)))
        
        return results
    
    async def _convert_arxiv_paper(self, paper: arxiv.Result) -> PaperMetadata:
        """转换arxiv论文为PaperMetadata"""
        authors = [Author(name=author.name) for author in paper.authors]
        
        # 提取分类（统一为可序列化字符串）
        categories = [getattr(cat, 'term', str(cat)) for cat in getattr(paper, 'categories', [])]
        
        # 提取arxiv ID
        arxiv_id = paper.entry_id.split('/')[-1]
        if 'v' in arxiv_id:
            arxiv_id = arxiv_id.split('v')[0]
        
        return PaperMetadata(
            title=paper.title,
            authors=authors,
            abstract=paper.summary,
            publication_date=paper.published,
            arxiv_id=arxiv_id,
            pdf_url=paper.pdf_url,
            html_url=paper.entry_id,
            categories=categories,
            source_type=SourceType.ARXIV,
            source_url=paper.entry_id,
            raw_data={
                "arxiv_result": {
                    "entry_id": paper.entry_id,
                    "updated": paper.updated.isoformat() if paper.updated else None,
                    "published": paper.published.isoformat() if paper.published else None,
                    "comment": paper.comment,
                    "journal_ref": paper.journal_ref,
                    "doi": paper.doi,
                    "primary_category": paper.primary_category
                }
            }
        )
    
    def _get_recent_date(self, days_back: int = 7) -> str:
        """获取最近日期（用于搜索）"""
        recent_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        return recent_date.strftime("%Y%m%d%H%M%S")
    
    def get_source_type(self) -> SourceType:
        return SourceType.ARXIV
    
    async def close(self) -> None:
        """关闭Arxiv源"""
        self._initialized = False
        self.logger.info("Arxiv source closed")


class ConferenceSource(BaseDiscoverySource):
    """会议论文源 - 真正的网站爬虫实现"""
    
    def __init__(self, config: ConfigManager):
        super().__init__(config)
        if not AIOHTTP_AVAILABLE:
            raise SourceError("aiohttp library not available. Please install: pip install aiohttp")
        if not BS4_AVAILABLE:
            raise SourceError("beautifulsoup4 library not available. Please install: pip install beautifulsoup4")
        if not FEEDPARSER_AVAILABLE:
            raise SourceError("feedparser library not available. Please install: pip install feedparser")
        
        self.session: Optional['aiohttp.ClientSession'] = None
        self.conference_configs = config.sources.top_conference_urls if config.sources else []
    
    async def initialize(self) -> None:
        """初始化会议源"""
        if self._initialized:
            return
        
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
            )
            
            self._initialized = True
            self.logger.info(f"Conference source initialized with {len(self.conference_configs)} conferences")
            
        except Exception as e:
            raise SourceError(f"Failed to initialize Conference source: {e}")
    
    async def discover_papers(self, query: Optional[str] = None, limit: int = 50) -> List[PaperMetadata]:
        """从会议网站发现论文"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        papers = []
        
        try:
            # 并行爬取所有配置的会议
            tasks = []
            for conf_config in self.conference_configs:
                task = self._crawl_conference(conf_config, limit // len(self.conference_configs) + 1)
                tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, list):
                        papers.extend(result)
                    elif isinstance(result, Exception):
                        self.logger.warning(f"Conference crawling error: {result}")
            
            # 限制总数
            papers = papers[:limit]
            
            duration = time.time() - start_time
            log_performance("conference_discovery", duration, papers_count=len(papers))
            
            self.logger.info(f"Discovered {len(papers)} papers from conferences in {duration:.2f}s")
            return papers
            
        except Exception as e:
            self.logger.error(f"Conference discovery failed: {e}")
            raise SourceError(f"Conference discovery error: {e}")
    
    async def _crawl_conference(self, conf_config, limit: int) -> List[PaperMetadata]:
        """爬取单个会议"""
        try:
            parser_name = conf_config.parser
            if parser_name == "neurips":
                return await self._parse_neurips(conf_config.url, limit)
            elif parser_name == "icml":
                return await self._parse_icml(conf_config.url, limit)
            elif parser_name == "iclr":
                return await self._parse_iclr(conf_config.url, limit)
            elif parser_name == "generic_rss":
                return await self._parse_generic_rss(conf_config.url, limit)
            else:
                return await self._parse_generic_html(conf_config.url, limit)
                
        except Exception as e:
            self.logger.warning(f"Failed to crawl {conf_config.name}: {e}")
            return []
    
    async def _parse_neurips(self, url: str, limit: int) -> List[PaperMetadata]:
        """解析NeurIPS会议页面"""
        papers = []
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return papers
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # NeurIPS特定的HTML解析逻辑
                paper_elements = soup.find_all('div', class_='paper')
                
                for element in paper_elements[:limit]:
                    title_elem = element.find('h4') or element.find('a', class_='paper-title')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    
                    # 提取作者
                    authors = []
                    authors_elem = element.find('div', class_='authors') or element.find('span', class_='authors')
                    if authors_elem:
                        author_text = authors_elem.get_text(strip=True)
                        for author_name in author_text.split(','):
                            authors.append(Author(name=author_name.strip()))
                    
                    # 提取摘要链接
                    abstract_link = None
                    link_elem = element.find('a', href=True)
                    if link_elem:
                        abstract_link = urljoin(url, link_elem['href'])
                    
                    paper = PaperMetadata(
                        title=title,
                        authors=authors,
                        venue="NeurIPS",
                        source_type=SourceType.CONFERENCE,
                        source_url=abstract_link or url,
                        html_url=abstract_link,
                        raw_data={"parser": "neurips", "original_url": url}
                    )
                    
                    papers.append(paper)
                    
        except Exception as e:
            self.logger.error(f"NeurIPS parsing error: {e}")
        
        return papers
    
    async def _parse_icml(self, url: str, limit: int) -> List[PaperMetadata]:
        """解析ICML会议页面"""
        papers = []
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return papers
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # ICML特定的HTML解析逻辑
                paper_elements = soup.find_all('div', class_='paper-container') or soup.find_all('article')
                
                for element in paper_elements[:limit]:
                    title_elem = element.find('h3') or element.find('h2') or element.find('a', class_='title')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    
                    # 提取作者
                    authors = []
                    authors_elem = element.find('div', class_='authors') or element.find('p', class_='authors')
                    if authors_elem:
                        author_text = authors_elem.get_text(strip=True)
                        for author_name in re.split(r'[,;&]', author_text):
                            if author_name.strip():
                                authors.append(Author(name=author_name.strip()))
                    
                    # 提取PDF链接
                    pdf_link = None
                    pdf_elem = element.find('a', href=re.compile(r'\.pdf$', re.I))
                    if pdf_elem:
                        pdf_link = urljoin(url, pdf_elem['href'])
                    
                    paper = PaperMetadata(
                        title=title,
                        authors=authors,
                        venue="ICML",
                        source_type=SourceType.CONFERENCE,
                        source_url=url,
                        pdf_url=pdf_link,
                        raw_data={"parser": "icml", "original_url": url}
                    )
                    
                    papers.append(paper)
                    
        except Exception as e:
            self.logger.error(f"ICML parsing error: {e}")
        
        return papers
    
    async def _parse_iclr(self, url: str, limit: int) -> List[PaperMetadata]:
        """解析ICLR会议页面"""
        papers = []
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return papers
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # ICLR使用OpenReview平台
                paper_elements = soup.find_all('div', class_='note') or soup.find_all('li', class_='note')
                
                for element in paper_elements[:limit]:
                    title_elem = element.find('h4', class_='note-content-title') or element.find('strong')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    
                    # 提取作者信息
                    authors = []
                    authors_elem = element.find('div', class_='note-content-authors')
                    if authors_elem:
                        author_links = authors_elem.find_all('a')
                        for link in author_links:
                            authors.append(Author(name=link.get_text(strip=True)))
                    
                    # 提取摘要
                    abstract = None
                    abstract_elem = element.find('div', class_='note-content-abstract')
                    if abstract_elem:
                        abstract = abstract_elem.get_text(strip=True)
                    
                    # 提取OpenReview链接
                    paper_link = None
                    link_elem = element.find('a', class_='note-content-title')
                    if link_elem and link_elem.get('href'):
                        paper_link = urljoin(url, link_elem['href'])
                    
                    paper = PaperMetadata(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        venue="ICLR",
                        source_type=SourceType.CONFERENCE,
                        source_url=paper_link or url,
                        html_url=paper_link,
                        raw_data={"parser": "iclr", "original_url": url}
                    )
                    
                    papers.append(paper)
                    
        except Exception as e:
            self.logger.error(f"ICLR parsing error: {e}")
        
        return papers
    
    async def _parse_generic_rss(self, url: str, limit: int) -> List[PaperMetadata]:
        """解析通用RSS源"""
        papers = []
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return papers
                
                rss_content = await response.text()
                
                # 在线程池中解析RSS（feedparser是同步的）
                loop = asyncio.get_event_loop()
                feed = await loop.run_in_executor(None, feedparser.parse, rss_content)
                
                for entry in feed.entries[:limit]:
                    title = getattr(entry, 'title', '')
                    summary = getattr(entry, 'summary', '')
                    link = getattr(entry, 'link', '')
                    published = getattr(entry, 'published_parsed', None)
                    
                    pub_date = None
                    if published:
                        pub_date = datetime(*published[:6], tzinfo=timezone.utc)
                    
                    # 从作者字段提取作者
                    authors = []
                    if hasattr(entry, 'author'):
                        authors.append(Author(name=entry.author))
                    elif hasattr(entry, 'authors'):
                        for author_info in entry.authors:
                            if isinstance(author_info, dict) and 'name' in author_info:
                                authors.append(Author(name=author_info['name']))
                            elif isinstance(author_info, str):
                                authors.append(Author(name=author_info))
                    
                    paper = PaperMetadata(
                        title=title,
                        authors=authors,
                        abstract=summary,
                        publication_date=pub_date,
                        source_type=SourceType.CONFERENCE,
                        source_url=link,
                        html_url=link,
                        raw_data={"parser": "generic_rss", "rss_url": url}
                    )
                    
                    papers.append(paper)
                    
        except Exception as e:
            self.logger.error(f"RSS parsing error: {e}")
        
        return papers
    
    async def _parse_generic_html(self, url: str, limit: int) -> List[PaperMetadata]:
        """通用HTML解析器"""
        papers = []
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return papers
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # 尝试通用的论文结构模式
                potential_elements = (
                    soup.find_all('div', class_=re.compile(r'paper|article|item', re.I)) +
                    soup.find_all('article') +
                    soup.find_all('li', class_=re.compile(r'paper|publication', re.I))
                )
                
                for element in potential_elements[:limit]:
                    # 查找标题
                    title_elem = (
                        element.find('h1') or element.find('h2') or element.find('h3') or
                        element.find('h4') or element.find('a', class_=re.compile(r'title', re.I)) or
                        element.find('strong')
                    )
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    if len(title) < 10:  # 过滤太短的标题
                        continue
                    
                    # 查找作者
                    authors = []
                    authors_elem = element.find(['div', 'p', 'span'], class_=re.compile(r'author', re.I))
                    if authors_elem:
                        author_text = authors_elem.get_text(strip=True)
                        for author_name in re.split(r'[,;&]', author_text):
                            if author_name.strip() and len(author_name.strip()) > 2:
                                authors.append(Author(name=author_name.strip()))
                    
                    # 查找摘要
                    abstract = None
                    abstract_elem = element.find(['div', 'p'], class_=re.compile(r'abstract|summary', re.I))
                    if abstract_elem:
                        abstract = abstract_elem.get_text(strip=True)
                    
                    # 查找链接
                    paper_link = None
                    link_elem = element.find('a', href=True)
                    if link_elem:
                        paper_link = urljoin(url, link_elem['href'])
                    
                    paper = PaperMetadata(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        source_type=SourceType.CONFERENCE,
                        source_url=paper_link or url,
                        html_url=paper_link,
                        raw_data={"parser": "generic_html", "original_url": url}
                    )
                    
                    papers.append(paper)
                    
        except Exception as e:
            self.logger.error(f"Generic HTML parsing error: {e}")
        
        return papers
    
    def get_source_type(self) -> SourceType:
        return SourceType.CONFERENCE
    
    async def close(self) -> None:
        """关闭会议源"""
        if self.session and not self.session.closed:
            await self.session.close()
        
        self._initialized = False
        self.logger.info("Conference source closed")


class SocialSource(BaseDiscoverySource):
    """社交媒体论文源 - Twitter/X官方API v2实现"""
    
    def __init__(self, config: ConfigManager):
        super().__init__(config)
        if not AIOHTTP_AVAILABLE:
            raise SourceError("aiohttp library not available. Please install: pip install aiohttp")
        
        self.session: Optional['aiohttp.ClientSession'] = None
        self.bearer_token = config.get_api_key("twitter")
        self.twitter_handles = config.preferences.twitter_handles_list if config.preferences else []
        
        if not self.bearer_token:
            raise SourceError("Twitter Bearer Token is required for SocialSource")
    
    async def initialize(self) -> None:
        """初始化社交媒体源"""
        if self._initialized:
            return
        
        try:
            self.session = aiohttp.ClientSession(
                headers={
                    'Authorization': f'Bearer {self.bearer_token}',
                    'Content-Type': 'application/json'
                },
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            self._initialized = True
            self.logger.info(f"Social source initialized with {len(self.twitter_handles)} handles")
            
        except Exception as e:
            raise SourceError(f"Failed to initialize Social source: {e}")
    
    async def discover_papers(self, query: Optional[str] = None, limit: int = 50) -> List[PaperMetadata]:
        """从社交媒体发现论文"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        papers = []
        
        try:
            if query:
                # 基于查询的搜索
                search_papers = await self._search_tweets(query, limit)
                papers.extend(search_papers)
            else:
                # 基于关注列表的发现
                for handle in self.twitter_handles[:10]:  # 限制handle数量以避免超出API限制
                    handle_papers = await self._get_user_tweets(handle, limit // len(self.twitter_handles) + 1)
                    papers.extend(handle_papers)
            
            # 去重并限制数量
            seen_titles = set()
            unique_papers = []
            for paper in papers:
                if paper.title not in seen_titles:
                    seen_titles.add(paper.title)
                    unique_papers.append(paper)
            
            papers = unique_papers[:limit]
            
            duration = time.time() - start_time
            log_performance("social_discovery", duration, papers_count=len(papers))
            
            self.logger.info(f"Discovered {len(papers)} papers from social media in {duration:.2f}s")
            return papers
            
        except Exception as e:
            self.logger.error(f"Social discovery failed: {e}")
            raise SourceError(f"Social discovery error: {e}")
    
    async def _search_tweets(self, query: str, limit: int) -> List[PaperMetadata]:
        """搜索包含论文的推文"""
        papers = []
        
        try:
            # 构建搜索查询，寻找包含arxiv或论文链接的推文
            search_query = f"({query}) (arxiv.org OR doi.org OR pdf) -is:retweet"
            
            fetched = 0
            next_token = None
            # 简单速率保护：每次循环后小睡，避免打在速率墙上
            while fetched < limit:
                page_size = min(100, limit - fetched)
                params = {
                    'query': search_query,
                    'max_results': page_size,
                    'tweet.fields': 'created_at,author_id,text,public_metrics,entities',
                    'user.fields': 'name,username,description',
                    'expansions': 'author_id'
                }
                if next_token:
                    params['next_token'] = next_token
                async with self.session.get('https://api.twitter.com/2/tweets/search/recent', params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        page = await self._extract_papers_from_tweets(data)
                        papers.extend(page)
                        fetched += len(page)
                        next_token = data.get('meta', {}).get('next_token')
                        if not next_token:
                            break
                        # 轻微退避
                        await asyncio.sleep(0.3)
                    else:
                        self.logger.warning(f"Twitter search failed: {response.status}")
                        break
                    
        except Exception as e:
            self.logger.warning(f"Twitter search error: {e}")
        
        return papers
    
    async def _get_user_tweets(self, username: str, limit: int) -> List[PaperMetadata]:
        """获取特定用户的推文中的论文"""
        papers = []
        
        try:
            # 首先获取用户ID
            user_response = await self.session.get(
                f'https://api.twitter.com/2/users/by/username/{username}'
            )
            
            if user_response.status != 200:
                return papers
            
            user_data = await user_response.json()
            user_id = user_data['data']['id']
            
            # 获取用户推文
            fetched = 0
            next_token = None
            while fetched < limit:
                page_size = min(100, limit - fetched)
                params = {
                    'max_results': page_size,
                    'tweet.fields': 'created_at,text,public_metrics,entities',
                    'exclude': 'retweets,replies'
                }
                if next_token:
                    params['pagination_token'] = next_token
                async with self.session.get(f'https://api.twitter.com/2/users/{user_id}/tweets', params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        page = await self._extract_papers_from_tweets(data, username)
                        papers.extend(page)
                        fetched += len(page)
                        # 去冗余：唯一来源的分页令牌
                        next_token = data.get('meta', {}).get('next_token')
                        if not next_token:
                            break
                        await asyncio.sleep(0.3)
                    else:
                        self.logger.warning(f"Twitter user timeline failed: {response.status}")
                        break
                    
        except Exception as e:
            self.logger.warning(f"Error getting tweets for {username}: {e}")
        
        return papers
    
    async def _extract_papers_from_tweets(self, tweet_data: Dict[str, Any], username: Optional[str] = None) -> List[PaperMetadata]:
        """从推文数据中提取论文信息"""
        papers = []
        
        try:
            tweets = tweet_data.get('data', [])
            users = {user['id']: user for user in tweet_data.get('includes', {}).get('users', [])}
            
            for tweet in tweets:
                text = tweet.get('text', '')
                
                # 查找arxiv链接
                arxiv_urls = re.findall(r'https?://arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', text)
                
                # 查找DOI
                dois = re.findall(r'doi\.org/([^/\s]+/[^\s]+)', text)
                
                # 查找一般的PDF链接
                pdf_urls = re.findall(r'https?://[^\s]+\.pdf', text)
                
                if arxiv_urls or dois or pdf_urls:
                    # 尝试从推文中提取论文标题
                    title = self._extract_title_from_tweet(text)
                    
                    if not title:
                        continue
                    
                    # 获取作者信息（推文作者）
                    author_id = tweet.get('author_id')
                    author_name = username
                    if author_id in users:
                        author_name = users[author_id].get('name', username)
                    
                    authors = [Author(name=author_name)] if author_name else []
                    
                    # 创建论文元数据
                    paper = PaperMetadata(
                        title=title,
                        authors=authors,
                        source_type=SourceType.SOCIAL,
                        source_url=f"https://twitter.com/i/status/{tweet['id']}",
                        publication_date=datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                        raw_data={
                            "tweet_text": text,
                            "tweet_id": tweet['id'],
                            "arxiv_ids": arxiv_urls,
                            "dois": dois,
                            "pdf_urls": pdf_urls,
                            "metrics": tweet.get('public_metrics', {})
                        }
                    )
                    
                    # 设置相关URL
                    if arxiv_urls:
                        paper.arxiv_id = arxiv_urls[0]
                        paper.pdf_url = f"https://arxiv.org/pdf/{arxiv_urls[0]}.pdf"
                        paper.html_url = f"https://arxiv.org/abs/{arxiv_urls[0]}"
                    elif pdf_urls:
                        paper.pdf_url = pdf_urls[0]
                    
                    papers.append(paper)
                    
        except Exception as e:
            self.logger.error(f"Error extracting papers from tweets: {e}")
        
        return papers
    
    def _extract_title_from_tweet(self, text: str) -> Optional[str]:
        """从推文中提取可能的论文标题"""
        # 移除URL
        text = re.sub(r'https?://[^\s]+', '', text)
        
        # 查找引号中的内容（可能是标题）
        quoted_matches = re.findall(r'"([^"]{20,})"', text)
        if quoted_matches:
            return quoted_matches[0].strip()
        
        # 查找以大写字母开头的长短语
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                sentence[0].isupper() and 
                not sentence.startswith('http') and
                not sentence.startswith('@')):
                return sentence
        
        # 如果没有找到合适的标题，返回清理后的推文前部分
        clean_text = re.sub(r'[@#]\w+', '', text).strip()
        if len(clean_text) > 20:
            return clean_text[:100] + ('...' if len(clean_text) > 100 else '')
        
        return None
    
    def get_source_type(self) -> SourceType:
        return SourceType.SOCIAL
    
    async def close(self) -> None:
        """关闭社交媒体源"""
        if self.session and not self.session.closed:
            await self.session.close()
        
        self._initialized = False
        self.logger.info("Social source closed")


if WATCHDOG_AVAILABLE:
    class ManualIngestionHandler(FileSystemEventHandler):
        """手动摄入文件系统事件处理器"""
        
        def __init__(self, callback):
            self.callback = callback
            self.logger = get_logger("manual_ingestion_handler")
        
        def on_created(self, event):
            if not event.is_directory and event.src_path.endswith('.pdf'):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(self.callback(event.src_path), loop)
                except RuntimeError:
                    pass
        
        def on_modified(self, event):
            if not event.is_directory and event.src_path.endswith('.pdf'):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(self.callback(event.src_path), loop)
                except RuntimeError:
                    pass


if WATCHDOG_AVAILABLE:
    class ManualIngestionSource(BaseDiscoverySource):
        """手动摄入源 - 监听指定文件夹"""
        
        def __init__(self, config: ConfigManager):
            super().__init__(config)
            # WATCHDOG_AVAILABLE 已经保证为 True
            
            self.watch_folder = Path(config.paths.watch_folder) if config.paths else Path("./input/papers_to_read")
            self.observer: Optional['Observer'] = None
            self.discovered_papers: List[PaperMetadata] = []
            self._papers_lock = asyncio.Lock()
        
        async def initialize(self) -> None:
            """初始化手动摄入源"""
            if self._initialized:
                return
            
            try:
                # 确保监听文件夹存在
                self.watch_folder.mkdir(parents=True, exist_ok=True)
                
                # 设置文件系统监听
                event_handler = ManualIngestionHandler(self._handle_new_file)
                self.observer = Observer()
                self.observer.schedule(event_handler, str(self.watch_folder), recursive=False)
                self.observer.start()
                
                # 处理现有文件
                await self._process_existing_files()
                
                self._initialized = True
                self.logger.info(f"Manual ingestion source initialized, watching: {self.watch_folder}")
                
            except Exception as e:
                raise SourceError(f"Failed to initialize Manual ingestion source: {e}")
        
        async def _handle_new_file(self, file_path: str):
            """处理新文件"""
            try:
                paper_metadata = await self._extract_metadata_from_pdf(file_path)
                if paper_metadata:
                    async with self._papers_lock:
                        self.discovered_papers.append(paper_metadata)
                    self.logger.info(f"Ingested paper: {paper_metadata.title}")
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
        
        async def _process_existing_files(self):
            """处理现有的PDF文件"""
            pdf_files = list(self.watch_folder.glob("*.pdf"))
            
            for pdf_file in pdf_files:
                await self._handle_new_file(str(pdf_file))
        
        async def _extract_metadata_from_pdf(self, file_path: str) -> Optional[PaperMetadata]:
            """从PDF文件提取元数据"""
            try:
                file_path_obj = Path(file_path)
                
                # 基本信息
                title = file_path_obj.stem.replace('_', ' ').replace('-', ' ')
                
                # 尝试从文件名解析更多信息
                # 格式示例: "Author_Year_Title.pdf"
                parts = file_path_obj.stem.split('_')
                authors = []
                
                if len(parts) >= 3:
                    # 假设第一部分是作者
                    author_part = parts[0]
                    if author_part and not author_part.isdigit():
                        authors.append(Author(name=author_part.replace('-', ' ')))
                    
                    # 重新构建标题（跳过可能的年份）
                    title_parts = []
                    for part in parts[1:]:
                        if not part.isdigit() or len(part) != 4:  # 跳过年份
                            title_parts.append(part)
                    
                    if title_parts:
                        title = ' '.join(title_parts)
                
                # 获取文件修改时间作为发现时间
                stat = file_path_obj.stat()
                discovery_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                
                paper = PaperMetadata(
                    title=title,
                    authors=authors,
                    source_type=SourceType.MANUAL,
                    source_url=f"file://{file_path}",
                    pdf_url=f"file://{file_path}",
                    publication_date=discovery_time,
                    raw_data={
                        "file_path": file_path,
                        "file_size": stat.st_size,
                        "modified_time": discovery_time.isoformat()
                    }
                )
                
                return paper
                
            except Exception as e:
                self.logger.error(f"Error extracting metadata from {file_path}: {e}")
                return None
        
        async def discover_papers(self, query: Optional[str] = None, limit: int = 50) -> List[PaperMetadata]:
            """获取发现的论文"""
            if not self._initialized:
                await self.initialize()
            
            async with self._papers_lock:
                # 返回最近发现的论文
                papers = self.discovered_papers[-limit:]
                # 清空列表以避免重复返回
                self.discovered_papers = []
                
            self.logger.info(f"Retrieved {len(papers)} manually ingested papers")
            return papers
        
        def get_source_type(self) -> SourceType:
            return SourceType.MANUAL
        
        async def close(self) -> None:
            """关闭手动摄入源"""
            if self.observer:
                self.observer.stop()
                # 在事件循环中避免阻塞，使用后台线程等待停止完成
                try:
                    await asyncio.to_thread(self.observer.join)
                except Exception:
                    pass
            
            self._initialized = False
            self.logger.info("Manual ingestion source closed")
else:
    class ManualIngestionSource(BaseDiscoverySource):
        """手动摄入源占位实现（watchdog 未安装时禁用该源，避免导入期错误）。"""
        def __init__(self, config: ConfigManager):
            super().__init__(config)
            raise SourceError("watchdog library not available. Please install: pip install watchdog")


class TasteEngine:
    """
    学术品味引擎
    
    基于SOTA嵌入模型实现持久化的学术品味学习，
    支持冷启动、探索与利用平衡、反馈学习机制。
    """
    
    def __init__(self, config: ConfigManager, embedding_service: EmbeddingProviderService, memory_system: HybridMemorySystem):
        """
        无副作用构造函数
        
        Args:
            config: 配置管理器
            embedding_service: 嵌入服务
            memory_system: 记忆系统
        """
        self.config = config
        self.embedding_service = embedding_service
        self.memory_system = memory_system
        self.logger = get_logger("taste_engine")
        
        self.taste_vector: Optional[TasteVector] = None
        self.taste_file = Path(config.paths.logs_folder) / "taste_vector.json" if config.paths else Path("./logs/taste_vector.json")
        self._initialized = False
        
        # 配置参数
        self.epsilon = config.preferences.exploration_epsilon if config.preferences else 0.1
        self.positive_weight = config.feedback.positive_weight_increase if config.feedback else 0.2
        self.negative_weight = config.feedback.negative_weight_decrease if config.feedback else 0.05
    
    async def initialize(self) -> None:
        """异步初始化品味引擎"""
        if self._initialized:
            return
        
        try:
            # 尝试加载现有的品味向量
            if await self._load_taste_vector():
                self.logger.info("Loaded existing taste vector")
            else:
                # 冷启动：基于种子论文和关键词初始化
                await self._cold_start_initialization()
                self.logger.info("Initialized taste vector with cold start")
            
            self._initialized = True
            
        except Exception as e:
            raise TasteEngineError(f"Failed to initialize taste engine: {e}")
    
    async def _load_taste_vector(self) -> bool:
        """加载现有的品味向量"""
        try:
            if not self.taste_file.exists():
                return False
            
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(self.taste_file, 'r') as f:
                    data = json.loads(await f.read())
            else:
                # 同步回退
                with open(self.taste_file, 'r') as f:
                    data = json.loads(f.read())
            
            # 重建TasteVector对象
            self.taste_vector = TasteVector(
                interests=data['interests'],
                weights=data.get('weights', {}),
                keywords=data.get('keywords', []),
                exploration_factor=data.get('exploration_factor', self.epsilon),
                last_updated=datetime.fromisoformat(data.get('last_updated', datetime.now(timezone.utc).isoformat())),
                positive_examples=data.get('positive_examples', []),
                negative_examples=data.get('negative_examples', [])
            )
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load taste vector: {e}")
            return False
    
    async def _save_taste_vector(self) -> None:
        """保存品味向量"""
        if not self.taste_vector:
            return
        
        try:
            self.taste_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'interests': self.taste_vector.interests,
                'weights': self.taste_vector.weights,
                'keywords': self.taste_vector.keywords,
                'exploration_factor': self.taste_vector.exploration_factor,
                'last_updated': self.taste_vector.last_updated.isoformat(),
                'positive_examples': self.taste_vector.positive_examples,
                'negative_examples': self.taste_vector.negative_examples
            }
            
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(self.taste_file, 'w') as f:
                    await f.write(json.dumps(data, indent=2))
            else:
                # 同步回退
                with open(self.taste_file, 'w') as f:
                    f.write(json.dumps(data, indent=2))
                
        except Exception as e:
            self.logger.error(f"Failed to save taste vector: {e}")

    async def save(self) -> None:
        """公共保存入口：持久化当前 TasteVector。"""
        await self._save_taste_vector()
    
    async def _cold_start_initialization(self) -> None:
        """冷启动初始化"""
        try:
            # 获取初始关键词和种子论文
            initial_keywords = self.config.preferences.initial_keywords if self.config.preferences else []
            seed_papers = self.config.preferences.seed_papers if self.config.preferences else []
            
            # 生成初始兴趣嵌入
            if initial_keywords:
                # 合并关键词生成种子嵌入
                seed_text = " ".join(initial_keywords)
                interests_embedding = await self.embedding_service.embed_text(seed_text)
            else:
                # 使用默认的AI关键词
                default_keywords = ["artificial intelligence", "machine learning", "deep learning"]
                seed_text = " ".join(default_keywords)
                interests_embedding = await self.embedding_service.embed_text(seed_text)
                initial_keywords = default_keywords
            
            # 初始化品味向量
            self.taste_vector = TasteVector(
                interests=interests_embedding,
                keywords=initial_keywords,
                exploration_factor=self.epsilon,
                weights={kw: 1.0 for kw in initial_keywords}
            )
            
            # 如果有种子论文，处理它们
            for seed_url in seed_papers[:5]:  # 限制种子论文数量
                try:
                    await self._process_seed_paper(seed_url)
                except Exception as e:
                    self.logger.warning(f"Failed to process seed paper {seed_url}: {e}")
            
            # 保存初始化的品味向量
            await self._save_taste_vector()
            
        except Exception as e:
            # 如果冷启动失败，创建最小的品味向量（维度获取失败时提供安全回退）
            try:
                embedding_dim = self.embedding_service.get_dimensions()
            except Exception:
                model_name = getattr(getattr(self.config, "embedding_settings", None), "model", "")
                name = (model_name or "").lower()
                # 常见默认维度：all-MiniLM-L6-v2 → 384；若未知则退回 384
                if "all-minilm-l6-v2" in name:
                    embedding_dim = 384
                elif "all-minilm-l12-v2" in name:
                    embedding_dim = 384  # 常见实现也为 384
                else:
                    embedding_dim = 384
            self.taste_vector = TasteVector(
                interests=[0.0] * embedding_dim,
                keywords=["artificial intelligence"],
                exploration_factor=self.epsilon
            )
            await self._save_taste_vector()
            # 不中断初始化：记录告警并回退为零向量
            self.logger.warning(f"Cold start initialization failed, fallback to zero vector (dim={embedding_dim}): {e}")
            return
    
    async def _process_seed_paper(self, paper_url: str) -> None:
        """处理种子论文：从 arXiv 等来源解析元数据并更新品味向量（真实实现）。"""
        try:
            # 解析 arXiv ID（支持 https://arxiv.org/abs/<id> 或 /pdf/<id>.pdf）
            arxiv_id = None
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:\.pdf)?", paper_url)
            if m:
                arxiv_id = m.group(1)

            title = None
            abstract = None
            authors: List[str] = []

            if ARXIV_AVAILABLE and arxiv_id:
                try:
                    client = arxiv.Client(page_size=1, delay_seconds=3, num_retries=2)
                    search = arxiv.Search(id_list=[arxiv_id])
                    loop = asyncio.get_event_loop()
                    results: List[arxiv.Result] = await loop.run_in_executor(
                        None, lambda: list(client.results(search))
                    )
                    if results:
                        r = results[0]
                        title = r.title
                        abstract = r.summary
                        authors = [a.name for a in r.authors]
                except Exception as e:
                    self.logger.warning(f"arXiv lookup failed for {paper_url}: {e}")

            # 如果不是 arXiv 链接，尽力从 URL 文本提取关键信息（标题退化为文件名/片段）
            if not title:
                title = paper_url.split('/')[-1].replace('-', ' ').replace('_', ' ')[:120]

            # 组合用于嵌入的文本
            seed_text_parts = [title or ""]
            if abstract:
                seed_text_parts.append(abstract)
            seed_text = "\n".join([p for p in seed_text_parts if p])

            if not seed_text.strip():
                return

            # 生成嵌入并更新兴趣向量
            seed_embedding = await self.embedding_service.embed_text(seed_text)

            import numpy as np
            if self.taste_vector and seed_embedding:
                current = np.array(self.taste_vector.interests)
                new = np.array(seed_embedding)
                # 以较小学习率向正例方向靠拢
                lr = 0.1
                updated = current + lr * (new - current)
                norm = np.linalg.norm(updated)
                if norm > 0:
                    self.taste_vector.interests = (updated / norm).tolist()

            # 增强作者偏好权重
            for author_name in authors[:8]:
                key = f"author:{author_name}"
                current_weight = self.taste_vector.weights.get(key, 1.0)
                self.taste_vector.weights[key] = min(current_weight + self.positive_weight, 3.0)

            # 记录为正例
            if self.taste_vector and self.taste_vector.positive_examples is not None:
                self.taste_vector.positive_examples.append(f"seed:{paper_url}")

        except Exception as e:
            self.logger.warning(f"Failed to process seed paper {paper_url}: {e}")
    
    async def score_paper(self, paper: DiscoveredPaper) -> float:
        """
        为论文评分
        
        Args:
            paper: 发现的论文
            
        Returns:
            相关性评分 (0-1之间)
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.taste_vector:
            return 0.5  # 默认中性评分
        
        try:
            # 生成论文嵌入
            if paper.embedding is None:
                paper_text = self._extract_paper_text(paper)
                paper.embedding = await self.embedding_service.embed_text(paper_text)
            
            # 计算基础相似度
            base_similarity = await self._calculate_similarity(paper.embedding, self.taste_vector.interests)
            
            # 应用关键词权重
            keyword_boost = self._calculate_keyword_boost(paper)
            
            # 应用作者权重（如果有历史偏好）
            author_boost = self._calculate_author_boost(paper)
            
            # 应用时间衰减（更新的论文略有优势）
            time_boost = self._calculate_time_boost(paper)
            
            # 综合评分
            final_score = (base_similarity * 0.6 + 
                          keyword_boost * 0.2 + 
                          author_boost * 0.1 + 
                          time_boost * 0.1)
            
            # 应用ε-greedy策略
            if random.random() < self.taste_vector.exploration_factor:
                # 探索模式：增加随机性
                exploration_noise = random.uniform(-0.2, 0.2)
                final_score = max(0.0, min(1.0, final_score + exploration_noise))
            
            paper.relevance_score = final_score
            
            self.logger.debug(f"Scored paper '{paper.metadata.title}': {final_score:.3f}")
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error scoring paper: {e}")
            return 0.5
    
    def _extract_paper_text(self, paper: DiscoveredPaper) -> str:
        """提取论文的文本用于嵌入"""
        text_parts = []
        
        if paper.metadata.title:
            text_parts.append(paper.metadata.title)
        
        if paper.metadata.abstract:
            text_parts.append(paper.metadata.abstract)
        
        if paper.metadata.keywords:
            text_parts.append(" ".join(paper.metadata.keywords))
        
        # 添加作者信息
        if paper.metadata.authors:
            author_names = [author.name for author in paper.metadata.authors]
            text_parts.append(" ".join(author_names))
        
        return " ".join(text_parts)
    
    async def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """计算两个嵌入向量的相似度"""
        try:
            import numpy as np
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 余弦相似度
            cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            # 映射到0-1范围
            return (cosine_sim + 1) / 2
            
        except Exception:
            return 0.5
    
    def _calculate_keyword_boost(self, paper: DiscoveredPaper) -> float:
        """计算关键词加成"""
        if not self.taste_vector or not self.taste_vector.keywords:
            return 0.0
        
        paper_text = self._extract_paper_text(paper).lower()
        
        boost = 0.0
        for keyword in self.taste_vector.keywords:
            if keyword.lower() in paper_text:
                weight = self.taste_vector.weights.get(keyword, 1.0)
                boost += weight * 0.1  # 每个匹配关键词的基础加成
        
        return min(boost, 1.0)
    
    def _calculate_author_boost(self, paper: DiscoveredPaper) -> float:
        """计算作者加成：基于已学习的作者权重进行加成。"""
        if not self.taste_vector or not paper.metadata.authors:
            return 0.5
        boost = 0.5
        for author in paper.metadata.authors:
            key = f"author:{author.name}"
            if key in self.taste_vector.weights:
                boost += 0.05 * self.taste_vector.weights.get(key, 1.0)
        return max(0.3, min(0.9, boost))
    
    def _calculate_time_boost(self, paper: DiscoveredPaper) -> float:
        """计算时间加成"""
        if not paper.metadata.publication_date:
            return 0.5
        
        # 计算论文发布时间与当前时间的差距
        now = datetime.now(timezone.utc)
        age_days = (now - paper.metadata.publication_date).days
        
        # 新论文有轻微优势
        if age_days <= 7:
            return 0.8
        elif age_days <= 30:
            return 0.6
        elif age_days <= 90:
            return 0.5
        else:
            return 0.4
    
    async def learn_from_feedback(self, paper: DiscoveredPaper, feedback: bool) -> None:
        """
        从反馈中学习
        
        Args:
            paper: 论文
            feedback: True为正反馈，False为负反馈
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.taste_vector:
            return
        
        try:
            # 记录反馈
            paper.taste_feedback = feedback
            
            if feedback:
                # 正反馈
                if paper.id not in self.taste_vector.positive_examples:
                    self.taste_vector.positive_examples.append(paper.id)
                
                # 增强相关特征的权重
                await self._enhance_positive_features(paper)
                
            else:
                # 负反馈
                if paper.id not in self.taste_vector.negative_examples:
                    self.taste_vector.negative_examples.append(paper.id)
                
                # 降低相关特征的权重
                await self._diminish_negative_features(paper)
            
            # 更新品味向量
            await self._update_taste_vector(paper, feedback)
            
            # 保存更新的品味向量
            self.taste_vector.last_updated = datetime.now(timezone.utc)
            await self._save_taste_vector()
            
            self.logger.info(f"Learned from {'positive' if feedback else 'negative'} feedback for paper: {paper.metadata.title}")
            
        except Exception as e:
            self.logger.error(f"Error learning from feedback: {e}")

    def _iter_author_names(self, paper: DiscoveredPaper) -> List[str]:
        try:
            return [a.name for a in (paper.metadata.authors or []) if a and a.name]
        except Exception:
            return []
    
    async def _enhance_positive_features(self, paper: DiscoveredPaper) -> None:
        """增强正例特征"""
        # 增强关键词权重
        paper_text = self._extract_paper_text(paper).lower()
        
        for keyword in self.taste_vector.keywords:
            if keyword.lower() in paper_text:
                current_weight = self.taste_vector.weights.get(keyword, 1.0)
                self.taste_vector.weights[keyword] = min(current_weight + self.positive_weight, 3.0)
        
        # 添加新的关键词（从论文中提取）
        if paper.metadata.keywords:
            for kw in paper.metadata.keywords[:3]:  # 限制新增关键词数量
                if kw not in self.taste_vector.keywords:
                    self.taste_vector.keywords.append(kw)
                    self.taste_vector.weights[kw] = 1.0
        # 增强作者权重
        for author_name in self._iter_author_names(paper):
            key = f"author:{author_name}"
            current_weight = self.taste_vector.weights.get(key, 1.0)
            self.taste_vector.weights[key] = min(current_weight + self.positive_weight, 3.0)
    
    async def _diminish_negative_features(self, paper: DiscoveredPaper) -> None:
        """降低负例特征"""
        paper_text = self._extract_paper_text(paper).lower()
        
        for keyword in self.taste_vector.keywords:
            if keyword.lower() in paper_text:
                current_weight = self.taste_vector.weights.get(keyword, 1.0)
                self.taste_vector.weights[keyword] = max(current_weight - self.negative_weight, 0.1)
        # 降低作者权重
        for author_name in self._iter_author_names(paper):
            key = f"author:{author_name}"
            current_weight = self.taste_vector.weights.get(key, 1.0)
            self.taste_vector.weights[key] = max(current_weight - self.negative_weight, 0.1)
    
    async def _update_taste_vector(self, paper: DiscoveredPaper, feedback: bool) -> None:
        """更新品味向量"""
        if not paper.embedding:
            return
        
        try:
            import numpy as np
            
            current_interests = np.array(self.taste_vector.interests)
            paper_embedding = np.array(paper.embedding)
            
            if feedback:
                # 正反馈：向论文嵌入方向调整
                learning_rate = 0.1
                updated_interests = current_interests + learning_rate * (paper_embedding - current_interests)
            else:
                # 负反馈：远离论文嵌入方向
                learning_rate = 0.05
                updated_interests = current_interests - learning_rate * (paper_embedding - current_interests)
            
            # 归一化
            norm = np.linalg.norm(updated_interests)
            if norm > 0:
                self.taste_vector.interests = (updated_interests / norm).tolist()
            
        except Exception as e:
            self.logger.warning(f"Failed to update taste vector: {e}")
    
    async def get_taste_summary(self) -> Dict[str, Any]:
        """获取品味总结"""
        if not self._initialized:
            await self.initialize()
        
        if not self.taste_vector:
            return {"error": "Taste vector not initialized"}
        
        return {
            "keywords": self.taste_vector.keywords,
            "keyword_weights": self.taste_vector.weights,
            "exploration_factor": self.taste_vector.exploration_factor,
            "last_updated": self.taste_vector.last_updated.isoformat(),
            "positive_examples_count": len(self.taste_vector.positive_examples),
            "negative_examples_count": len(self.taste_vector.negative_examples),
            "embedding_dimension": len(self.taste_vector.interests)
        }
    
    async def close(self) -> None:
        """关闭品味引擎"""
        if self.taste_vector:
            await self._save_taste_vector()
        
        self._initialized = False
        self.logger.info("Taste engine closed")


class DiscoveryEngine:
    """
    学术论文发现与品鉴引擎
    
    整合多种信息源和智能品味引擎，提供全面的学术论文发现能力。
    """
    
    def __init__(self, config: ConfigManager, embedding_service: EmbeddingProviderService, 
                 llm_service: LLMProviderService, memory_system: HybridMemorySystem):
        """
        无副作用构造函数
        
        Args:
            config: 配置管理器
            embedding_service: 嵌入服务
            llm_service: LLM服务
            memory_system: 记忆系统
        """
        if not config.is_initialized():
            raise ValueError("ConfigManager must be initialized")
        
        self.config = config
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.memory_system = memory_system
        self.logger = get_logger("discovery_engine")
        self.learning_loop: Optional[LearningLoop] = None
        
        # 信息源
        self.sources: Dict[SourceType, BaseDiscoverySource] = {}
        
        # 品味引擎
        self.taste_engine: Optional[TasteEngine] = None
        
        # 系统状态
        self._initialized = False
        self._discovery_stats = {
            "total_discovered": 0,
            "source_counts": defaultdict(int),
            "quality_distribution": defaultdict(int),
            "last_discovery": None
        }
    
    async def initialize(self) -> None:
        """异步初始化发现引擎"""
        if self._initialized:
            return
        
        try:
            self.logger.info("Initializing discovery engine...")
            
            # 初始化信息源
            await self._initialize_sources()
            
            # 初始化品味引擎
            self.taste_engine = TasteEngine(self.config, self.embedding_service, self.memory_system)
            await self.taste_engine.initialize()

            # 将 TasteEngine 注入学习循环（仅在外部已注入的情况下补充 taste_engine 引用，避免双实例）
            try:
                if self.learning_loop:
                    self.learning_loop.set_taste_engine(self.taste_engine)
            except Exception as e:
                self.logger.warning(f"LearningLoop attach failed (non-fatal): {e}")
            
            self._initialized = True
            self.logger.info(f"Discovery engine initialized with {len(self.sources)} sources")
            
        except Exception as e:
            raise DiscoveryError(f"Failed to initialize discovery engine: {e}")
    
    async def _initialize_sources(self) -> None:
        """初始化所有信息源"""
        source_configs = [
            (SourceType.ARXIV, ArxivSource),
            (SourceType.CONFERENCE, ConferenceSource),
            (SourceType.SOCIAL, SocialSource),
            (SourceType.MANUAL, ManualIngestionSource)
        ]
        
        initialization_tasks = []
        task_source_pairs: List[Tuple[SourceType, BaseDiscoverySource]] = []
        
        for source_type, source_class in source_configs:
            try:
                source = source_class(self.config)
                self.sources[source_type] = source
                initialization_tasks.append(source.initialize())
                task_source_pairs.append((source_type, source))
            except Exception as e:
                self.logger.warning(f"Failed to create {source_type.value} source: {e}")
        
        # 并行初始化所有源
        if initialization_tasks:
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)

            # 检查初始化结果（严格按任务-源配对判断）
            for (source_type, _), result in zip(task_source_pairs, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to initialize {source_type.value}: {result}")
                    if source_type in self.sources:
                        del self.sources[source_type]
        
        if not self.sources:
            raise DiscoveryError("No discovery sources available")
    
    async def discover_papers(
        self,
        query: Optional[str] = None,
        source_types: Optional[List[SourceType]] = None,
        limit: int = 100,
        use_taste_filtering: bool = True,
        min_quality_score: float = 0.5
    ) -> List[DiscoveredPaper]:
        """
        发现学术论文
        
        Args:
            query: 搜索查询
            source_types: 使用的信息源类型
            limit: 结果限制
            use_taste_filtering: 是否使用品味过滤
            min_quality_score: 最小质量分数
            
        Returns:
            发现的论文列表
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        source_types = source_types or list(self.sources.keys())
        
        try:
            # 并行从各个源发现论文
            discovery_tasks = []
            
            for source_type in source_types:
                if source_type in self.sources:
                    source = self.sources[source_type]
                    task = source.discover_papers(query, limit // len(source_types) + 10)
                    discovery_tasks.append((source_type, task))
            
            # 执行并行发现
            results = await asyncio.gather(*[task for _, task in discovery_tasks], return_exceptions=True)
            
            # 收集所有论文
            all_papers = []
            for i, (source_type, _) in enumerate(discovery_tasks):
                result = results[i]
                if isinstance(result, list):
                    for paper_metadata in result:
                        paper = DiscoveredPaper(
                            id=str(uuid.uuid4()),
                            metadata=paper_metadata
                        )
                        all_papers.append(paper)
                        self._discovery_stats["source_counts"][source_type] += 1
                elif isinstance(result, Exception):
                    self.logger.warning(f"Discovery failed for {source_type.value}: {result}")
            
            # 去重
            unique_papers = self._deduplicate_papers(all_papers)
            
            # 使用品味引擎评分和过滤
            if use_taste_filtering and self.taste_engine:
                scored_papers = []
                for paper in unique_papers:
                    score = await self.taste_engine.score_paper(paper)
                    if score >= min_quality_score:
                        scored_papers.append(paper)
                
                # 按相关性排序
                scored_papers.sort(key=lambda p: p.relevance_score or 0, reverse=True)
                final_papers = scored_papers[:limit]
            else:
                final_papers = unique_papers[:limit]
            
            # 存储到记忆系统
            await self._store_papers_to_memory(final_papers)

            # 建立 id -> paper 映射，便于反馈学习
            if not hasattr(self, "_last_discovered_index"):
                self._last_discovered_index = {}
            for p in final_papers:
                self._last_discovered_index[p.id] = p
            
            # 更新统计
            self._discovery_stats["total_discovered"] += len(final_papers)
            self._discovery_stats["last_discovery"] = datetime.now(timezone.utc).isoformat()
            
            duration = time.time() - start_time
            log_performance("paper_discovery", duration, 
                           papers_count=len(final_papers), query=query)
            
            self.logger.info(f"Discovered {len(final_papers)} papers in {duration:.2f}s")
            return final_papers
            
        except Exception as e:
            self.logger.error(f"Paper discovery failed: {e}")
            raise DiscoveryError(f"Discovery error: {e}")
    
    def _deduplicate_papers(self, papers: List[DiscoveredPaper]) -> List[DiscoveredPaper]:
        """去重论文"""
        seen_titles = set()
        seen_arxiv_ids = set()
        seen_dois = set()
        unique_papers = []
        
        for paper in papers:
            # 标题去重
            title_key = ""
            if paper.metadata.title:
                # 标准化标题：小写、压缩空白、去除常见标点，提升去重鲁棒性
                try:
                    t = paper.metadata.title
                    t = re.sub(r"\s+", " ", t.lower()).strip()
                    # 仅保留字母数字与下划线以及空格，避免不同标点导致的重复
                    t = re.sub(r"[^\w\s]", "", t)
                    title_key = t
                except Exception:
                    title_key = paper.metadata.title.lower().strip()
            
            # ArXiv ID去重
            arxiv_key = paper.metadata.arxiv_id if paper.metadata.arxiv_id else ""
            
            # DOI去重
            doi_key = paper.metadata.doi if paper.metadata.doi else ""
            
            is_duplicate = False
            
            if title_key and title_key in seen_titles:
                is_duplicate = True
            elif arxiv_key and arxiv_key in seen_arxiv_ids:
                is_duplicate = True
            elif doi_key and doi_key in seen_dois:
                is_duplicate = True
            
            if not is_duplicate:
                if title_key:
                    seen_titles.add(title_key)
                if arxiv_key:
                    seen_arxiv_ids.add(arxiv_key)
                if doi_key:
                    seen_dois.add(doi_key)
                
                unique_papers.append(paper)
        
        self.logger.debug(f"Deduplicated {len(papers)} papers to {len(unique_papers)} unique papers")
        return unique_papers
    
    async def _store_papers_to_memory(self, papers: List[DiscoveredPaper]) -> None:
        """将论文存储到记忆系统"""
        try:
            for paper in papers:
                # 存储为情节记忆
                content = f"Title: {paper.metadata.title}\n"
                if paper.metadata.abstract:
                    content += f"Abstract: {paper.metadata.abstract}\n"
                
                if paper.metadata.authors:
                    author_names = [author.name for author in paper.metadata.authors]
                    content += f"Authors: {', '.join(author_names)}\n"
                
                metadata = {
                    "discovery_timestamp": (paper.discovery_timestamp.isoformat() if getattr(paper, "discovery_timestamp", None) else ""),
                    "source_type": (paper.metadata.source_type.value if getattr(paper.metadata, "source_type", None) else "unknown"),
                    "relevance_score": float(paper.relevance_score or 0.0),
                    "arxiv_id": str(getattr(paper.metadata, "arxiv_id", "") or ""),
                    "doi": str(getattr(paper.metadata, "doi", "") or ""),
                    "venue": str(getattr(paper.metadata, "venue", "") or ""),
                }
                
                await self.memory_system.store_episodic_memory(
                    content=content,
                    source="discovery_engine",
                    context="discovered_paper",
                    tags=["paper", "discovery"] + paper.tags,
                    metadata=metadata,
                    importance_score=paper.relevance_score or 0.5
                )
                
                # 为论文创建概念节点
                paper_node_id = await self.memory_system.store_conceptual_memory(
                    name=paper.metadata.title or "Unknown Paper",
                    node_type=NodeType.PAPER,
                    attributes={
                        "arxiv_id": str(getattr(paper.metadata, "arxiv_id", "") or ""),
                        "doi": str(getattr(paper.metadata, "doi", "") or ""),
                        "venue": str(getattr(paper.metadata, "venue", "") or ""),
                        "source_type": (paper.metadata.source_type.value if getattr(paper.metadata, "source_type", None) else "unknown"),
                    },
                    importance_score=paper.relevance_score or 0.5
                )
                
                # 为作者创建节点和关系
                for author in paper.metadata.authors:
                    author_node_id = await self.memory_system.store_conceptual_memory(
                        name=author.name,
                        node_type=NodeType.AUTHOR,
                        attributes={
                            "affiliation": author.affiliation,
                            "email": author.email,
                            "orcid": author.orcid
                        }
                    )
                    
                    # 添加作者-论文关系
                    await self.memory_system.add_concept_relationship(
                        author_node_id, paper_node_id, EdgeType.AUTHORED
                    )
                
        except Exception as e:
            self.logger.warning(f"Error storing papers to memory: {e}")
    
    async def provide_feedback(self, paper_id: str, feedback: bool) -> bool:
        """
        提供论文反馈
        
        Args:
            paper_id: 论文ID
            feedback: 反馈（True为正面，False为负面）
            
        Returns:
            是否成功
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            index: Dict[str, DiscoveredPaper] = getattr(self, "_last_discovered_index", {})
            paper = index.get(paper_id)
            if not paper:
                self.logger.warning(f"Feedback paper not found in index: {paper_id}")
                return False
            if not self.taste_engine:
                self.logger.warning("Taste engine not initialized")
                return False

            await self.taste_engine.learn_from_feedback(paper, feedback)
            # 学习循环记录人类反馈（非阻塞）
            try:
                if self.learning_loop:
                    await self.learning_loop.record_human_feedback(paper, feedback)
            except Exception as _e:
                self.logger.warning(f"LearningLoop human feedback failed: {_e}")
            self.logger.info(
                f"Applied {'positive' if feedback else 'negative'} feedback for paper: {paper_id}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error providing feedback: {e}")
            return False
    
    async def get_discovery_insights(self) -> Dict[str, Any]:
        """获取发现引擎洞察"""
        if not self._initialized:
            await self.initialize()
        
        insights = {
            "statistics": self._discovery_stats.copy(),
            "sources": {
                source_type.value: {
                    "available": True,
                    "type": source.get_source_type().value
                }
                for source_type, source in self.sources.items()
            }
        }
        
        # 添加品味引擎洞察
        if self.taste_engine:
            taste_summary = await self.taste_engine.get_taste_summary()
            insights["taste_engine"] = taste_summary
        
        # 添加记忆系统统计
        if self.memory_system:
            memory_stats = self.memory_system.get_memory_stats()
            insights["memory_system"] = memory_stats
        
        return insights
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    async def close(self) -> None:
        """关闭发现引擎"""
        # 关闭所有信息源
        close_tasks = []
        for source in self.sources.values():
            close_tasks.append(source.close())
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # 关闭品味引擎
        if self.taste_engine:
            await self.taste_engine.close()
        
        self._initialized = False
        self.logger.info("Discovery engine closed")

    async def rank_papers(self, papers: List[DiscoveredPaper]) -> List[DiscoveredPaper]:
        """使用 TasteEngine 对论文进行打分排序。"""
        if not papers:
            return []
        if not self._initialized:
            await self.initialize()
        scored: List[DiscoveredPaper] = []
        for p in papers:
            try:
                score = await self.taste_engine.score_paper(p) if self.taste_engine else 0.5
            except Exception:
                score = 0.5
            p.relevance_score = score
            scored.append(p)
        scored.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        return scored

    async def update_taste_from_paper(self, metadata: PaperMetadata, positive: bool) -> None:
        """根据处理完成的论文元数据更新品味引擎。"""
        try:
            index: Dict[str, DiscoveredPaper] = getattr(self, "_last_discovered_index", {})
            candidate = None
            for p in index.values():
                if metadata.arxiv_id and p.metadata.arxiv_id == metadata.arxiv_id:
                    candidate = p
                    break
                if metadata.html_url and p.metadata.html_url == metadata.html_url:
                    candidate = p
                    break
                if metadata.title and p.metadata.title == metadata.title:
                    candidate = p
                    break
            if candidate and self.taste_engine:
                await self.taste_engine.learn_from_feedback(candidate, positive)
                try:
                    if self.learning_loop:
                        await self.learning_loop.record_human_feedback(candidate, positive)
                except Exception as _e:
                    self.logger.warning(f"LearningLoop update_taste_from_paper failed: {_e}")
        except Exception as e:
            self.logger.warning(f"update_taste_from_paper failed: {e}")
