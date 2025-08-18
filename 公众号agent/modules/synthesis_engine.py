"""
内容合成引擎 - 将解读内容合成Deep Scholar AI风格文章

核心功能：
1. 内容聚合：按原文顺序组合分析对，集成全局分析和长期记忆
2. 结构化生成：生成标题、引言、核心观点、深度思考
3. 微信公众号适配：专业排版，图片集成，术语处理
4. 智能交付：异步保存，文件名规范，关联图片管理

严格遵循依赖注入原则，使用外部Jinja2模板，绝不使用简单匹配算法。
"""

import asyncio
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import aiofiles
import jinja2
import numpy as np
import shutil

from .config_manager import ConfigManager
from .llm_provider import LLMProviderService, LLMRequest, LLMResponse
from .memory_system import HybridMemorySystem, MemoryType, MemorySearchResult
from .logger import get_logger, log_synthesis_operation, log_performance


@dataclass
class AnalysisPair:
    """分析对数据结构"""
    original: str
    analysis: str
    section_id: str
    confidence_score: float = 0.0
    

@dataclass
class GlobalAnalysisMap:
    """全局分析映射"""
    summary: str
    key_insights: List[str]
    technical_terms: Dict[str, str]
    methodology_analysis: str
    significance_assessment: str


@dataclass
class PaperMetadata:
    """论文元数据"""
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    institutions: Optional[List[str]] = None
    publication_date: Optional[str] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None


@dataclass
class FigureReference:
    """图片引用信息"""
    figure_id: str
    caption: str
    file_path: str
    context: str
    relevance_score: float = 0.0


@dataclass
class ArticleSection:
    """文章段落结构"""
    section_type: str  # title, introduction, insights, thinking, content
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesizedArticle:
    """合成文章结果"""
    title: str
    content: str
    metadata: PaperMetadata
    figures: List[FigureReference]
    word_count: int
    generation_timestamp: datetime
    article_id: str
    file_path: Optional[str] = None


class WeChatMarkdownFormatter:
    """微信公众号Markdown格式化器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def format_content(self, content: str, figures: List[FigureReference]) -> str:
        """
        格式化内容为微信公众号适配的Markdown
        
        Args:
            content: 原始内容
            figures: 图片引用列表
            
        Returns:
            格式化后的内容
        """
        try:
            # 处理专业术语加粗
            content = self._format_technical_terms(content)
            
            # 处理引用块
            content = self._format_quotes(content)
            
            # 集成图片
            content = self._integrate_figures(content, figures)
            
            # 优化段落结构
            content = self._optimize_paragraphs(content)
            
            # 移除多余的格式标记
            content = self._clean_formatting(content)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Content formatting failed: {e}")
            return content
    
    def _format_technical_terms(self, content: str) -> str:
        """格式化专业术语：避免代码块/行内代码/URL命中，安全加粗。

        步骤：
        1) 暂存并替换所有代码块 ```...``` 与行内代码 `...` 为占位符
        2) 暂存并替换 URL（http/https/ftp）为占位符
        2.5) 暂存并替换已存在的加粗段 **...** 与 LaTeX 数学（\(\) / \[\]）为占位符，避免误伤
        3) 仅对正文区域进行术语正则加粗
        4) 还原占位符
        """
        if not content:
            return content

        original = content

        # 1) 三反引号代码块
        codeblock_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        codeblocks = []
        import uuid as _uuid
        def _cb_repl(m):
            codeblocks.append(m.group(0))
            return f"{'{'}CB_{_uuid.uuid4().hex}{'}'}"
        content = codeblock_pattern.sub(_cb_repl, content)

        # 2) 行内代码 `...`
        inline_code_pattern = re.compile(r"`[^`\n]+`")
        inline_codes = []
        def _ic_repl(m):
            inline_codes.append(m.group(0))
            return f"{'{'}IC_{_uuid.uuid4().hex}{'}'}"
        content = inline_code_pattern.sub(_ic_repl, content)

        # 3) URL
        url_pattern = re.compile(r"\b(?:https?|ftp)://[\w\-\.\?\,\:/#%&=~+@;!\*\(\)]+", re.IGNORECASE)
        urls = []
        def _url_repl(m):
            urls.append(m.group(0))
            return f"{'{'}URL_{_uuid.uuid4().hex}{'}'}"
        content = url_pattern.sub(_url_repl, content)

        # 2.5) 已存在的加粗段 **...** 与 LaTeX 数学
        # 目的：避免对已加粗区域进行再次加粗，导致 ****词**** 的问题
        # 同时保护 LaTeX 数学，避免粗体替换破坏公式
        bold_pattern = re.compile(r"\*\*[\s\S]*?\*\*")
        bold_segments: List[str] = []
        def _bold_repl(m):
            bold_segments.append(m.group(0))
            return f"{'{'}BOLD_{_uuid.uuid4().hex}{'}'}"
        content = bold_pattern.sub(_bold_repl, content)

        # LaTeX 数学（行间与行内）
        math_display_pattern = re.compile(r"\\\[[\s\S]*?\\\]", re.MULTILINE)
        math_inline_pattern = re.compile(r"\\\([^\)\n]*?\\\)")
        math_segments: List[str] = []
        def _math_disp_repl(m):
            math_segments.append(m.group(0))
            return f"{'{'}MATHD_{_uuid.uuid4().hex}{'}'}"
        def _math_inl_repl(m):
            math_segments.append(m.group(0))
            return f"{'{'}MATHI_{_uuid.uuid4().hex}{'}'}"
        content = math_display_pattern.sub(_math_disp_repl, content)
        content = math_inline_pattern.sub(_math_inl_repl, content)

        # 4) 对正文进行术语加粗（大小写不敏感，单词边界/中文场景都保守处理）
        technical_patterns = [
            r"\b(Transformer|BERT|GPT|ViT|ResNet|LSTM|GRU|CNN|RNN)\b",
            r"\b(Self\-Attention|Cross\-Attention|Multi\-Head Attention)\b",
            r"\b(Neural Network|Deep Learning|Machine Learning)\b",
            r"\b(Gradient Descent|Backpropagation|Dropout|Batch Normalization)\b",
            r"\b(Reinforcement Learning|Supervised Learning|Unsupervised Learning)\b",
            r"\b(Natural Language Processing|Computer Vision|Speech Recognition)\b",
            # 中文常见术语（保守，不覆盖日常词）：
            r"(自注意力|多头注意力|卷积神经网络|循环神经网络|深度学习|机器学习)"
        ]

        def bold_repl(match: re.Match) -> str:
            text = match.group(0)
            # 额外防护：若匹配项本身已包含加粗标记，直接返回
            if text.startswith("**") and text.endswith("**"):
                return text
            return f"**{text}**"

        for pattern in technical_patterns:
            content = re.sub(pattern, bold_repl, content, flags=re.IGNORECASE)

        # 5) 还原 URL / 行内代码 / 代码块 / 粗体 / 数学
        def restore_placeholders(text: str) -> str:
            # 使用顺序替换保证一一对应；由于使用 UUID，碰撞概率可忽略
            token_original_pairs = []
            token_original_pairs.extend([(f"URL_", u) for u in urls])
            token_original_pairs.extend([(f"IC_", ic) for ic in inline_codes])
            token_original_pairs.extend([(f"CB_", cb) for cb in codeblocks])
            token_original_pairs.extend([(f"BOLD_", b) for b in bold_segments])
            # 将数学分为显示与行内两类占位符统一还原
            token_original_pairs.extend([(f"MATHD_", m) for m in math_segments])
            token_original_pairs.extend([(f"MATHI_", m) for m in math_segments])

            for token, original in token_original_pairs:
                # 由于 token 中包含 UUID，不便逐个替换；采用正则按前缀匹配捕获并按序替换
                try:
                    pattern = re.compile(r"\\{" + re.escape(token) + r"[0-9a-fA-F]{32}\\}")
                    def _repl_fn(_m, _orig=original):
                        return _orig
                    text = pattern.sub(_repl_fn, text, count=1)
                except Exception:
                    # 失败则跳过该占位
                    pass
            return text

        content = restore_placeholders(content)
        return content
    
    def _format_quotes(self, content: str) -> str:
        """格式化引用块（在占位符保护下进行，避免污染代码块/行内代码/URL）。"""
        if not content:
            return content

        # 1) 占位保护代码块、行内代码与 URL
        codeblock_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        inline_code_pattern = re.compile(r"`[^`\n]+`")
        url_pattern = re.compile(r"\b(?:https?|ftp)://[\w\-\.\?\,\:/#%&=~+@;!\*\(\)]+", re.IGNORECASE)

        cb_store: List[str] = []
        ic_store: List[str] = []
        url_store: List[str] = []
        import uuid as _uuid2

        def _cb_repl(m):
            cb_store.append(m.group(0))
            return f"{'{'}QCB_{_uuid2.uuid4().hex}{'}'}"

        def _ic_repl(m):
            ic_store.append(m.group(0))
            return f"{'{'}QIC_{_uuid2.uuid4().hex}{'}'}"

        def _url_repl(m):
            url_store.append(m.group(0))
            return f"{'{'}QURL_{_uuid2.uuid4().hex}{'}'}"

        content = codeblock_pattern.sub(_cb_repl, content)
        content = inline_code_pattern.sub(_ic_repl, content)
        content = url_pattern.sub(_url_repl, content)

        # 2) 将重要引用转换为块级引用（仅在正文区域）
        quote_patterns = [
            r'(".*?")',  # 双引号引用
            r'(「.*?」)',  # 中文引号引用
        ]

        for pattern in quote_patterns:
            content = re.sub(pattern, r'> \1', content)

        # 3) 还原占位符
        # 使用前缀+UUID 的方式按序替换
        def _restore_seq(text: str, prefix: str, originals: List[str]) -> str:
            for original in originals:
                try:
                    pattern = re.compile(r"\\{" + re.escape(prefix) + r"[0-9a-fA-F]{32}\\}")
                    text = pattern.sub(lambda _m, _orig=original: _orig, text, count=1)
                except Exception:
                    pass
            return text

        content = _restore_seq(content, "QURL_", url_store)
        content = _restore_seq(content, "QIC_", ic_store)
        content = _restore_seq(content, "QCB_", cb_store)

        return content
    
    def _integrate_figures(self, content: str, figures: List[FigureReference]) -> str:
        """智能集成图片"""
        if not figures:
            return content
        
        # 按相关性排序
        sorted_figures = sorted(figures, key=lambda f: f.relevance_score, reverse=True)
        
        # 在适当位置插入图片
        for figure in sorted_figures:
            # 寻找最佳插入位置
            insertion_point = self._find_optimal_insertion_point(content, figure)
            if insertion_point > 0:
                img_markdown = f"\n\n![{figure.caption}]({figure.file_path})\n*{figure.caption}*\n\n"
                content = content[:insertion_point] + img_markdown + content[insertion_point:]
        
        return content
    
    def _find_optimal_insertion_point(self, content: str, figure: FigureReference) -> int:
        """寻找图片的最佳插入位置"""
        # 简化实现：根据上下文关键词匹配
        context_keywords = figure.context.lower().split()
        content_lower = content.lower()
        
        best_position = 0
        best_score = 0
        
        paragraphs = content.split('\n\n')
        current_pos = 0
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            score = sum(1 for keyword in context_keywords if keyword in paragraph_lower)
            
            if score > best_score:
                best_score = score
                best_position = current_pos + len(paragraph)
            
            current_pos += len(paragraph) + 2  # 加上 \n\n
        
        return best_position
    
    def _optimize_paragraphs(self, content: str) -> str:
        """优化段落结构"""
        # 确保段落间有适当的空行
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 确保标题前后有空行
        content = re.sub(r'(\n)([#]+.*?)(\n)', r'\1\n\2\n\n', content)
        
        return content
    
    def _clean_formatting(self, content: str) -> str:
        """清理多余的格式标记"""
        # 保护 LaTeX 数学块，避免后续空白清理破坏公式
        if not content:
            return content

        math_blocks = []  # type: ignore[var-annotated]
        placeholder_prefix = "MATHPH_"
        import uuid as _uuid

        def _stash_math(m: re.Match) -> str:
            token = placeholder_prefix + _uuid.uuid4().hex
            math_blocks.append((token, m.group(0)))
            return token

        # 行间与行内数学
        content = re.sub(r"\\\[[\s\S]*?\\\]", _stash_math, content)
        content = re.sub(r"\\\([^\)\n]*?\\\)", _stash_math, content)

        # 移除多余的空格（不触及已替换的占位符）
        content = re.sub(r' +', ' ', content)
        
        # 移除行尾空格
        content = re.sub(r' +\n', '\n', content)
        
        # 确保文章结尾只有一个换行
        content = content.rstrip() + '\n'

        # 还原数学块
        for token, original in math_blocks:
            content = content.replace(token, original)
        
        return content


class SynthesisEngine:
    """
    内容合成引擎
    
    将AnalysisCore的输出合成为完整的微信公众号文章，
    严格遵循依赖注入原则和模板化设计。
    """
    
    def __init__(self, 
                 config_manager: ConfigManager,
                 llm_provider: LLMProviderService,
                 memory_system: HybridMemorySystem):
        """
        无副作用构造函数
        
        Args:
            config_manager: 配置管理器
            llm_provider: LLM服务提供者
            memory_system: 记忆系统
        """
        self.config_manager = config_manager
        self.llm_provider = llm_provider
        self.memory_system = memory_system
        
        self.formatter = WeChatMarkdownFormatter()
        self.template_env: Optional[jinja2.Environment] = None
        self._initialized = False
        
        self.logger = get_logger(__name__)
    
    async def initialize(self) -> None:
        """异步初始化合成引擎"""
        if self._initialized:
            return
        
        try:
            # 初始化Jinja2模板环境
            template_path = Path(self.config_manager.paths.prompt_templates_folder)
            if not template_path.exists():
                raise FileNotFoundError(f"Template folder not found: {template_path}")
            
            self.template_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(template_path),
                enable_async=True,
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            self._initialized = True
            self.logger.info("SynthesisEngine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SynthesisEngine: {e}")
            raise
    
    async def synthesize_article(self,
                                analysis_pairs: List[AnalysisPair],
                                global_analysis: GlobalAnalysisMap,
                                paper_metadata: PaperMetadata,
                                figures: Optional[List[FigureReference]] = None) -> SynthesizedArticle:
        """
        合成完整文章
        
        Args:
            analysis_pairs: 分析对列表
            global_analysis: 全局分析结果
            paper_metadata: 论文元数据
            figures: 图片引用列表
            
        Returns:
            合成的文章
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            start_time = datetime.now()
            article_id = str(uuid.uuid4())
            
            self.logger.info(f"Starting article synthesis for {paper_metadata.title}")
            
            # 获取相关长期记忆
            long_term_memories = await self._retrieve_relevant_memories(global_analysis)
            
            # 生成文章各部分
            sections = await self._generate_article_sections(
                analysis_pairs, global_analysis, paper_metadata, long_term_memories
            )
            
            # 组装完整文章
            full_content = await self._assemble_article(sections)
            
            # 格式化为微信公众号样式
            # 先尝试语义驱动的图片插入；若未插入任何图片或失败，则由 formatter 插入
            content_with_figures = full_content
            inserted_semantically = False
            try:
                if figures:
                    before_len = len(full_content)
                    tmp = await self._insert_figures_semantically(full_content, figures)
                    # 判断是否产生了插入（粗略比较长度变化）
                    if tmp and len(tmp) > before_len:
                        content_with_figures = tmp
                        inserted_semantically = True
            except Exception as _e:
                self.logger.warning(f"Semantic figure insertion failed, fallback to formatter insertion: {_e}")

            if figures and not inserted_semantically:
                # 语义插入未发生，交由 formatter 执行插入
                formatted_content = self.formatter.format_content(content_with_figures, figures)
            else:
                # 已完成插入或无图片：避免重复插入，传空列表
                formatted_content = self.formatter.format_content(content_with_figures, [])
            
            # 计算字数
            word_count = len(re.findall(r'\S+', formatted_content))
            
            # 创建合成结果
            article = SynthesizedArticle(
                title=sections['title'].content,
                content=formatted_content,
                metadata=paper_metadata,
                figures=figures or [],
                word_count=word_count,
                generation_timestamp=datetime.now(timezone.utc),
                article_id=article_id
            )
            
            # 保存文章
            await self._save_article(article)
            
            # 记录性能指标
            duration = (datetime.now() - start_time).total_seconds()
            log_performance(
                operation="article_synthesis",
                duration=duration,
                metadata={
                    "word_count": word_count,
                    "sections_count": len(sections),
                    "figures_count": len(figures) if figures else 0,
                },
            )
            
            self.logger.info(f"Article synthesis completed: {word_count} words, {duration:.2f}s")
            return article
            
        except Exception as e:
            self.logger.error(f"Article synthesis failed: {e}")
            raise

    async def _insert_figures_semantically(self, content: str, figures: List[FigureReference]) -> str:
        """使用语义相似度将图片插入到最相关的段落末尾。

        要求：
        - 仅在可用嵌入服务时执行；否则抛出异常由上游回退。
        - 多图插入时，按相关性排序后逐一插入，动态维护偏移。
        """
        if not figures:
            return content

        # 获取嵌入服务
        embedding_service = None
        try:
            if isinstance(self.memory_system, HybridMemorySystem):
                embedding_service = self.memory_system.embedding_service
        except Exception:
            embedding_service = None
        if embedding_service is None:
            raise RuntimeError("Embedding service not available for semantic figure insertion")

        # 段落切分：优先按空行切分；若中文密集文本，进一步使用句子聚合形成段落
        paragraphs = content.split("\n\n")
        if len(paragraphs) <= 1 and len(content) > 0:
            # 简单的中文/英文句子切分，再合并为约 4-6 句一段
            sentence_endings = set(list("。！？!?.；;"))
            sentences: list[str] = []
            cur = []
            for ch in content:
                cur.append(ch)
                if ch in sentence_endings:
                    s = "".join(cur).strip()
                    if s:
                        sentences.append(s)
                    cur = []
            tail = "".join(cur).strip()
            if tail:
                sentences.append(tail)
            if sentences:
                grouped: list[str] = []
                acc: list[str] = []
                for i, s in enumerate(sentences, 1):
                    acc.append(s)
                    if i % 5 == 0:
                        grouped.append("".join(acc))
                        acc = []
                if acc:
                    grouped.append("".join(acc))
                if grouped:
                    paragraphs = grouped
        if not paragraphs:
            return content

        # 预先嵌入段落（一次性）
        paragraph_embeddings = await embedding_service.embed_texts(paragraphs)
        paragraph_matrix = np.array(paragraph_embeddings)
        # 归一化确保稳定余弦相似度
        norms = np.linalg.norm(paragraph_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        paragraph_matrix = paragraph_matrix / norms

        # 处理图片，按其原 relevance_score 降序（若相同，保持稳定顺序）
        sorted_figures = sorted(figures, key=lambda f: f.relevance_score, reverse=True)

        # 计算每张图片的最佳插入点
        # 构造累积偏移以处理多次插入
        offsets: List[int] = []  # 段落边界绝对位置列表（段落末尾位置）
        abs_pos = 0
        for p in paragraphs:
            abs_pos += len(p)
            offsets.append(abs_pos)
            abs_pos += 2  # for \n\n
        # 当前内容的可变副本
        result = content
        cumulative_delta = 0

        for fig in sorted_figures:
            if not getattr(fig, "context", None):
                continue
            # 嵌入图片上下文
            ctx_vec = await embedding_service.embed_text(fig.context)
            ctx_vec = np.array(ctx_vec)
            denom = np.linalg.norm(ctx_vec) or 1.0
            ctx_vec = ctx_vec / denom

            # 计算与所有段落的余弦相似度
            sims = paragraph_matrix.dot(ctx_vec)
            best_idx = int(np.argmax(sims))
            # 计算插入的绝对位置（考虑累计插入偏移）
            insert_at = offsets[best_idx] + cumulative_delta
            img_md = f"\n\n![{fig.caption}]({fig.file_path})\n*{fig.caption}*\n\n"
            result = result[:insert_at] + img_md + result[insert_at:]
            cumulative_delta += len(img_md)

        return result
    
    async def _retrieve_relevant_memories(self, global_analysis: GlobalAnalysisMap) -> List[MemorySearchResult]:
        """检索相关的长期记忆"""
        try:
            # 构造查询内容
            query_content = f"{global_analysis.summary} {' '.join(global_analysis.key_insights)}"
            
            # 从记忆系统检索相关内容
            memories = await self.memory_system.search_memory(
                query=query_content,
                memory_types=[MemoryType.EPISODIC, MemoryType.CONCEPTUAL],
                limit=5,
                similarity_threshold=0.7,
                include_related=True,
                search_depth=2,
            )
            
            return memories
            
        except Exception as e:
            self.logger.warning(f"Failed to retrieve relevant memories: {e}")
            return []
    
    async def _generate_article_sections(self,
                                       analysis_pairs: List[AnalysisPair],
                                       global_analysis: GlobalAnalysisMap,
                                       paper_metadata: PaperMetadata,
                                       long_term_memories: List[MemorySearchResult]) -> Dict[str, ArticleSection]:
        """生成文章各个部分"""
        sections = {}
        
        # 先生成标题，以便引言能使用选定标题（杜绝简化）
        try:
            titles = await self._generate_titles(global_analysis, analysis_pairs, long_term_memories)
        except Exception:
            titles = ["深度解读最新研究"]
        selected_title = titles[0] if isinstance(titles, list) and titles else "深度解读最新研究"

        # 并行生成其余部分（引言使用选定标题）
        tasks = [
            self._generate_introduction(selected_title, global_analysis, paper_metadata, long_term_memories),
            self._generate_key_insights(global_analysis, analysis_pairs, long_term_memories),
            self._generate_deep_thinking(global_analysis, analysis_pairs, paper_metadata, long_term_memories)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        introduction = results[0] if not isinstance(results[0], Exception) else "本文将深入分析这项重要研究。"
        key_insights = results[1] if not isinstance(results[1], Exception) else "## 核心洞察\n\n暂无详细分析。"
        deep_thinking = results[2] if not isinstance(results[2], Exception) else "## Deep Scholar AI深度思考\n\n持续分析中。"
        
        sections['title'] = ArticleSection('title', selected_title)
        sections['introduction'] = ArticleSection('introduction', introduction)
        sections['insights'] = ArticleSection('insights', key_insights)
        sections['thinking'] = ArticleSection('thinking', deep_thinking)
        
        # 生成主要内容部分
        content_sections = await self._generate_main_content(analysis_pairs)
        sections['content'] = ArticleSection('content', content_sections)
        
        return sections
    
    async def _generate_titles(self,
                             global_analysis: GlobalAnalysisMap,
                             analysis_pairs: List[AnalysisPair],
                             long_term_memories: List[MemorySearchResult]) -> List[str]:
        """生成标题候选"""
        try:
            template = self.template_env.get_template('synthesis_title_generation.jinja2')
            prompt = await template.render_async(
                global_analysis=global_analysis,
                analysis_pairs=analysis_pairs,
                long_term_memories=long_term_memories
            )
            
            request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                model=self.config_manager.llm_settings.primary_model,
                temperature=0.8,
                max_tokens=500
            )
            
            response = await self.llm_provider.generate(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                use_insight_model=True,
            )
            
            # 解析标题（鲁棒解析：支持“标题X:”或“标题X：”或纯行标题）
            titles: List[str] = []
            for raw in response.content.split('\n'):
                line = raw.strip()
                if not line:
                    continue
                # 常见前缀
                if line.startswith('标题'):
                    sep_pos = max(line.find(':'), line.find('：'))
                    if sep_pos != -1:
                        candidate = line[sep_pos + 1 :].strip()
                        if candidate:
                            titles.append(candidate)
                            continue
                # 兜底：行本身作为标题候选（长度控制）
                if 4 <= len(line) <= 50 and not line.startswith('#'):
                    titles.append(line)

            # 去重，保留顺序
            seen: Set[str] = set()
            unique_titles: List[str] = []
            for t in titles:
                if t not in seen:
                    seen.add(t)
                    unique_titles.append(t)

            return unique_titles[:5] if unique_titles else ["深度解读最新研究"]
            
        except Exception as e:
            self.logger.error(f"Title generation failed: {e}")
            return ["深度解读最新研究"]
    
    async def _generate_introduction(self,
                                   selected_title: str,
                                   global_analysis: GlobalAnalysisMap,
                                   paper_metadata: PaperMetadata,
                                   long_term_memories: List[MemorySearchResult]) -> str:
        """生成引言"""
        try:
            template = self.template_env.get_template('synthesis_introduction.jinja2')
            prompt = await template.render_async(
                selected_title=selected_title,
                global_analysis=global_analysis,
                paper_metadata=paper_metadata,
                long_term_memories=long_term_memories
            )
            
            request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                model=self.config_manager.llm_settings.insight_model,
                temperature=0.6,
                max_tokens=900
            )
            
            response = await self.llm_provider.generate(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                use_insight_model=True,
            )
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"Introduction generation failed: {e}")
            return "本文将深入分析这项重要研究的核心贡献和技术创新。"
    
    async def _generate_key_insights(self,
                                   global_analysis: GlobalAnalysisMap,
                                   analysis_pairs: List[AnalysisPair],
                                  long_term_memories: List[MemorySearchResult]) -> str:
        """生成核心洞察"""
        try:
            template = self.template_env.get_template('synthesis_key_insights.jinja2')
            prompt = await template.render_async(
                global_analysis=global_analysis,
                analysis_pairs=analysis_pairs,
                long_term_memories=long_term_memories
            )
            
            request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                model=self.config_manager.llm_settings.insight_model,
                temperature=0.5,
                max_tokens=min(
                    int(getattr(self.config_manager.llm_settings, 'max_tokens', 4000) or 4000),
                    8000
                )
            )
            
            response = await self.llm_provider.generate(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                use_insight_model=True,
            )
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"Key insights generation failed: {e}")
            return "## 核心洞察\n\n分析正在进行中，请稍后查看详细内容。"
    
    async def _generate_deep_thinking(self,
                                    global_analysis: GlobalAnalysisMap,
                                    analysis_pairs: List[AnalysisPair],
                                    paper_metadata: PaperMetadata,
                                   long_term_memories: List[MemorySearchResult]) -> str:
        """生成深度思考部分"""
        try:
            # 先生成核心洞察用于上下文
            key_insights_content = await self._generate_key_insights(global_analysis, analysis_pairs, long_term_memories)
            
            template = self.template_env.get_template('synthesis_deep_thinking.jinja2')
            prompt = await template.render_async(
                paper_metadata=paper_metadata,
                global_analysis=global_analysis,
                key_insights=[key_insights_content],
                analysis_pairs=analysis_pairs,
                long_term_memories=long_term_memories
            )
            
            # 使用最强模型进行深度思考（散文体，无标题/列表）
            request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                model=self.config_manager.llm_settings.insight_model,
                temperature=0.5,
                max_tokens=3000
            )
            
            response = await self.llm_provider.generate(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            # 直接返回散文体内容（不加小标题，保持自然人类叙述风格）
            content = response.content.strip()
            return content
            
        except Exception as e:
            self.logger.error(f"Deep thinking generation failed: {e}")
            return "## Deep Scholar AI深度思考\n\n深度分析正在进行中，我们将为您提供最专业的学术见解。"
    
    async def _generate_main_content(self, analysis_pairs: List[AnalysisPair]) -> str:
        """生成主要内容部分"""
        try:
            content_parts = []
            
            for _, pair in enumerate(analysis_pairs, 1):
                # 取消二级标题和引用块，改为自然衔接的连续叙述
                paragraph = pair.analysis.strip()
                if paragraph:
                    content_parts.append(paragraph)
            
            # 段落之间以空行分隔，避免列表化风格
            return "\n\n".join(content_parts)
            
        except Exception as e:
            self.logger.error(f"Main content generation failed: {e}")
            return "## 技术解读\n\n详细分析正在生成中。"
    
    async def _assemble_article(self, sections: Dict[str, ArticleSection]) -> str:
        """组装完整文章"""
        try:
            article_parts = []
            
            # 标题
            article_parts.append(f"# {sections['title'].content}\n")
            
            # 引言
            article_parts.append(sections['introduction'].content)
            article_parts.append("\n")
            
            # 核心洞察
            article_parts.append(sections['insights'].content)
            article_parts.append("\n")
            
            # 主要内容
            article_parts.append(sections['content'].content)
            article_parts.append("\n")
            
            # 深度思考
            article_parts.append(sections['thinking'].content)
            
            return "\n".join(article_parts)
            
        except Exception as e:
            self.logger.error(f"Article assembly failed: {e}")
            raise
    
    async def _save_article(self, article: SynthesizedArticle) -> None:
        """保存文章到文件系统"""
        try:
            # 生成文件名
            timestamp = article.generation_timestamp.strftime("%Y%m%d_%H%M%S")
            safe_title = re.sub(r'[^\w\-_\.]', '_', article.title)[:50]
            filename = f"{timestamp}_{safe_title}_{article.article_id[:8]}.md"
            
            # 确保输出目录存在
            output_dir = Path(self.config_manager.paths.drafts_folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = output_dir / filename
            
            # 若包含图片：先复制图片到输出目录下的专属文件夹，并重写内容中的图片链接
            content_to_save = article.content
            original_parents: set = set()
            if article.figures:
                figure_folder = output_dir / f"{article.article_id[:8]}_figures"
                figure_folder.mkdir(exist_ok=True)
                old_to_new_paths = {}
                for fig in article.figures:
                    try:
                        src_path = Path(fig.file_path)
                        # 跳过不存在的源文件（尽量避免中断保存流程）
                        if not src_path.exists():
                            continue
                        original_parents.add(str(src_path.parent))
                        dest_name = src_path.name
                        dest_path = figure_folder / dest_name
                        # 执行复制（后台线程避免阻塞）
                        await asyncio.to_thread(shutil.copyfile, str(src_path), str(dest_path))
                        # 文章内使用相对路径，确保草稿可移植
                        new_rel_path = f"./{figure_folder.name}/{dest_name}"
                        old_to_new_paths[str(src_path)] = new_rel_path
                        # 更新 FigureReference 内的路径，供 figures_info.json 使用
                        fig.file_path = new_rel_path
                    except Exception:
                        # 单张图片失败不应中断文章保存
                        continue
                # 重写内容中的链接
                for old, new in old_to_new_paths.items():
                    content_to_save = content_to_save.replace(old, new)
            
            # 准备文章内容（包含元数据）
            metadata_header = self._generate_metadata_header(article)
            full_content = f"{metadata_header}\n\n{content_to_save}"
            
            # 异步保存文件
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(full_content)
            
            # 如果有图片，创建关联的图片文件夹并记录信息
            if article.figures:
                await self._setup_figure_folder(article, output_dir)
            
            await log_synthesis_operation(
                operation="article_saved",
                article_id=article.article_id,
                file_path=str(file_path),
                word_count=article.word_count
            )
            
            # 写回保存路径
            article.file_path = str(file_path)

            # 复制完成后，尽力清理原始临时图片目录（若存在）
            try:
                parents = {Path(p) for p in original_parents}
                for p in parents:
                    # 仅清理位于 temp_downloads 路径下的目录，避免误删
                    try:
                        temp_root = Path(self.config_manager.paths.temp_downloads_folder).resolve()
                        if p.resolve().is_dir() and str(p.resolve()).startswith(str(temp_root)):
                            await asyncio.to_thread(shutil.rmtree, str(p))
                    except Exception:
                        # 清理失败不影响主流程
                        continue
            except Exception:
                pass

            self.logger.info(f"Article saved: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save article: {e}")
            raise
    
    def _generate_metadata_header(self, article: SynthesizedArticle) -> str:
        """生成文章元数据头部"""
        metadata_lines = [
            "---",
            f"title: {article.title}",
            f"article_id: {article.article_id}",
            f"generated_at: {article.generation_timestamp.isoformat()}",
            f"word_count: {article.word_count}",
            f"source: Deep Scholar AI"
        ]
        
        if article.metadata and article.metadata.title:
            metadata_lines.append(f"paper_title: {article.metadata.title}")
        
        if article.metadata and article.metadata.authors:
            metadata_lines.append(f"paper_authors: {', '.join(article.metadata.authors)}")
        
        if article.metadata and article.metadata.arxiv_id:
            metadata_lines.append(f"arxiv_id: {article.metadata.arxiv_id}")
        
        metadata_lines.append("---")
        
        return "\n".join(metadata_lines)
    
    async def _setup_figure_folder(self, article: SynthesizedArticle, output_dir: Path) -> None:
        """设置图片文件夹"""
        try:
            # 创建图片文件夹
            figure_folder = output_dir / f"{article.article_id[:8]}_figures"
            figure_folder.mkdir(exist_ok=True)
            
            # 记录图片信息
            figure_info = []
            for figure in article.figures:
                figure_info.append({
                    "id": figure.figure_id,
                    "caption": figure.caption,
                    "file_path": figure.file_path,
                    "relevance_score": figure.relevance_score
                })
            
            # 保存图片信息文件
            info_file = figure_folder / "figures_info.json"
            async with aiofiles.open(info_file, 'w', encoding='utf-8') as f:
                import json
                await f.write(json.dumps(figure_info, ensure_ascii=False, indent=2))
            
        except Exception as e:
            self.logger.warning(f"Failed to setup figure folder: {e}")
