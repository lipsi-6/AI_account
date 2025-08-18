"""
论文分析核心模块 - 系统的"大脑"

实现真正深刻的、连贯的论文解读，包括：
1. GlobalAnalysisMap生成 - 全局分析上下文
2. 智能意群切分 - SOTA语义嵌入+聚类算法
3. 逐意群递归讲解 - 基于Master Prompt的深度解读

严格要求：
- LLMProvider作为依赖注入，绝不内部实例化
- Prompt管理：外部Jinja2模板文件
- 无成本限制，追求极致精细
- 使用memory_system检索相关历史
- 完整的错误处理和日志记录
"""

import asyncio
import time
import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import json
import uuid

import numpy as np
import jinja2

from .config_manager import ConfigManager
from .llm_provider import LLMProviderService, LLMResponse
from .memory_system import HybridMemorySystem, MemoryType, MemorySearchQuery
from .embedding_provider import EmbeddingProviderService
from .logger import get_logger, log_performance
from .learning_loop import LearningLoop


class AnalysisType(Enum):
    """分析类型枚举"""
    GLOBAL_OVERVIEW = "global_overview"
    SEMANTIC_CHUNK = "semantic_chunk"
    DETAILED_EXPLANATION = "detailed_explanation"
    CONCEPT_EXTRACTION = "concept_extraction"


class ChunkingStrategy(Enum):
    """切分策略枚举"""
    PARAGRAPH_BASED = "paragraph_based"
    SEMANTIC_EMBEDDING = "semantic_embedding"
    HIERARCHICAL_HYBRID = "hierarchical_hybrid"


@dataclass
class GlobalAnalysisMap:
    """全局分析地图 - 作为所有LLM调用的上下文"""
    paper_id: str
    title: str
    authors: List[str]
    abstract_summary: str
    research_domain: str
    key_contributions: List[str]
    methodology_overview: str
    main_findings: List[str]
    limitations: List[str]
    future_directions: List[str]
    technical_complexity: float  # 0-1评分
    novelty_score: float  # 0-1评分
    practical_relevance: float  # 0-1评分
    related_concepts: List[str]
    terminology_glossary: Dict[str, str] = field(default_factory=dict)
    mathematical_concepts: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SemanticChunk:
    """语义意群"""
    chunk_id: str
    content: str
    start_position: int
    end_position: int
    section_type: str  # abstract, introduction, method, results, conclusion, etc.
    semantic_topic: str
    embedding: Optional[List[float]] = None
    importance_score: float = 1.0
    complexity_level: int = 1  # 1-5级别
    mathematical_content: bool = False
    contains_formulas: bool = False
    key_concepts: List[str] = field(default_factory=list)
    related_chunks: List[str] = field(default_factory=list)


@dataclass
class ChunkExplanation:
    """意群解释结果"""
    chunk_id: str
    explanation: str
    key_insights: List[str]
    terminology_explained: Dict[str, str]
    mathematical_explanations: Dict[str, str]
    connections_to_other_chunks: List[str]
    confidence_score: float
    processing_time: float
    llm_model_used: str


@dataclass
class AnalysisRequest:
    """分析请求"""
    paper_content: str
    paper_title: str
    paper_authors: List[str]
    analysis_depth: int = 3  # 1-5级别
    include_mathematical_details: bool = True
    include_historical_context: bool = True
    custom_focus_areas: Optional[List[str]] = None
    output_language: str = "zh-CN"


@dataclass
class AnalysisResult:
    """分析结果"""
    paper_id: str
    global_map: GlobalAnalysisMap
    semantic_chunks: List[SemanticChunk]
    chunk_explanations: List[ChunkExplanation]
    overall_summary: str
    research_insights: List[str]
    practical_implications: List[str]
    related_work_connections: List[str]
    total_processing_time: float
    quality_metrics: Dict[str, float]


class AnalysisError(Exception):
    """分析错误基类"""
    pass


class GlobalAnalysisError(AnalysisError):
    """全局分析错误"""
    pass


class ChunkingError(AnalysisError):
    """切分错误"""
    pass


class ExplanationError(AnalysisError):
    """解释错误"""
    pass


class PromptTemplateManager:
    """Prompt模板管理器"""
    
    def __init__(self, templates_dir: Path):
        """
        初始化模板管理器
        
        Args:
            templates_dir: 模板文件目录
        """
        self.templates_dir = Path(templates_dir)
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        # 确保 tojson 过滤器存在，避免模板中使用 tojson 时环境缺失导致渲染失败
        if 'tojson' not in self.env.filters:
            try:
                import json as _json
                self.env.filters['tojson'] = lambda v: _json.dumps(v, ensure_ascii=False)
            except Exception:
                # 保底：转换为字符串
                self.env.filters['tojson'] = lambda v: str(v)
        self.logger = get_logger("prompt_template_manager")

    async def initialize(self) -> None:
        """显式初始化：确保基础模板存在（避免构造函数副作用）。"""
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_base_templates()
    
    def _ensure_base_templates(self) -> None:
        """确保基础模板文件存在"""
        base_templates = {
            "global_analysis.j2": """
您是一位世界顶级的学术研究专家，具有深厚的跨学科知识背景。请对以下学术论文进行全局深度分析，并严格按照指定 JSON 架构输出。

论文信息：
- 标题: {{ title }}
- 作者: {{ authors | join(', ') }}

论文内容：
{{ content }}

输出要求（极其重要）：
- 严格输出 UTF-8 单个 JSON 对象（无额外文字、无 markdown 代码块），键名与类型必须完全匹配下述架构；字符串请避免换行过多。
- 架构键名（全部必填）：
  {
    "abstract_summary": string,                      // 摘要层面的全局总结
    "research_domain": string,                       // 研究领域（如：Natural Language Processing）
    "key_contributions": string[],                   // 主要学术贡献（3-8 条）
    "methodology_overview": string,                  // 方法论/技术路线概述
    "main_findings": string[],                       // 关键实验或理论发现（3-8 条）
    "limitations": string[],                         // 局限性（2-6 条）
    "future_directions": string[],                   // 未来工作方向（2-6 条）
    "technical_complexity": number,                  // 技术复杂度评分 ∈ [0,1]
    "novelty_score": number,                         // 新颖性评分 ∈ [0,1]
    "practical_relevance": number,                   // 实用相关性评分 ∈ [0,1]
    "related_concepts": string[],                    // 相关概念/术语（5-15 个，短语）
    "terminology_glossary": {                        // 术语词汇表：术语 -> 简要定义
      "<term>": "<definition>",
      "...": "..."
    },
    "mathematical_concepts": string[]                // 涉及的数学概念/对象（如：KL divergence, convexity, spectral norm）
  }

请直接输出满足上述键名与类型约束的 JSON 对象。
""",
            
            "semantic_chunking.j2": """
您是文本语义分析专家。请基于以下全局分析上下文对论文内容进行智能意群切分。

## 全局分析上下文
{{ global_context | tojson }}

## 论文内容
{{ content }}

## 切分要求
1. **语义连贯性**: 确保每个意群内部语义连贯
2. **逻辑完整性**: 避免在逻辑论述中间切分
3. **数学公式处理**: 数学公式与其解释保持在同一意群
4. **重要性评估**: 为每个意群评估重要性分数(0-1)
5. **复杂度分级**: 评估每个意群的技术复杂度(1-5级)
6. **概念标注**: 标识每个意群包含的关键概念

请返回语义切分结果，包含详细的元数据信息。
""",
            
            "chunk_explanation.j2": """
您是学术论文解读专家，擅长将复杂的学术内容转化为深入浅出的解释。

## 全局分析上下文
{{ global_context | tojson }}

## 历史记忆上下文
{% if memory_context %}
### 相关历史知识
{% for memory in memory_context %}
- {{ memory.content or memory.name }} (相似度: {{ "%.2f" | format(memory.similarity_score) }})
{% endfor %}
{% endif %}

## 当前意群信息
### 意群内容
{{ chunk.content }}

### 意群元数据
- 章节类型: {{ chunk.section_type }}
- 语义主题: {{ chunk.semantic_topic }}
- 重要性评分: {{ chunk.importance_score }}
- 复杂度级别: {{ chunk.complexity_level }}
- 包含数学内容: {{ chunk.mathematical_content }}
- 包含公式: {{ chunk.contains_formulas }}
- 关键概念: {{ chunk.key_concepts | join(', ') }}

## 解释要求
请为这个意群提供深度解释，包括：

1. **核心内容解释**: 用{{ output_language }}深入浅出地解释主要内容
2. **关键洞察提取**: 识别这个意群的关键学术洞察
3. **术语解释**: 解释涉及的专业术语和概念
4. **数学公式说明**: 如果包含数学内容，提供详细的数学解释
5. **与其他部分的联系**: 分析与论文其他部分的逻辑关系
6. **置信度评估**: 对解释质量进行自我评估(0-1分)

请确保解释既保持学术严谨性，又具有良好的可读性。
""",
            
            "chunk_explanation_json.j2": """
您是学术论文解读专家，擅长将复杂的学术内容转化为深入浅出的解释。

现在请基于以下信息，严格输出一个 JSON 对象（UTF-8，无前后说明文字、无 markdown 代码块，无额外键）：

必需字段（全部必填）：
- explanation: string  // 面向 {{ output_language }} 的高质量深入解释（段落文本）
- key_insights: string[]  // 3-8 条关键洞察，每条为完整句子
- terminology_explained: object  // 术语词典，键为术语，值为其{{ output_language }}解释
- mathematical_explanations: object  // 数学相关概念或公式的解释，键为概念或符号，值为解释
- confidence_score: number  // 0-1 之间的小数，代表你对本段解释质量的置信度

上下文：
## 全局分析上下文
{{ global_context | tojson }}

## 历史记忆上下文（如无可为空数组）
{% if memory_context %}
{{ memory_context | tojson }}
{% else %}
[]
{% endif %}

## 当前意群信息
{{ chunk | tojson }}

输出要求：
1. 严格输出单个 JSON 对象，不能有任何额外文本
2. 所有字段必须存在且类型正确
3. explanation 为高质量连续文本，不得为空
4. key_insights 每条不超过 200 字
5. confidence_score ∈ [0,1]
6. 风格约束（重要）：explanation 必须为中文散文体连续叙述，不使用列表、编号、条目符号（如"-"、"1.")、小标题或分节，不使用“首先/其次/再次/最后”等机械衔接词，不出现“作为AI”等自我指涉或模板化语句；尽量避免口号式总结，优先给出可检验的推理链与约束条件。
""",
            
            "master_prompt.j2": """
# 深度学者AI - Master Prompt

您是世界顶级的学术研究专家和知识解读大师，具备以下特质：

## 核心能力
- **跨学科专业知识**: 精通计算机科学、数学、物理、生物学、经济学等多个领域
- **深度分析能力**: 能够识别复杂学术内容的核心本质和深层联系
- **清晰表达能力**: 能将复杂概念转化为易于理解的表述
- **批判性思维**: 具备客观分析和评估学术工作的能力
- **创新洞察力**: 能够识别研究的创新点和潜在影响

## 分析原则
1. **准确性第一**: 确保所有解释和分析的学术准确性
2. **深度优于广度**: 宁可深入分析少数要点，也不浅尝辄止，健谈且实事求是，从第一性原理思考，避免过度简化或夸大。
3. **逻辑连贯性**: 保持前后一致的逻辑链条
4. **实用性导向**: 关注研究的实际应用价值和意义
5. **批判性审视**: 客观评估研究的优缺点和局限性

## 输出要求
- 使用{{ output_language }}进行解释
- 保持学术严谨性的同时确保可读性
- 提供具体的例子和类比来辅助理解
- 明确指出关键概念和重要发现
- 建立与现有知识体系的联系

## 上下文信息
### 全局分析地图
{{ global_context | tojson }}

### 记忆检索结果
{% if memory_context %}
{% for memory in memory_context %}
**记忆条目 {{ loop.index }}**:
{{ memory.content }}
(相似度: {{ "%.3f" | format(memory.similarity_score) }}, 类型: {{ memory.memory_type.value }})
{% endfor %}
{% endif %}

请基于以上上下文和您的专业知识，对接下来的内容进行深度分析。
"""
        }
        
        for template_name, template_content in base_templates.items():
            template_path = self.templates_dir / template_name
            if not template_path.exists():
                template_path.write_text(template_content.strip(), encoding='utf-8')
                self.logger.info(f"Created template: {template_name}")
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """
        渲染模板
        
        Args:
            template_name: 模板名称
            **kwargs: 模板变量
            
        Returns:
            渲染后的内容
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to render template {template_name}: {e}")
            raise AnalysisError(f"Template rendering failed: {e}")


class SemanticChunker:
    """SOTA语义切分器"""
    
    def __init__(self, embedding_service: EmbeddingProviderService, llm_service: Optional[LLMProviderService] = None):
        """
        初始化语义切分器
        
        Args:
            embedding_service: 嵌入服务
        """
        self.embedding_service = embedding_service
        self.llm_service: Optional[LLMProviderService] = llm_service
        self.logger = get_logger("semantic_chunker")
    
    async def chunk_document(
        self,
        content: str,
        global_map: GlobalAnalysisMap,
        strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL_HYBRID,
        target_chunk_size: int = 500,
        similarity_threshold: float = 0.75
    ) -> List[SemanticChunk]:
        """
        对文档进行语义切分
        
        Args:
            content: 文档内容
            global_map: 全局分析地图
            strategy: 切分策略
            target_chunk_size: 目标意群大小
            similarity_threshold: 相似度阈值
            
        Returns:
            语义意群列表
        """
        start_time = time.time()
        
        try:
            if strategy == ChunkingStrategy.HIERARCHICAL_HYBRID:
                chunks = await self._hierarchical_hybrid_chunking(
                    content, global_map, target_chunk_size, similarity_threshold
                )
            elif strategy == ChunkingStrategy.SEMANTIC_EMBEDDING:
                chunks = await self._semantic_embedding_chunking(
                    content, global_map, target_chunk_size, similarity_threshold
                )
            else:
                chunks = await self._paragraph_based_chunking(
                    content, global_map, target_chunk_size
                )
            
            # 后处理：计算意群间的关系
            chunks = await self._compute_chunk_relationships(chunks)
            
            duration = time.time() - start_time
            log_performance("semantic_chunking", duration, chunks_count=len(chunks))
            
            self.logger.info(f"Generated {len(chunks)} semantic chunks using {strategy.value} strategy")
            return chunks
            
        except Exception as e:
            raise ChunkingError(f"Semantic chunking failed: {e}")
    
    async def _hierarchical_hybrid_chunking(
        self,
        content: str,
        global_map: GlobalAnalysisMap,
        target_chunk_size: int,
        similarity_threshold: float
    ) -> List[SemanticChunk]:
        """分层混合切分策略"""
        
        # 第一阶段：基于结构的粗切分
        coarse_chunks = self._structure_based_splitting(content)
        
        # 第二阶段：语义嵌入细粒度切分
        refined_chunks = []
        
        for coarse_chunk in coarse_chunks:
            if len(coarse_chunk['content']) <= target_chunk_size:
                # 小于目标大小，直接保留
                chunk = await self._create_semantic_chunk(
                    coarse_chunk['content'],
                    coarse_chunk['start'],
                    coarse_chunk['end'],
                    coarse_chunk['section_type'],
                    global_map
                )
                refined_chunks.append(chunk)
            else:
                # 大于目标大小，进行语义切分
                sub_chunks = await self._semantic_split_large_chunk(
                    coarse_chunk, target_chunk_size, similarity_threshold, global_map
                )
                refined_chunks.extend(sub_chunks)
        
        return refined_chunks
    
    def _structure_based_splitting(self, content: str) -> List[Dict[str, Any]]:
        """基于结构的粗切分"""
        chunks = []
        
        # 定义章节模式
        section_patterns = [
            (r'\n\s*#+\s*(.+)', 'heading'),
            (r'\n\s*Abstract\s*\n', 'abstract'),
            (r'\n\s*Introduction\s*\n', 'introduction'),
            (r'\n\s*Related Work\s*\n', 'related_work'),
            (r'\n\s*Methodology\s*\n', 'methodology'),
            (r'\n\s*Method\s*\n', 'method'),
            (r'\n\s*Experiments?\s*\n', 'experiments'),
            (r'\n\s*Results?\s*\n', 'results'),
            (r'\n\s*Discussion\s*\n', 'discussion'),
            (r'\n\s*Conclusion\s*\n', 'conclusion'),
            (r'\n\s*References?\s*\n', 'references'),
        ]
        
        # 查找章节边界
        section_boundaries = []
        for pattern, section_type in section_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                section_boundaries.append({
                    'position': match.start(),
                    'section_type': section_type,
                    'title': match.group(1) if match.groups() else section_type
                })
        
        # 按位置排序
        section_boundaries.sort(key=lambda x: x['position'])
        
        # 生成粗切分块
        for i, boundary in enumerate(section_boundaries):
            start_pos = boundary['position']
            end_pos = section_boundaries[i + 1]['position'] if i + 1 < len(section_boundaries) else len(content)
            
            chunk_content = content[start_pos:end_pos].strip()
            if chunk_content:
                chunks.append({
                    'content': chunk_content,
                    'start': start_pos,
                    'end': end_pos,
                    'section_type': boundary['section_type']
                })
        
        # 如果没有找到明显的章节结构，按段落切分
        if not chunks:
            paragraphs = content.split('\n\n')
            current_pos = 0
            for para in paragraphs:
                if para.strip():
                    chunks.append({
                        'content': para.strip(),
                        'start': current_pos,
                        'end': current_pos + len(para),
                        'section_type': 'paragraph'
                    })
                current_pos += len(para) + 2
        
        return chunks
    
    async def _semantic_split_large_chunk(
        self,
        large_chunk: Dict[str, Any],
        target_size: int,
        similarity_threshold: float,
        global_map: GlobalAnalysisMap
    ) -> List[SemanticChunk]:
        """对大块进行语义切分"""
        
        content = large_chunk['content']
        
        # 将大块按句子分割
        sentences = self._split_into_sentences(content)
        
        if len(sentences) <= 1:
            # 只有一个句子，无法进一步切分
            chunk = await self._create_semantic_chunk(
                content,
                large_chunk['start'],
                large_chunk['end'],
                large_chunk['section_type'],
                global_map
            )
            return [chunk]
        
        # 生成句子嵌入
        sentence_embeddings = await self.embedding_service.embed_texts(sentences)

        # 使用聚类算法进行语义聚类（先计算相似度矩阵）
        similarity_matrix = self._cosine_similarity_matrix(sentence_embeddings)
        
        # 使用层次聚类（确保聚类数在合法范围内，且不超过样本数）
        # 估算聚类数：与目标块大小成比例；短文本允许单簇，避免过度分裂
        estimated = max(1, len(sentences) * target_size // max(1, len(content)))
        if len(sentences) <= 3:
            n_clusters = 1
        else:
            n_clusters = max(1, min(len(sentences), estimated))

        # 优先使用直接基于嵌入的聚类（metric='cosine'）
        try:
            from sklearn.cluster import AgglomerativeClustering  # 延迟导入，避免全局导入期错误
        except Exception as e:
            raise ChunkingError(
                f"Semantic chunking requires scikit-learn; import failed: {e}. "
                "Please install a scikit-learn version compatible with your NumPy."
            )

        try:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(np.array(sentence_embeddings))
        except Exception:
            # 回退：以预计算距离矩阵方式进行
            distance_matrix = 1 - similarity_matrix
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='precomputed',
                    linkage='average'
                )
            except TypeError:
                # 兼容旧版本：使用 affinity 参数
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    linkage='average'
                )
            cluster_labels = clustering.fit_predict(distance_matrix)
        
        # 根据聚类结果生成意群
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        semantic_chunks = []
        current_pos = large_chunk['start']
        
        for cluster_sentences in clusters.values():
            cluster_content = ' '.join(sentences[i] for i in cluster_sentences)
            chunk_start = current_pos
            chunk_end = chunk_start + len(cluster_content)
            
            chunk = await self._create_semantic_chunk(
                cluster_content,
                chunk_start,
                chunk_end,
                large_chunk['section_type'],
                global_map
            )
            semantic_chunks.append(chunk)
            current_pos = chunk_end + 1
        
        return semantic_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子（增强的中英文混合支持，面向中文优化）。

        策略：
        - 优先按中文句末标点与英文句末标点进行边界切分（保留标点与右侧引号/括号）
        - 兼容省略号（……/…）与复合标点（？！等）
        - 对缺乏空格的中文文本进行精细切分；英文保持原逻辑
        - 最终进行去噪：去除过短片段
        """
        if not text:
            return []

        # 快速检测是否包含 CJK 字符
        contains_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))

        # 扫描式切分，保留句末标点与尾随右引号/右括号
        endings = set(list("。！？!?.；;"))  # 句末标点集合
        right_trailers = set(list("”’』」）)】]"))  # 可能跟随在句末标点后的右侧符号

        sentences: List[str] = []
        current: List[str] = []
        i = 0
        text_len = len(text)
        while i < text_len:
            ch = text[i]
            current.append(ch)

            # 处理省略号：中文“……”或单个“…”；英文“...”
            def is_ellipsis_at(pos: int) -> int:
                # 返回省略号长度（0 表示不是省略号）
                if pos + 1 < text_len and text[pos] == '…' and text[pos + 1] == '…':
                    return 2
                if text[pos] == '…':
                    return 1
                if pos + 2 < text_len and text[pos] == '.' and text[pos + 1] == '.' and text[pos + 2] == '.':
                    return 3
                return 0

            ellipsis_len = is_ellipsis_at(i)
            is_sentence_end = False

            if ellipsis_len:
                # 将省略号整体并入当前句
                for _ in range(ellipsis_len - 1):
                    i += 1
                    if i < text_len:
                        current.append(text[i])
                is_sentence_end = True
            elif ch in endings:
                # 句末标点，可能后面还跟随右引号/右括号
                # 消费连续的右侧符号
                j = i + 1
                while j < text_len and text[j] in right_trailers:
                    current.append(text[j])
                    j += 1
                # 将 i 前进到消费后的最后位置
                i = j - 1
                is_sentence_end = True

            if is_sentence_end:
                sentence = ''.join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []

            i += 1

        # 处理收尾残留
        tail = ''.join(current).strip()
        if tail:
            sentences.append(tail)

        # 清理与过滤：中文允许较短阈值，英文稍长
        cleaned: List[str] = []
        for s in sentences:
            s_clean = re.sub(r"\s+", " ", s).strip()
            if not s_clean:
                continue
            # 若主要为中文或混合中文，允许更短的最小长度
            min_len = 6 if contains_cjk else 10
            if len(s_clean) >= min_len:
                cleaned.append(s_clean)

        return cleaned
    
    async def _semantic_embedding_chunking(
        self,
        content: str,
        global_map: GlobalAnalysisMap,
        target_chunk_size: int,
        similarity_threshold: float
    ) -> List[SemanticChunk]:
        """纯语义嵌入切分策略"""
        
        # 将内容分割成较小的文本单元
        text_units = self._split_into_units(content, max_unit_size=100)
        
        if not text_units:
            return []
        
        # 生成嵌入向量
        embeddings = await self.embedding_service.embed_texts(text_units)
        
        # 计算相似度矩阵
        similarity_matrix = self._cosine_similarity_matrix(embeddings)
        
        # 使用滑动窗口方法确定切分点
        chunk_boundaries = self._find_semantic_boundaries(
            similarity_matrix, similarity_threshold
        )
        
        # 生成语义意群（使用累计长度精确估算偏移，而非固定倍数）
        chunks = []
        # 预计算各单元的字符长度（含单词间空格近似）
        unit_lengths: List[int] = []
        for u in text_units:
            try:
                unit_lengths.append(len(u))
            except Exception:
                unit_lengths.append(0)
        # 前缀和，便于区间起止位置计算
        prefix_sum: List[int] = [0]
        for ln in unit_lengths:
            prefix_sum.append(prefix_sum[-1] + ln + 1)  # +1 近似空格/分隔

        for start_idx, end_idx in chunk_boundaries:
            chunk_content = ' '.join(text_units[start_idx:end_idx + 1])
            # 起止位置基于前缀和精确估算（再微调去掉首空格）
            start_pos = max(0, prefix_sum[start_idx])
            end_pos = max(start_pos, prefix_sum[end_idx + 1] - 1)
            chunk = await self._create_semantic_chunk(
                chunk_content,
                start_pos,
                end_pos,
                'semantic_unit',
                global_map
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_units(self, text: str, max_unit_size: int = 100) -> List[str]:
        """将文本分割成较小的单元"""
        words = text.split()
        units = []
        
        current_unit = []
        current_size = 0
        
        for word in words:
            current_unit.append(word)
            current_size += len(word) + 1  # +1 for space
            
            if current_size >= max_unit_size:
                units.append(' '.join(current_unit))
                current_unit = []
                current_size = 0
        
        if current_unit:
            units.append(' '.join(current_unit))
        
        return units
    
    def _find_semantic_boundaries(
        self,
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> List[Tuple[int, int]]:
        """查找语义边界"""
        n = similarity_matrix.shape[0]
        boundaries = []
        current_start = 0
        
        for i in range(1, n):
            # 计算当前单元与前面单元的平均相似度
            avg_similarity = np.mean(similarity_matrix[current_start:i, i])
            
            if avg_similarity < threshold:
                # 找到语义边界
                boundaries.append((current_start, i - 1))
                current_start = i
        
        # 添加最后一个块
        if current_start < n:
            boundaries.append((current_start, n - 1))
        
        return boundaries
    
    async def _paragraph_based_chunking(
        self,
        content: str,
        global_map: GlobalAnalysisMap,
        target_chunk_size: int
    ) -> List[SemanticChunk]:
        """基于段落的切分策略"""
        
        paragraphs = content.split('\n\n')
        chunks = []
        current_pos = 0
        
        current_chunk_content = []
        current_chunk_size = 0
        chunk_start = current_pos
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            if current_chunk_size + para_size <= target_chunk_size:
                # 添加到当前块
                current_chunk_content.append(para)
                current_chunk_size += para_size
            else:
                # 生成当前块
                if current_chunk_content:
                    chunk_content = '\n\n'.join(current_chunk_content)
                    chunk = await self._create_semantic_chunk(
                        chunk_content,
                        chunk_start,
                        current_pos,
                        'paragraph_group',
                        global_map
                    )
                    chunks.append(chunk)
                
                # 开始新块
                current_chunk_content = [para]
                current_chunk_size = para_size
                chunk_start = current_pos
            
            current_pos += para_size + 2  # +2 for \n\n
        
        # 处理最后一个块
        if current_chunk_content:
            chunk_content = '\n\n'.join(current_chunk_content)
            chunk = await self._create_semantic_chunk(
                chunk_content,
                chunk_start,
                current_pos,
                'paragraph_group',
                global_map
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _create_semantic_chunk(
        self,
        content: str,
        start_pos: int,
        end_pos: int,
        section_type: str,
        global_map: GlobalAnalysisMap
    ) -> SemanticChunk:
        """创建语义意群"""
        
        chunk_id = str(uuid.uuid4())
        
        # 生成嵌入向量（失败不致命，使用空向量）
        try:
            embedding = await self.embedding_service.embed_text(content)
        except Exception as _e:
            self.logger.warning(f"Embedding generation failed for chunk {chunk_id}, fallback to empty vector: {_e}")
            embedding = []
        
        # 分析内容特征
        mathematical_content = self._contains_mathematical_content(content)
        contains_formulas = self._contains_formulas(content)
        key_concepts = self._extract_key_concepts(content, global_map)
        
        # 评估重要性和复杂度
        importance_score = self._calculate_importance_score(content, global_map)
        complexity_level = self._assess_complexity_level(content)
        
        # 生成语义主题
        semantic_topic = await self._generate_semantic_topic(content, global_map)
        
        return SemanticChunk(
            chunk_id=chunk_id,
            content=content,
            start_position=start_pos,
            end_position=end_pos,
            section_type=section_type,
            semantic_topic=semantic_topic,
            embedding=embedding,
            importance_score=importance_score,
            complexity_level=complexity_level,
            mathematical_content=mathematical_content,
            contains_formulas=contains_formulas,
            key_concepts=key_concepts
        )
    
    def _contains_mathematical_content(self, content: str) -> bool:
        """检测是否包含数学内容"""
        math_indicators = [
            r'\$.*?\$',  # LaTeX math
            r'\\begin\{equation\}',
            r'\\begin\{align\}',
            r'\b(theorem|lemma|proof|corollary)\b',
            r'\b(algorithm|complexity|convergence)\b',
            r'\b(optimization|gradient|derivative)\b',
            r'\b(matrix|vector|tensor)\b',
            r'O\([^)]+\)',  # Big O notation
        ]
        
        for pattern in math_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _contains_formulas(self, content: str) -> bool:
        """检测是否包含公式"""
        formula_patterns = [
            r'\$.*?\$',
            r'\\begin\{equation\}',
            r'\\begin\{align\}',
            r'\\begin\{eqnarray\}',
            r'[=<>±∝∈∉∪∩∀∃∇∂∫∑∏]',
        ]
        
        for pattern in formula_patterns:
            if re.search(pattern, content):
                return True
        
        return False
    
    def _extract_key_concepts(self, content: str, global_map: GlobalAnalysisMap) -> List[str]:
        """提取关键概念"""
        concepts = []
        
        # 从全局地图中的相关概念查找
        for concept in global_map.related_concepts:
            if concept.lower() in content.lower():
                concepts.append(concept)
        
        # 提取术语词汇表中的概念
        for term in global_map.terminology_glossary.keys():
            if term.lower() in content.lower():
                concepts.append(term)
        
        # 简单的技术术语识别
        tech_patterns = [
            r'\b[A-Z][a-z]*[A-Z][A-Za-z]*\b',  # CamelCase
            r'\b[a-z]+-[a-z]+\b',  # hyphenated terms
            r'\b\w+\s+algorithm\b',
            r'\b\w+\s+method\b',
            r'\b\w+\s+model\b',
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            concepts.extend(matches[:3])  # 限制数量
        
        return list(set(concepts))  # 去重
    
    def _calculate_importance_score(self, content: str, global_map: GlobalAnalysisMap) -> float:
        """计算重要性评分"""
        score = 0.5  # 基础分数
        
        # 基于关键词的重要性
        important_keywords = [
            'novel', 'innovative', 'breakthrough', 'significant',
            'important', 'critical', 'key', 'main', 'primary',
            'contribution', 'finding', 'result', 'conclusion'
        ]
        
        content_lower = content.lower()
        for keyword in important_keywords:
            if keyword in content_lower:
                score += 0.1
        
        # 基于与主要贡献的相关性
        for contribution in global_map.key_contributions:
            contribution_words = (contribution or "").lower().split()
            if not contribution_words:
                continue
            overlap = sum(1 for word in contribution_words if word in content_lower)
            score += (overlap / max(1, len(contribution_words))) * 0.2
        
        return min(1.0, score)  # 限制在1.0以内
    
    def _assess_complexity_level(self, content: str) -> int:
        """评估复杂度级别 (1-5)"""
        complexity_indicators = {
            'mathematical': ['equation', 'formula', 'theorem', 'proof'],
            'technical': ['algorithm', 'implementation', 'optimization'],
            'advanced': ['complexity', 'theoretical', 'formal', 'rigorous'],
            'domain_specific': ['neural', 'quantum', 'molecular', 'genomic']
        }
        
        content_lower = content.lower()
        complexity_score = 1
        
        for category, indicators in complexity_indicators.items():
            for indicator in indicators:
                if indicator in content_lower:
                    complexity_score += 1
                    break
        
        return min(5, complexity_score)
    
    async def _generate_semantic_topic(self, content: str, global_map: GlobalAnalysisMap) -> str:
        """生成语义主题：优先使用 LLM 精准生成，失败时回退启发式。"""
        try:
            if self.llm_service:
                prompt = (
                    "你是一位学术编辑。请基于如下论文上下文与片段内容，生成一个高度概括的中文主题短语（8-16字），不含标点，不要解释：\n"
                    f"论文标题: {global_map.title}\n"
                    f"研究领域: {global_map.research_domain}\n"
                    f"关键贡献: {', '.join(global_map.key_contributions[:3])}\n"
                    "片段:\n" + content[:1200]
                )
                response: LLMResponse = await self.llm_service.generate(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=48,
                )
                topic = (response.content or "").strip().splitlines()[0].strip()
                # 规范化主题：去除多余空白和标点
                topic = topic.replace("：", "").replace(":", "").strip()
                if 2 <= len(topic) <= 32:
                    return topic
        except Exception as e:
            self.logger.warning(f"LLM topic generation failed, falling back: {e}")

        # 回退：基于关键概念与特征
        key_concepts = self._extract_key_concepts(content, global_map)
        if key_concepts:
            return f"{key_concepts[0]}与{key_concepts[1] if len(key_concepts) > 1 else '相关讨论'}"
        if len(content) < 200:
            return "内容概述"
        if self._contains_mathematical_content(content):
            return "数学与方法"
        if 'experiment' in content.lower():
            return "实验与结果"
        if 'conclusion' in content.lower():
            return "结论与展望"
        return "技术要点"
    
    async def _compute_chunk_relationships(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """计算意群间的关系"""
        
        if len(chunks) <= 1:
            return chunks
        
        # 计算所有意群间的相似度；若嵌入缺失则跳过关系构建
        embeddings = [chunk.embedding for chunk in chunks]
        if (not embeddings) or any((not isinstance(e, list) or len(e) == 0) for e in embeddings):
            return chunks
        similarity_matrix = self._cosine_similarity_matrix(embeddings)
        
        # 为每个意群找到最相关的其他意群
        for i, chunk in enumerate(chunks):
            related_chunks = []
            
            for j, similarity in enumerate(similarity_matrix[i]):
                if i != j and similarity > 0.7:  # 相似度阈值
                    related_chunks.append(chunks[j].chunk_id)
            
            chunk.related_chunks = related_chunks[:3]  # 限制相关意群数量
        
        return chunks

    def _cosine_similarity_matrix(self, vectors: List[List[float]]) -> np.ndarray:
        """使用 NumPy 计算余弦相似度矩阵，避免对 sklearn.metrics 的硬依赖。"""
        try:
            mat = np.array(vectors, dtype=float)
            if mat.ndim != 2 or mat.shape[0] == 0:
                return np.zeros((0, 0), dtype=float)
            # 归一化
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
            # 相似度 = 归一化向量点积
            return mat @ mat.T
        except Exception:
            # 最保底返回零矩阵
            n = len(vectors) if isinstance(vectors, list) else 0
            return np.zeros((n, n), dtype=float)


class DeepAnalysisCore:
    """论文分析核心引擎"""
    
    def __init__(
        self,
        config: ConfigManager,
        llm_service: LLMProviderService,
        memory_system: HybridMemorySystem,
        embedding_service: EmbeddingProviderService
    ):
        """
        无副作用构造函数 - 严格依赖注入
        
        Args:
            config: 配置管理器
            llm_service: LLM服务
            memory_system: 混合记忆系统
            embedding_service: 嵌入服务
        """
        if not config.is_initialized():
            raise ValueError("ConfigManager must be initialized")
        
        self.config = config
        self.llm_service = llm_service
        self.memory_system = memory_system
        self.embedding_service = embedding_service
        self.logger = get_logger("deep_analysis_core")
        # 学习循环（由上层注入可选，不强依赖）
        self.learning_loop: Optional[LearningLoop] = None
        
        # 初始化组件
        self.template_manager = PromptTemplateManager(
            Path(config.paths.prompt_templates_folder) if hasattr(config, 'paths') else Path("prompts")
        )
        self.semantic_chunker = SemanticChunker(embedding_service, self.llm_service)
        
        # 状态
        self._initialized = False
        self._processing_stats = {
            "total_papers_analyzed": 0,
            "total_chunks_processed": 0,
            "total_processing_time": 0.0,
            "average_quality_score": 0.0
        }
    
    async def initialize(self) -> None:
        """异步初始化"""
        if self._initialized:
            return
        
        try:
            # 确保依赖服务已初始化
            if not self.memory_system.is_initialized():
                await self.memory_system.initialize()
            # 显式初始化模板管理器（避免构造副作用）
            try:
                await self.template_manager.initialize()
            except Exception as e:
                self.logger.warning(f"PromptTemplateManager initialize failed: {e}")
            
            self._initialized = True
            self.logger.info("Deep analysis core initialized successfully")
            
        except Exception as e:
            raise AnalysisError(f"Failed to initialize analysis core: {e}")
    
    async def analyze_paper(self, request: AnalysisRequest) -> AnalysisResult:
        """
        深度分析论文
        
        Args:
            request: 分析请求
            
        Returns:
            分析结果
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        paper_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Starting deep analysis for paper: {request.paper_title}")
            
            # 阶段1：生成全局分析地图
            global_map = await self._generate_global_analysis_map(
                paper_id, request.paper_content, request.paper_title, request.paper_authors
            )
            
            # 阶段2：智能语义切分
            semantic_chunks = await self.semantic_chunker.chunk_document(
                request.paper_content,
                global_map,
                strategy=ChunkingStrategy.HIERARCHICAL_HYBRID
            )
            
            # 阶段3：逐意群深度解释
            chunk_explanations = await self._explain_chunks_recursively(
                semantic_chunks, global_map, request
            )
            
            # 阶段4：生成整体分析结果
            analysis_result = await self._synthesize_analysis_result(
                paper_id, global_map, semantic_chunks, chunk_explanations, request
            )
            
            # 更新统计信息
            total_time = time.time() - start_time
            self._update_processing_stats(total_time, len(semantic_chunks), analysis_result.quality_metrics)
            
            # 存储到记忆系统
            await self._store_analysis_to_memory(analysis_result)

            # 将自评质量写入学习循环（非阻塞）
            try:
                if self.learning_loop:
                    await self.learning_loop.record_self_evaluation_from_analysis(
                        analysis_result,
                        authors=request.paper_authors,
                    )
            except Exception as _e:
                self.logger.warning(f"LearningLoop self-eval failed: {_e}")
            
            self.logger.info(f"Completed paper analysis in {total_time:.2f}s with {len(semantic_chunks)} chunks")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Paper analysis failed: {e}")
            raise AnalysisError(f"Analysis failed: {e}")
    
    async def _generate_global_analysis_map(
        self,
        paper_id: str,
        content: str,
        title: str,
        authors: List[str]
    ) -> GlobalAnalysisMap:
        """生成全局分析地图"""
        
        start_time = time.time()
        
        try:
            # 渲染全局分析提示词
            prompt = self.template_manager.render_template(
                "global_analysis.j2",
                title=title,
                authors=authors,
                content=content[:8000]  # 限制长度以避免token限制
            )
            
            # 调用LLM进行全局分析
            response = await self.llm_service.generate(
                messages=[{"role": "user", "content": prompt}],
                use_insight_model=True,
                temperature=0.3
            )
            
            # 解析响应并构建GlobalAnalysisMap（先剥离可能的代码围栏）
            raw = (response.content or "").strip()
            try:
                if raw.startswith("```"):
                    first_newline = raw.find("\n")
                    last_triple = raw.rfind("```")
                    if first_newline != -1 and last_triple != -1 and last_triple > first_newline:
                        candidate = raw[first_newline + 1:last_triple].strip()
                        if candidate:
                            raw = candidate
            except Exception:
                pass

            global_map = await self._parse_global_analysis_response(
                paper_id, title, authors, raw
            )
            
            duration = time.time() - start_time
            log_performance("global_analysis_generation", duration)
            
            return global_map
            
        except Exception as e:
            raise GlobalAnalysisError(f"Global analysis generation failed: {e}")
    
    async def _parse_global_analysis_response(
        self,
        paper_id: str,
        title: str,
        authors: List[str],
        response_content: str
    ) -> GlobalAnalysisMap:
        """解析全局分析响应"""
        
        try:
            # 尝试解析JSON响应
            if response_content.strip().startswith('{'):
                analysis_data = json.loads(response_content)
            else:
                # 如果不是JSON，尝试从文本中提取信息
                analysis_data = self._extract_analysis_from_text(response_content)
            
            return GlobalAnalysisMap(
                paper_id=paper_id,
                title=title,
                authors=authors,
                abstract_summary=analysis_data.get('abstract_summary', ''),
                research_domain=analysis_data.get('research_domain', ''),
                key_contributions=analysis_data.get('key_contributions', []),
                methodology_overview=analysis_data.get('methodology_overview', ''),
                main_findings=analysis_data.get('main_findings', []),
                limitations=analysis_data.get('limitations', []),
                future_directions=analysis_data.get('future_directions', []),
                technical_complexity=analysis_data.get('technical_complexity', 0.5),
                novelty_score=analysis_data.get('novelty_score', 0.5),
                practical_relevance=analysis_data.get('practical_relevance', 0.5),
                related_concepts=analysis_data.get('related_concepts', []),
                terminology_glossary=analysis_data.get('terminology_glossary', {}),
                mathematical_concepts=analysis_data.get('mathematical_concepts', [])
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse analysis response, using fallback: {e}")
            return self._create_fallback_global_map(paper_id, title, authors, response_content)
    
    def _extract_analysis_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取分析信息"""
        analysis_data = {}
        
        # 使用正则表达式提取各个部分
        patterns = {
            'research_domain': r'研究领域[：:]\s*([^\n]+)',
            'abstract_summary': r'摘要总结[：:]\s*([^\n]+)',
            'technical_complexity': r'技术复杂度[：:]\s*([0-9.]+)',
            'novelty_score': r'新颖性评分[：:]\s*([0-9.]+)',
            'practical_relevance': r'实用性评分[：:]\s*([0-9.]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip()
                if key in ['technical_complexity', 'novelty_score', 'practical_relevance']:
                    try:
                        analysis_data[key] = float(value)
                    except ValueError:
                        analysis_data[key] = 0.5
                else:
                    analysis_data[key] = value
        
        # 提取列表项
        list_patterns = {
            'key_contributions': r'主要贡献[：:]([^#]*?)(?=\n\s*[#\w]|$)',
            'main_findings': r'主要发现[：:]([^#]*?)(?=\n\s*[#\w]|$)',
            'limitations': r'局限性[：:]([^#]*?)(?=\n\s*[#\w]|$)',
        }
        
        for key, pattern in list_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1)
                items = [item.strip('- ').strip() for item in content.split('\n') if item.strip()]
                analysis_data[key] = items[:5]  # 限制数量
        
        return analysis_data
    
    def _create_fallback_global_map(
        self,
        paper_id: str,
        title: str,
        authors: List[str],
        content: str
    ) -> GlobalAnalysisMap:
        """创建后备全局地图"""
        return GlobalAnalysisMap(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract_summary=f"对论文《{title}》的分析",
            research_domain="计算机科学/人工智能",
            key_contributions=["提出了新的方法", "验证了理论假设"],
            methodology_overview="采用实验和理论分析相结合的方法",
            main_findings=["获得了显著的性能提升", "验证了方法的有效性"],
            limitations=["需要更多的实验验证", "计算复杂度较高"],
            future_directions=["扩展到更多应用场景", "优化算法效率"],
            technical_complexity=0.7,
            novelty_score=0.6,
            practical_relevance=0.5,
            related_concepts=["机器学习", "深度学习", "算法优化"],
            terminology_glossary={},
            mathematical_concepts=["梯度下降", "损失函数", "优化算法"]
        )
    
    async def _explain_chunks_recursively(
        self,
        chunks: List[SemanticChunk],
        global_map: GlobalAnalysisMap,
        request: AnalysisRequest
    ) -> List[ChunkExplanation]:
        """逐意群递归解释"""
        
        explanations = []
        
        # 并行处理多个意群，但控制并发数量
        # 使用配置的并发上限，避免硬编码
        max_conc = 3
        try:
            max_conc = int(self.config.performance.max_concurrent_llm_calls)
        except Exception:
            max_conc = 3
        semaphore = asyncio.Semaphore(max(1, max_conc))
        
        async def explain_chunk(chunk: SemanticChunk) -> ChunkExplanation:
            async with semaphore:
                return await self._explain_single_chunk(chunk, global_map, request)
        
        # 创建所有任务
        tasks = [explain_chunk(chunk) for chunk in chunks]
        
        # 执行任务
        explanations = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        valid_explanations = []
        for i, result in enumerate(explanations):
            if isinstance(result, Exception):
                self.logger.warning(f"Failed to explain chunk {chunks[i].chunk_id}: {result}")
                # 创建后备解释
                fallback_explanation = self._create_fallback_explanation(chunks[i])
                valid_explanations.append(fallback_explanation)
            else:
                valid_explanations.append(result)
        
        return valid_explanations
    
    async def _explain_single_chunk(
        self,
        chunk: SemanticChunk,
        global_map: GlobalAnalysisMap,
        request: AnalysisRequest
    ) -> ChunkExplanation:
        """解释单个意群"""
        
        start_time = time.time()
        
        try:
            # 从记忆系统检索相关上下文
            memory_context = await self._retrieve_memory_context(chunk, global_map)

            # 将记忆上下文序列化为可JSON的轻量结构，避免模板序列化失败
            def _serialize_memory_item(item: Any) -> Dict[str, Any]:
                try:
                    # 期望 item 为 MemorySearchResult
                    mem_type = None
                    try:
                        mem_type = getattr(item, 'memory_type', None)
                        mem_type = getattr(mem_type, 'value', str(mem_type)) if mem_type is not None else None
                    except Exception:
                        mem_type = None

                    sim = None
                    try:
                        sim = float(getattr(item, 'similarity_score', 0.0))
                    except Exception:
                        sim = 0.0

                    entry = getattr(item, 'entry', None)
                    name = None
                    content = None
                    try:
                        if entry is not None:
                            name = getattr(entry, 'name', None)
                            content = getattr(entry, 'content', None)
                    except Exception:
                        pass

                    return {
                        'memory_type': mem_type,
                        'similarity_score': sim,
                        'name': name if isinstance(name, str) else None,
                        'content': content if isinstance(content, str) else None,
                    }
                except Exception:
                    return {'memory_type': None, 'similarity_score': 0.0, 'name': None, 'content': None}

            safe_memory_context: List[Dict[str, Any]] = []
            try:
                for it in memory_context or []:
                    safe_memory_context.append(_serialize_memory_item(it))
            except Exception:
                safe_memory_context = []
            
            # 使用结构化 JSON 模板，避免伪实现与解析脆弱性
            json_prompt = self.template_manager.render_template(
                "chunk_explanation_json.j2",
                global_context=global_map.__dict__,
                memory_context=safe_memory_context,
                chunk=chunk.__dict__,
                output_language=request.output_language
            )

            response = await self.llm_service.generate(
                messages=[{"role": "user", "content": json_prompt}],
                use_insight_model=True,
                temperature=0.3,
                max_tokens=1200
            )

            explanation = await self._parse_chunk_explanation_json(chunk, response, start_time)
            return explanation
            
        except Exception as e:
            raise ExplanationError(f"Failed to explain chunk {chunk.chunk_id}: {e}")
    
    async def _retrieve_memory_context(
        self,
        chunk: SemanticChunk,
        global_map: GlobalAnalysisMap
    ) -> List[Any]:
        """检索记忆上下文"""
        
        try:
            # 构建查询字符串
            query_parts = [chunk.semantic_topic]
            query_parts.extend(chunk.key_concepts[:3])
            query_parts.extend(global_map.related_concepts[:3])
            
            query = " ".join(query_parts)
            
            # 检索相关记忆
            memory_results = await self.memory_system.search_memory(
                query=query,
                memory_types=[MemoryType.EPISODIC, MemoryType.CONCEPTUAL],
                limit=5,
                similarity_threshold=0.6
            )
            
            return memory_results
            
        except Exception as e:
            self.logger.warning(f"Failed to retrieve memory context: {e}")
            return []
    
    async def _parse_chunk_explanation_json(
        self,
        chunk: SemanticChunk,
        response: LLMResponse,
        start_time: float
    ) -> ChunkExplanation:
        """解析结构化 JSON 意群解释结果（严格实现，无伪实现）。"""
        duration = time.time() - start_time
        raw = (response.content or "").strip()

        # 若响应包含 Markdown 代码块围栏，先剥离，提升 JSON 解析稳健性
        try:
            if raw.startswith("```"):
                # 形如 ```json\n{...}\n``` 或 ```\n{...}\n```
                first_newline = raw.find("\n")
                last_triple = raw.rfind("```")
                if first_newline != -1 and last_triple != -1 and last_triple > first_newline:
                    candidate = raw[first_newline + 1:last_triple].strip()
                    if candidate:
                        raw = candidate
        except Exception:
            pass

        data: Dict[str, Any]
        try:
            # 尝试严格解析 JSON
            data = json.loads(raw)
        except Exception as e:
            # 回退：尝试提取首尾花括号内的 JSON 片段
            try:
                start_idx = raw.find('{')
                end_idx = raw.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    data = json.loads(raw[start_idx:end_idx+1])
                else:
                    raise
            except Exception:
                raise ExplanationError(f"Invalid JSON from LLM for chunk {chunk.chunk_id}: {e}")

        # 字段校验
        def require(key: str, expected_types):
            if key not in data:
                raise ExplanationError(f"Missing field '{key}' in JSON explanation")
            value = data[key]
            # 数值字段宽容：若传入 expected_types 含 float/int 但 value 为字符串，可尝试转为 float
            if not isinstance(value, expected_types):
                # 如果期望包含数值类型且当前是字符串，尝试转换
                if (float in (expected_types if isinstance(expected_types, tuple) else (expected_types,)) or int in (expected_types if isinstance(expected_types, tuple) else (expected_types,))) and isinstance(value, str):
                    try:
                        value = float(value.strip())
                        data[key] = value  # 回写规范化的值
                    except Exception:
                        raise ExplanationError(f"Field '{key}' has incorrect type and cannot be coerced to number")
                else:
                    raise ExplanationError(f"Field '{key}' has incorrect type, expected {expected_types}")
            return value

        explanation_text = require('explanation', str)
        if isinstance(explanation_text, str):
            explanation_text = explanation_text.strip()
        key_insights = require('key_insights', list)
        terminology_explained = require('terminology_explained', dict)
        math_explanations = require('mathematical_explanations', dict)
        confidence_score_any = require('confidence_score', (int, float, str))
        try:
            confidence_score = float(confidence_score_any)
        except Exception:
            raise ExplanationError("'confidence_score' is not a valid number")

        # 合法范围与内容约束
        if not explanation_text:
            raise ExplanationError("'explanation' must be non-empty")
        if not (0.0 <= confidence_score <= 1.0):
            raise ExplanationError("'confidence_score' must be within [0,1]")
        # 规范化 key_insights
        key_insights_norm: List[str] = []
        for item in key_insights:
            if isinstance(item, str) and item.strip():
                key_insights_norm.append(item.strip()[:400])
        if not key_insights_norm:
            key_insights_norm = self._extract_key_insights(explanation_text)

        return ChunkExplanation(
            chunk_id=chunk.chunk_id,
            explanation=explanation_text,
            key_insights=key_insights_norm[:8],
            terminology_explained={str(k): str(v) for k, v in terminology_explained.items()},
            mathematical_explanations={str(k): str(v) for k, v in math_explanations.items()},
            connections_to_other_chunks=chunk.related_chunks,
            confidence_score=confidence_score,
            processing_time=duration,
            llm_model_used=response.model
        )
    
    def _extract_key_insights(self, content: str) -> List[str]:
        """提取关键洞察"""
        insights = []
        
        # 查找洞察关键词
        insight_patterns = [
            r'关键洞察[：:]([^\n]+)',
            r'重要发现[：:]([^\n]+)',
            r'核心观点[：:]([^\n]+)',
        ]
        
        for pattern in insight_patterns:
            matches = re.findall(pattern, content)
            insights.extend(matches)
        
        # 如果没有找到明确的洞察，从内容中提取重要句子
        if not insights:
            sentences = content.split('。')
            for sentence in sentences[:3]:  # 取前3句
                if any(keyword in sentence for keyword in ['重要', '关键', '核心', '主要']):
                    insights.append(sentence.strip())
        
        return insights[:5]  # 限制数量
    
    def _extract_terminology(self, content: str) -> Dict[str, str]:
        """提取术语解释"""
        terminology = {}
        
        # 查找术语解释模式
        term_patterns = [
            r'(\w+)[：:]([^。\n]+)',
            r'术语\s*(\w+)\s*[：:]([^。\n]+)',
        ]
        
        for pattern in term_patterns:
            matches = re.findall(pattern, content)
            for term, explanation in matches:
                if len(explanation.strip()) > 10:  # 过滤过短的解释
                    terminology[term] = explanation.strip()
        
        return terminology
    
    def _extract_math_explanations(self, content: str) -> Dict[str, str]:
        """提取数学解释"""
        math_explanations = {}
        
        # 查找数学概念和公式的解释
        math_patterns = [
            r'公式\s*([^：:]+)[：:]([^。\n]+)',
            r'算法\s*([^：:]+)[：:]([^。\n]+)',
            r'定理\s*([^：:]+)[：:]([^。\n]+)',
        ]
        
        for pattern in math_patterns:
            matches = re.findall(pattern, content)
            for concept, explanation in matches:
                math_explanations[concept.strip()] = explanation.strip()
        
        return math_explanations
    
    def _create_fallback_explanation(self, chunk: SemanticChunk) -> ChunkExplanation:
        """创建后备解释"""
        return ChunkExplanation(
            chunk_id=chunk.chunk_id,
            explanation=f"对意群内容的基本描述：{chunk.content[:200]}...",
            key_insights=["需要进一步分析"],
            terminology_explained={},
            mathematical_explanations={},
            connections_to_other_chunks=chunk.related_chunks,
            confidence_score=0.3,
            processing_time=0.1,
            llm_model_used="fallback"
        )
    
    async def _synthesize_analysis_result(
        self,
        paper_id: str,
        global_map: GlobalAnalysisMap,
        chunks: List[SemanticChunk],
        explanations: List[ChunkExplanation],
        request: AnalysisRequest
    ) -> AnalysisResult:
        """综合分析结果"""
        
        # 生成整体总结
        overall_summary = await self._generate_overall_summary(
            global_map, explanations, request
        )
        
        # 提取研究洞察
        research_insights = self._extract_research_insights(explanations)
        
        # 分析实践意义
        practical_implications = self._analyze_practical_implications(global_map, explanations)
        
        # 建立相关工作联系
        related_work_connections = await self._find_related_work_connections(global_map)
        
        # 计算质量指标
        quality_metrics = self._calculate_quality_metrics(explanations)
        
        return AnalysisResult(
            paper_id=paper_id,
            global_map=global_map,
            semantic_chunks=chunks,
            chunk_explanations=explanations,
            overall_summary=overall_summary,
            research_insights=research_insights,
            practical_implications=practical_implications,
            related_work_connections=related_work_connections,
            total_processing_time=sum(exp.processing_time for exp in explanations),
            quality_metrics=quality_metrics
        )
    
    async def _generate_overall_summary(
        self,
        global_map: GlobalAnalysisMap,
        explanations: List[ChunkExplanation],
        request: AnalysisRequest
    ) -> str:
        """生成整体总结"""
        
        try:
            # 收集所有解释的关键点
            all_insights = []
            for explanation in explanations:
                all_insights.extend(explanation.key_insights)
            
            # 构建总结提示词
            summary_prompt = f"""
基于对论文《{global_map.title}》的深度分析，请生成一份全面的总结报告。

## 论文基本信息
- 标题：{global_map.title}
- 作者：{', '.join(global_map.authors)}
- 研究领域：{global_map.research_domain}

## 主要贡献
{chr(10).join('- ' + contrib for contrib in global_map.key_contributions)}

## 关键洞察
{chr(10).join('- ' + insight for insight in all_insights[:10])}

请用{request.output_language}生成一份结构清晰、观点深刻的总结报告，包括：
1. 研究背景和意义
2. 主要方法和创新点
3. 关键发现和结果
4. 研究价值和影响
5. 局限性和未来展望

确保总结既专业严谨又易于理解。
"""
            
            response = await self.llm_service.generate(
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.5
            )
            
            return response.content
            
        except Exception as e:
            self.logger.warning(f"Failed to generate overall summary: {e}")
            return f"对论文《{global_map.title}》的分析显示了其在{global_map.research_domain}领域的重要贡献。"
    
    def _extract_research_insights(self, explanations: List[ChunkExplanation]) -> List[str]:
        """提取研究洞察"""
        insights = []
        
        for explanation in explanations:
            insights.extend(explanation.key_insights)
        
        # 去重并排序
        unique_insights = list(set(insights))
        return unique_insights[:10]  # 返回前10个最重要的洞察
    
    def _analyze_practical_implications(
        self,
        global_map: GlobalAnalysisMap,
        explanations: List[ChunkExplanation]
    ) -> List[str]:
        """分析实践意义"""
        implications = []
        
        # 基于全局地图的实践相关性
        if global_map.practical_relevance > 0.7:
            implications.append("该研究具有很高的实际应用价值")
        
        # 从解释中提取实践相关内容
        practical_keywords = ['应用', '实践', '实际', '部署', '实现', '工程']
        
        for explanation in explanations:
            content = explanation.explanation.lower()
            if any(keyword in content for keyword in practical_keywords):
                # 提取包含实践关键词的句子
                sentences = explanation.explanation.split('。')
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in practical_keywords):
                        implications.append(sentence.strip())
                        break
        
        return implications[:5]  # 限制数量
    
    async def _find_related_work_connections(self, global_map: GlobalAnalysisMap) -> List[str]:
        """查找相关工作连接"""
        connections = []
        
        try:
            # 基于关键概念搜索相关工作
            for concept in global_map.related_concepts[:5]:
                memory_results = await self.memory_system.search_memory(
                    query=concept,
                    memory_types=[MemoryType.EPISODIC],
                    limit=2,
                    similarity_threshold=0.7
                )
                
                for result in memory_results:
                    connections.append(f"与{concept}相关的研究：{result.entry.content[:100]}...")
            
        except Exception as e:
            self.logger.warning(f"Failed to find related work connections: {e}")
        
        return connections[:8]  # 限制数量
    
    def _calculate_quality_metrics(self, explanations: List[ChunkExplanation]) -> Dict[str, float]:
        """计算质量指标"""
        if not explanations:
            return {"overall_quality": 0.0, "explanation_completeness": 0.0, "confidence": 0.0}
        
        # 计算平均置信度
        avg_confidence = sum(exp.confidence_score for exp in explanations) / len(explanations)
        
        # 计算解释完整性（基于解释长度和内容丰富性）
        completeness_scores = []
        for exp in explanations:
            score = 0.0
            if len(exp.explanation) > 100:
                score += 0.3
            if exp.key_insights:
                score += 0.3
            if exp.terminology_explained:
                score += 0.2
            if exp.mathematical_explanations:
                score += 0.2
            completeness_scores.append(score)
        
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        
        # 计算整体质量
        overall_quality = (avg_confidence * 0.6 + avg_completeness * 0.4)
        
        return {
            "overall_quality": overall_quality,
            "explanation_completeness": avg_completeness,
            "confidence": avg_confidence,
            "processed_chunks": len(explanations)
        }
    
    async def _store_analysis_to_memory(self, result: AnalysisResult) -> None:
        """将分析结果存储到记忆系统"""
        
        try:
            # 存储全局分析地图
            await self.memory_system.store_episodic_memory(
                content=f"论文分析：{result.global_map.title} - {result.global_map.abstract_summary}",
                source="paper_analysis",
                context=f"论文ID: {result.paper_id}",
                tags=["paper_analysis", "global_map"] + result.global_map.related_concepts[:3],
                metadata={
                    "paper_id": str(result.paper_id or ""),
                    "research_domain": str(result.global_map.research_domain or ""),
                    "novelty_score": float(result.global_map.novelty_score or 0.0),
                    "technical_complexity": float(result.global_map.technical_complexity or 0.0)
                },
                importance_score=result.global_map.novelty_score
            )
            
            # 存储关键概念到概念记忆
            from .memory_system import NodeType
            for concept in result.global_map.related_concepts[:5]:
                await self.memory_system.store_conceptual_memory(
                    name=concept,
                    node_type=NodeType.CONCEPT,
                    attributes={
                        "domain": result.global_map.research_domain,
                        "paper_title": result.global_map.title,
                        "paper_id": result.paper_id
                    }
                )
            
            # 存储重要的意群解释
            for explanation in result.chunk_explanations:
                if explanation.confidence_score > 0.7:
                    await self.memory_system.store_episodic_memory(
                        content=explanation.explanation,
                        source="chunk_explanation",
                        context=f"论文: {result.global_map.title}, 意群: {explanation.chunk_id}",
                        tags=["chunk_explanation"] + list(explanation.terminology_explained.keys())[:3],
                        metadata={
                            "paper_id": str(result.paper_id or ""),
                            "chunk_id": str(explanation.chunk_id or ""),
                            "confidence_score": float(explanation.confidence_score or 0.0)
                        },
                        importance_score=explanation.confidence_score
                    )
            
            self.logger.info(f"Stored analysis result to memory: {result.paper_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to store analysis to memory: {e}")
    
    def _update_processing_stats(
        self,
        processing_time: float,
        chunks_count: int,
        quality_metrics: Dict[str, float]
    ) -> None:
        """更新处理统计信息"""
        self._processing_stats["total_papers_analyzed"] += 1
        self._processing_stats["total_chunks_processed"] += chunks_count
        self._processing_stats["total_processing_time"] += processing_time
        
        # 更新平均质量分数
        current_avg = self._processing_stats["average_quality_score"]
        total_papers = self._processing_stats["total_papers_analyzed"]
        new_quality = quality_metrics.get("overall_quality", 0.0)
        
        self._processing_stats["average_quality_score"] = (
            (current_avg * (total_papers - 1) + new_quality) / total_papers
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self._processing_stats.copy()
        
        if stats["total_papers_analyzed"] > 0:
            stats["average_processing_time"] = (
                stats["total_processing_time"] / stats["total_papers_analyzed"]
            )
            stats["average_chunks_per_paper"] = (
                stats["total_chunks_processed"] / stats["total_papers_analyzed"]
            )
        
        return stats
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    async def close(self) -> None:
        """关闭分析核心"""
        self._initialized = False
        self.logger.info("Deep analysis core closed")
