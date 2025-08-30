"""
高精度多模态PDF摄取管道模块

严格遵循无副作用构造函数和依赖注入原则，实现SOTA级PDF解析：
- 首选Nougat神经网络解析，备用PyMuPDF+版面分析
- 真正的混合解析策略，智能合并多种解析结果
- 无损图像提取，完整元数据保留
- 健壮的异步架构，支持进度报告和故障恢复
"""

import asyncio
import aiohttp
import aiofiles
import hashlib
import tempfile
import time
import io
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncGenerator, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import re
from urllib.parse import urlparse, urljoin, unquote
from datetime import datetime

# PDF处理库
import fitz  # PyMuPDF
import pymupdf4llm
from PIL import Image
import numpy as np

# ML/AI库（对所有异常进行兜底，避免环境不兼容导致导入期崩溃）
try:
    import torch
    from transformers import NougatProcessor, VisionEncoderDecoderModel
    NOUGAT_AVAILABLE = True
except Exception:
    NOUGAT_AVAILABLE = False

# 版面分析库
try:
    import layoutparser as lp
    LAYOUT_PARSER_AVAILABLE = True
except ImportError:
    LAYOUT_PARSER_AVAILABLE = False

# 配置和日志
from .config_manager import ConfigManager
from .logger import get_logger, log_paper_processing, log_performance, LoggingContextManager


class ProcessingStage(Enum):
    """处理阶段枚举"""
    DOWNLOAD = "download"
    PARSE_NOUGAT = "parse_nougat"
    PARSE_PYMUPDF = "parse_pymupdf"
    LAYOUT_ANALYSIS = "layout_analysis"
    IMAGE_EXTRACTION = "image_extraction"
    METADATA_EXTRACTION = "metadata_extraction"
    CONTENT_MERGE = "content_merge"
    VALIDATION = "validation"
    COMPLETE = "complete"


class ParseQuality(Enum):
    """解析质量枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class PaperMetadata:
    """论文元数据结构"""
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    publication_date: Optional[datetime] = None
    journal: Optional[str] = None
    conference: Optional[str] = None
    pages: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    url: Optional[str] = None
    file_hash: Optional[str] = None
    file_size: int = 0
    page_count: int = 0
    language: str = "en"
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    producer: Optional[str] = None
    creator: Optional[str] = None


@dataclass
class ImageData:
    """图像数据结构"""
    image_id: str
    page_number: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    image_path: str
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    figure_number: Optional[str] = None
    image_type: str = "figure"  # figure, table, equation, diagram
    width: int = 0
    height: int = 0
    format: str = "PNG"
    file_size: int = 0


@dataclass
class TableData:
    """表格数据结构"""
    table_id: str
    page_number: int
    bbox: Tuple[float, float, float, float]
    caption: Optional[str] = None
    table_number: Optional[str] = None
    data: List[List[str]] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    markdown: Optional[str] = None


@dataclass
class PaperSection:
    """论文章节结构"""
    section_id: str
    title: str
    content: str
    level: int = 1  # 章节层级
    page_numbers: List[int] = field(default_factory=list)
    subsections: List['PaperSection'] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)  # 关联的图像ID
    tables: List[str] = field(default_factory=list)  # 关联的表格ID


@dataclass
class ParseResult:
    """解析结果结构"""
    parser_name: str
    success: bool
    quality: ParseQuality
    content: str
    sections: List[PaperSection] = field(default_factory=list)
    images: List[ImageData] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    metadata: Optional[PaperMetadata] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    confidence_score: float = 0.0


@dataclass
class ProcessedPaper:
    """最终处理结果"""
    paper_id: str
    source_url: str
    metadata: PaperMetadata
    sections: List[PaperSection]
    images: List[ImageData]
    tables: List[TableData]
    full_text: str
    abstract: str
    references: List[str] = field(default_factory=list)
    parse_results: List[ParseResult] = field(default_factory=list)
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProgressReport:
    """进度报告结构"""
    paper_id: str
    stage: ProcessingStage
    progress: float  # 0.0 - 1.0
    message: str
    started_at: datetime
    estimated_remaining: Optional[float] = None
    current_operation: Optional[str] = None


class PDFParsingError(Exception):
    """PDF解析错误"""
    pass


class DownloadError(Exception):
    """下载错误"""
    pass


class ValidationError(Exception):
    """验证错误"""
    pass


class PDFParser(ABC):
    """PDF解析器抽象基类"""
    
    @abstractmethod
    async def parse(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> ParseResult:
        """解析PDF文件"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取解析器名称"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查解析器是否可用"""
        pass


class NougatParser(PDFParser):
    """Nougat神经网络PDF解析器"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self.model_path = model_path
        self.device = self._determine_device(device)
        self.processor: Optional[NougatProcessor] = None
        self.model: Optional[VisionEncoderDecoderModel] = None
        self._initialized = False
        self.logger = get_logger("nougat_parser")
    
    def _determine_device(self, device: str) -> str:
        """确定计算设备"""
        if device == "auto":
            if NOUGAT_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            elif NOUGAT_AVAILABLE and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    async def initialize(self) -> None:
        """异步初始化Nougat模型"""
        if self._initialized or not NOUGAT_AVAILABLE:
            return
        
        try:
            self.logger.info(f"Initializing Nougat model on {self.device}")
            
            # 使用预训练模型或指定路径
            model_name = self.model_path or "facebook/nougat-base"
            
            # 离线优先加载（避免触发网络请求），失败后再回退在线加载
            try:
                self.processor = await asyncio.to_thread(
                    NougatProcessor.from_pretrained, model_name, local_files_only=True
                )
                self.model = await asyncio.to_thread(
                    VisionEncoderDecoderModel.from_pretrained, model_name, local_files_only=True
                )
                self.logger.info("Loaded Nougat locally (local_files_only=True)")
            except Exception as offline_err:
                self.logger.warning(f"Offline Nougat load failed, falling back to online: {offline_err}")
                self.processor = await asyncio.to_thread(
                    NougatProcessor.from_pretrained, model_name
                )
                self.model = await asyncio.to_thread(
                    VisionEncoderDecoderModel.from_pretrained, model_name
                )
            
            # 移动到指定设备
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()  # 设置为评估模式
            self._initialized = True
            
            self.logger.info("Nougat model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Nougat model: {e}")
            raise PDFParsingError(f"Nougat initialization failed: {e}")
    
    def get_name(self) -> str:
        return "Nougat"
    
    def is_available(self) -> bool:
        return NOUGAT_AVAILABLE and self._initialized
    
    async def parse(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> ParseResult:
        """使用Nougat解析PDF"""
        if not self.is_available():
            raise PDFParsingError("Nougat parser not available")
        
        start_time = time.time()
        
        try:
            # 转换PDF为图像
            images = await self._pdf_to_images(pdf_path, progress_callback)
            
            if progress_callback:
                await progress_callback("正在运行Nougat神经网络推理...")
            
            # 批量处理图像
            all_text = []
            total_pages = len(images)
            
            for i, image in enumerate(images):
                page_text = await self._process_page_image(image)
                all_text.append(page_text)
                
                if progress_callback:
                    progress = (i + 1) / total_pages * 0.8 + 0.2  # 20%-100%
                    await progress_callback(f"处理第 {i+1}/{total_pages} 页", progress)
            
            # 合并文本
            full_text = "\n\n".join(all_text)
            
            # 解析章节结构
            sections = await self._extract_sections(full_text)
            
            processing_time = time.time() - start_time
            
            # 评估解析质量
            quality = self._evaluate_quality(full_text, len(images))
            
            return ParseResult(
                parser_name=self.get_name(),
                success=True,
                quality=quality,
                content=full_text,
                sections=sections,
                processing_time=processing_time,
                confidence_score=self._calculate_confidence(full_text)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Nougat parsing failed: {e}")
            
            return ParseResult(
                parser_name=self.get_name(),
                success=False,
                quality=ParseQuality.FAILED,
                content="",
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _pdf_to_images(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> List[Image.Image]:
        """将PDF转换为图像"""
        images = []
        
        def convert_pages():
            doc = fitz.open(pdf_path)
            try:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    # 高分辨率渲染
                    mat = fitz.Matrix(2.0, 2.0)  # 2x缩放
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    images.append(img)
            finally:
                doc.close()
            return images
        
        if progress_callback:
            await progress_callback("正在将PDF转换为图像...")
        
        return await asyncio.to_thread(convert_pages)
    
    async def _process_page_image(self, image: Image.Image) -> str:
        """处理单页图像"""
        def process():
            # 预处理图像
            inputs = self.processor(image, return_tensors="pt")
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=4096)
            
            # 解码结果
            text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return text
        
        return await asyncio.to_thread(process)
    
    async def _extract_sections(self, text: str) -> List[PaperSection]:
        """从文本中提取章节结构"""
        sections = []
        
        # 简单的章节标题模式
        section_patterns = [
            r'^#+\s*(.+)$',  # Markdown标题
            r'^(\d+\.?\s*.+)$',  # 数字编号标题
            r'^([A-Z][A-Z\s]+)$',  # 全大写标题
        ]
        
        lines = text.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是章节标题
            is_section_title = False
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    # 保存前一个章节
                    if current_section and section_content:
                        current_section.content = '\n'.join(section_content)
                        sections.append(current_section)
                    
                    # 开始新章节
                    current_section = PaperSection(
                        section_id=f"section_{len(sections)}",
                        title=match.group(1).strip(),
                        content="",
                        level=1
                    )
                    section_content = []
                    is_section_title = True
                    break
            
            if not is_section_title:
                section_content.append(line)
        
        # 保存最后一个章节
        if current_section and section_content:
            current_section.content = '\n'.join(section_content)
            sections.append(current_section)
        
        return sections
    
    def _evaluate_quality(self, text: str, page_count: int) -> ParseQuality:
        """评估解析质量"""
        if not text or len(text) < 100:
            return ParseQuality.FAILED
        
        # 计算文本质量指标
        avg_chars_per_page = len(text) / page_count if page_count > 0 else 0
        
        # 检查文本结构
        has_sections = bool(re.search(r'^#+\s*\w+|^\d+\.?\s*\w+', text, re.MULTILINE))
        has_references = 'references' in text.lower() or 'bibliography' in text.lower()
        
        # 检查特殊字符比例（可能的OCR错误）
        special_char_ratio = len(re.findall(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', text)) / len(text)
        
        if avg_chars_per_page > 2000 and has_sections and has_references and special_char_ratio < 0.05:
            return ParseQuality.EXCELLENT
        elif avg_chars_per_page > 1500 and (has_sections or has_references) and special_char_ratio < 0.1:
            return ParseQuality.GOOD
        elif avg_chars_per_page > 1000 and special_char_ratio < 0.15:
            return ParseQuality.FAIR
        else:
            return ParseQuality.POOR
    
    def _calculate_confidence(self, text: str) -> float:
        """计算置信度分数"""
        if not text:
            return 0.0
        
        # 基于文本特征计算置信度
        factors = []
        
        # 长度因子
        length_factor = min(len(text) / 10000, 1.0)
        factors.append(length_factor)
        
        # 结构因子
        section_count = len(re.findall(r'^#+\s*\w+|^\d+\.?\s*\w+', text, re.MULTILINE))
        structure_factor = min(section_count / 10, 1.0)
        factors.append(structure_factor)
        
        # 语言质量因子
        word_count = len(text.split())
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        language_factor = min(avg_word_length / 6, 1.0)
        factors.append(language_factor)
        
        return sum(factors) / len(factors)


class PyMuPDFParser(PDFParser):
    """PyMuPDF解析器，结合版面分析"""
    
    def __init__(self):
        self.logger = get_logger("pymupdf_parser")
        self.layout_model = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """异步初始化版面分析模型"""
        if self._initialized:
            return
        
        try:
            if LAYOUT_PARSER_AVAILABLE:
                # 初始化版面分析模型
                self.layout_model = await asyncio.to_thread(
                    lp.Detectron2LayoutModel,
                    'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                )
                self.logger.info("Layout analysis model initialized")
            
            self._initialized = True
            
        except Exception as e:
            self.logger.warning(f"Layout model initialization failed: {e}")
            self._initialized = True  # 继续使用基础功能
    
    def get_name(self) -> str:
        return "PyMuPDF"
    
    def is_available(self) -> bool:
        return True  # PyMuPDF总是可用
    
    async def parse(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> ParseResult:
        """使用PyMuPDF解析PDF"""
        start_time = time.time()
        
        try:
            if progress_callback:
                await progress_callback("正在使用PyMuPDF解析PDF...")
            
            # 提取文本和结构
            doc = fitz.open(pdf_path)
            
            try:
                # 使用pymupdf4llm获取结构化文本
                full_text = await asyncio.to_thread(
                    pymupdf4llm.to_markdown, doc
                )
                
                if progress_callback:
                    await progress_callback("正在分析文档结构...", 0.3)
                
                # 解析章节
                sections = await self._extract_sections_from_markdown(full_text)
                
                if progress_callback:
                    await progress_callback("正在提取图像和表格...", 0.6)
                
                # 提取图像和表格
                images = await self._extract_images(doc)
                tables = await self._extract_tables(doc)
                
                if progress_callback:
                    await progress_callback("正在提取元数据...", 0.9)
                
                # 提取元数据
                metadata = await self._extract_metadata(doc)
                
                processing_time = time.time() - start_time
                quality = self._evaluate_quality(full_text, len(doc))
                
                return ParseResult(
                    parser_name=self.get_name(),
                    success=True,
                    quality=quality,
                    content=full_text,
                    sections=sections,
                    images=images,
                    tables=tables,
                    metadata=metadata,
                    processing_time=processing_time,
                    confidence_score=self._calculate_confidence(full_text)
                )
                
            finally:
                doc.close()
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"PyMuPDF parsing failed: {e}")
            
            return ParseResult(
                parser_name=self.get_name(),
                success=False,
                quality=ParseQuality.FAILED,
                content="",
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _extract_sections_from_markdown(self, markdown_text: str) -> List[PaperSection]:
        """从Markdown文本提取章节"""
        sections = []
        lines = markdown_text.split('\n')
        
        current_section = None
        section_content = []
        
        for line in lines:
            line = line.strip()
            
            # 检查Markdown标题
            if line.startswith('#'):
                # 保存前一个章节
                if current_section and section_content:
                    current_section.content = '\n'.join(section_content)
                    sections.append(current_section)
                
                # 计算标题级别
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                
                # 开始新章节
                current_section = PaperSection(
                    section_id=f"section_{len(sections)}",
                    title=title,
                    content="",
                    level=level
                )
                section_content = []
            else:
                if line:  # 非空行
                    section_content.append(line)
        
        # 保存最后一个章节
        if current_section and section_content:
            current_section.content = '\n'.join(section_content)
            sections.append(current_section)
        
        return sections
    
    async def _extract_images(self, doc: fitz.Document) -> List[ImageData]:
        """提取图像"""
        images = []
        
        def extract():
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # 确保不是CMYK
                        img_data = pix.tobytes("png")
                        
                        # 生成图像ID和路径
                        image_id = f"img_{page_num}_{img_index}"
                        
                        # 获取图像位置信息
                        img_rects = page.get_image_rects(xref)
                        rect = img_rects[0] if img_rects else None
                        if rect is not None and hasattr(rect, 'x0'):
                            bbox = (
                                float(rect.x0),
                                float(rect.y0),
                                float(rect.x1),
                                float(rect.y1),
                            )
                        else:
                            bbox = (0.0, 0.0, float(pix.width), float(pix.height))
                        
                        images.append(ImageData(
                            image_id=image_id,
                            page_number=page_num + 1,
                            bbox=bbox,
                            image_path="",  # 临时路径，后续会设置
                            width=pix.width,
                            height=pix.height,
                            file_size=len(img_data)
                        ))
                    
                    pix = None  # 释放内存
            
            return images
        
        return await asyncio.to_thread(extract)
    
    async def _extract_tables(self, doc: fitz.Document) -> List[TableData]:
        """提取表格"""
        tables = []
        
        def extract():
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 查找表格（PyMuPDF: page.find_tables() 返回 TableFinder，需要使用 .tables）
                try:
                    tf = page.find_tables()
                except Exception:
                    tf = None

                # 兼容不同版本返回：优先使用 TableFinder.tables，其次容忍列表
                page_tables = []
                if tf is not None:
                    if hasattr(tf, "tables") and isinstance(getattr(tf, "tables"), list):
                        page_tables = tf.tables
                    elif isinstance(tf, list):
                        page_tables = tf

                for table_index, table in enumerate(page_tables):
                    table_id = f"table_{page_num}_{table_index}"
                    
                    # 提取表格数据（标准 API: Table.extract() 返回二维单元格矩阵）
                    try:
                        table_matrix = table.extract()
                    except Exception:
                        table_matrix = None

                    headers = []
                    data = []
                    if isinstance(table_matrix, list) and table_matrix:
                        # 将第一行作为表头，其余作为数据
                        headers = table_matrix[0] if isinstance(table_matrix[0], list) else []
                        data = table_matrix[1:] if len(table_matrix) > 1 else []
                    
                    # 转换为Markdown
                    markdown = self._table_to_markdown(headers, data)

                    # 归一化 bbox 为 4 元组
                    tbbox = getattr(table, 'bbox', None)
                    if tbbox is not None and hasattr(tbbox, 'x0'):
                        bbox_tuple = (
                            float(tbbox.x0),
                            float(tbbox.y0),
                            float(tbbox.x1),
                            float(tbbox.y1),
                        )
                    elif isinstance(tbbox, (tuple, list)) and len(tbbox) == 4:
                        bbox_tuple = tuple(float(v) for v in tbbox)
                    else:
                        bbox_tuple = (0.0, 0.0, 0.0, 0.0)
                    
                    tables.append(TableData(
                        table_id=table_id,
                        page_number=page_num + 1,
                        bbox=bbox_tuple,
                        headers=headers,
                        data=data,
                        markdown=markdown
                    ))
            
            return tables
        
        return await asyncio.to_thread(extract)
    
    def _table_to_markdown(self, headers: List[str], data: List[List[str]]) -> str:
        """将表格转换为Markdown格式"""
        if not headers and not data:
            return ""
        
        lines = []
        
        # 添加表头
        if headers:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # 添加数据行
        for row in data:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        return "\n".join(lines)
    
    async def _extract_metadata(self, doc: fitz.Document) -> PaperMetadata:
        """提取PDF元数据"""
        def extract():
            metadata_dict = doc.metadata
            
            return PaperMetadata(
                title=metadata_dict.get('title'),
                authors=self._parse_authors(metadata_dict.get('author', '')),
                creator=metadata_dict.get('creator'),
                producer=metadata_dict.get('producer'),
                page_count=len(doc),
                creation_date=self._parse_date(metadata_dict.get('creationDate')),
                modification_date=self._parse_date(metadata_dict.get('modDate'))
            )
        
        return await asyncio.to_thread(extract)
    
    def _parse_authors(self, author_string: str) -> List[str]:
        """解析作者字符串"""
        if not author_string:
            return []
        
        # 简单的作者分割逻辑
        authors = re.split(r'[,;]|\sand\s', author_string)
        return [author.strip() for author in authors if author.strip()]
    
    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """解析日期字符串"""
        if not date_string:
            return None
        
        # PDF日期格式通常是 D:YYYYMMDDHHmmSSOHH'mm'
        try:
            if date_string.startswith('D:'):
                date_part = date_string[2:16]  # YYYYMMDDHHMMSS
                return datetime.strptime(date_part, '%Y%m%d%H%M%S')
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _evaluate_quality(self, text: str, page_count: int) -> ParseQuality:
        """评估解析质量"""
        if not text or len(text) < 100:
            return ParseQuality.FAILED
        
        # 基于文本特征评估
        avg_chars_per_page = len(text) / page_count if page_count > 0 else 0
        
        # 检查Markdown结构
        has_headings = bool(re.search(r'^#+\s', text, re.MULTILINE))
        has_tables = '|' in text
        has_lists = bool(re.search(r'^[\*\-\+]\s', text, re.MULTILINE))
        
        structure_score = sum([has_headings, has_tables, has_lists]) / 3
        
        if avg_chars_per_page > 1500 and structure_score > 0.6:
            return ParseQuality.EXCELLENT
        elif avg_chars_per_page > 1000 and structure_score > 0.3:
            return ParseQuality.GOOD
        elif avg_chars_per_page > 500:
            return ParseQuality.FAIR
        else:
            return ParseQuality.POOR
    
    def _calculate_confidence(self, text: str) -> float:
        """计算置信度分数"""
        if not text:
            return 0.0
        
        # 基于文本结构计算置信度
        factors = []
        
        # Markdown结构因子
        heading_count = len(re.findall(r'^#+\s', text, re.MULTILINE))
        structure_factor = min(heading_count / 5, 1.0)
        factors.append(structure_factor)
        
        # 文本质量因子
        word_count = len(text.split())
        if word_count > 0:
            avg_word_length = sum(len(word.strip('.,!?;:')) for word in text.split()) / word_count
            quality_factor = min(avg_word_length / 5, 1.0)
            factors.append(quality_factor)
        
        # 长度因子
        length_factor = min(len(text) / 5000, 1.0)
        factors.append(length_factor)
        
        return sum(factors) / len(factors) if factors else 0.0


class HybridParsingStrategy:
    """混合解析策略 - 智能合并多种解析结果"""
    
    def __init__(self):
        self.logger = get_logger("hybrid_parser")
    
    async def merge_results(self, results: List[ParseResult]) -> ParseResult:
        """智能合并多个解析结果"""
        if not results:
            raise ValueError("No parsing results to merge")
        
        # 过滤成功的结果
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            # 返回最好的失败结果
            best_failed = max(results, key=lambda x: x.confidence_score)
            return best_failed
        
        # 按质量和置信度排序
        successful_results.sort(
            key=lambda x: (self._quality_to_score(x.quality), x.confidence_score),
            reverse=True
        )
        
        primary_result = successful_results[0]
        
        # 如果只有一个成功结果，直接返回
        if len(successful_results) == 1:
            return primary_result
        
        # 合并多个结果
        merged_content = await self._merge_content(successful_results)
        merged_sections = await self._merge_sections(successful_results)
        merged_images = self._merge_images(successful_results)
        merged_tables = self._merge_tables(successful_results)
        
        # 计算合并后的质量和置信度
        avg_quality = self._calculate_average_quality(successful_results)
        avg_confidence = sum(r.confidence_score for r in successful_results) / len(successful_results)
        
        return ParseResult(
            parser_name="HybridParser",
            success=True,
            quality=avg_quality,
            content=merged_content,
            sections=merged_sections,
            images=merged_images,
            tables=merged_tables,
            metadata=primary_result.metadata,
            processing_time=sum(r.processing_time for r in successful_results),
            confidence_score=min(avg_confidence * 1.1, 1.0)  # 混合解析的额外加成
        )
    
    async def _merge_content(self, results: List[ParseResult]) -> str:
        """合并文本内容"""
        if not results:
            return ""
        
        # 选择最长且质量最好的文本
        best_content = ""
        best_score = 0
        
        for result in results:
            # 综合考虑长度和质量
            length_score = len(result.content) / 10000  # 归一化长度分数
            quality_score = self._quality_to_score(result.quality)
            
            combined_score = length_score * 0.7 + quality_score * 0.3
            
            if combined_score > best_score:
                best_score = combined_score
                best_content = result.content
        
        return best_content
    
    async def _merge_sections(self, results: List[ParseResult]) -> List[PaperSection]:
        """合并章节信息"""
        if not results:
            return []
        
        # 选择章节最多且结构最好的结果
        best_sections = []
        best_score = 0
        
        for result in results:
            section_count = len(result.sections)
            avg_content_length = (
                sum(len(s.content) for s in result.sections) / section_count
                if section_count > 0 else 0
            )
            
            score = section_count * 0.6 + avg_content_length / 1000 * 0.4
            
            if score > best_score:
                best_score = score
                best_sections = result.sections
        
        return best_sections
    
    def _merge_images(self, results: List[ParseResult]) -> List[ImageData]:
        """合并图像信息"""
        all_images = []
        seen_images = set()
        
        for result in results:
            for image in result.images:
                # 基于位置和尺寸去重
                image_key = (image.page_number, image.bbox, image.width, image.height)
                if image_key not in seen_images:
                    seen_images.add(image_key)
                    all_images.append(image)
        
        return all_images
    
    def _merge_tables(self, results: List[ParseResult]) -> List[TableData]:
        """合并表格信息"""
        all_tables = []
        seen_tables = set()
        
        for result in results:
            for table in result.tables:
                # 基于位置和内容去重
                table_key = (table.page_number, table.bbox, len(table.data))
                if table_key not in seen_tables:
                    seen_tables.add(table_key)
                    all_tables.append(table)
        
        return all_tables
    
    def _quality_to_score(self, quality: ParseQuality) -> float:
        """将质量枚举转换为分数"""
        quality_scores = {
            ParseQuality.EXCELLENT: 1.0,
            ParseQuality.GOOD: 0.8,
            ParseQuality.FAIR: 0.6,
            ParseQuality.POOR: 0.4,
            ParseQuality.FAILED: 0.0
        }
        return quality_scores.get(quality, 0.0)
    
    def _calculate_average_quality(self, results: List[ParseResult]) -> ParseQuality:
        """计算平均质量"""
        if not results:
            return ParseQuality.FAILED
        
        avg_score = sum(self._quality_to_score(r.quality) for r in results) / len(results)
        
        if avg_score >= 0.9:
            return ParseQuality.EXCELLENT
        elif avg_score >= 0.7:
            return ParseQuality.GOOD
        elif avg_score >= 0.5:
            return ParseQuality.FAIR
        elif avg_score >= 0.3:
            return ParseQuality.POOR
        else:
            return ParseQuality.FAILED


class PDFDownloader:
    """异步PDF下载器"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = get_logger("pdf_downloader")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),  # 5分钟超时
            headers={
                'User-Agent': 'DeepScholarAI/1.0 (PDF Research Tool)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def download(
        self,
        url: str,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[str, str]:
        """
        下载PDF文件
        
        Returns:
            Tuple[file_path, file_hash]
        """
        if not self.session:
            raise RuntimeError("Downloader not initialized")
        
        try:
            # 创建临时文件目录
            temp_dir = Path(self.config.paths.temp_downloads_folder)
            temp_dir.mkdir(parents=True, exist_ok=True)

            parsed = urlparse(url)

            # 分支1：本地文件（Windows 盘符/UNC、file://、或直接路径）
            is_windows_drive_path = bool(re.match(r'^[A-Za-z]:[\\/]', url))
            is_unc_path = url.startswith('\\\\') or url.startswith('//')
            is_plain_local = parsed.scheme == '' and not parsed.netloc
            is_file_scheme = parsed.scheme == 'file'

            if is_windows_drive_path or is_unc_path or is_plain_local or is_file_scheme:
                # 解析本地路径
                local_path = None
                if is_file_scheme:
                    # 处理 file:// URI（包含可能的盘符 netloc）
                    if parsed.netloc and re.match(r'^[A-Za-z]:$', parsed.netloc):
                        combined = f"{parsed.netloc}{parsed.path}"
                    else:
                        combined = parsed.path
                    # 去掉形如 /D:/ 前导斜杠并反解码
                    combined = unquote(combined)
                    if re.match(r'^/[A-Za-z]:', combined):
                        combined = combined.lstrip('/')
                    local_path = Path(combined)
                else:
                    # 处理直接传入的本地路径（相对或绝对，或 UNC）
                    local_path = Path(url)

                if not local_path.is_absolute():
                    # 解析为绝对路径
                    try:
                        local_path = Path(local_path).resolve()
                    except Exception:
                        pass

                if not local_path.exists() or not local_path.is_file():
                    raise DownloadError(f"Local PDF not found: {local_path}")

                # 校验PDF魔数
                async with aiofiles.open(local_path, 'rb') as f:
                    header = await f.read(8)
                # 放宽校验：允许 BOM/空白后再出现 %PDF-
                try:
                    magic_ok = (b"%PDF-" in header)
                except Exception:
                    magic_ok = False
                if not magic_ok:
                    raise DownloadError("Local file is not a valid PDF (magic header not found)")

                # 计算哈希
                hash_sha256 = hashlib.sha256()
                async with aiofiles.open(local_path, 'rb') as f:
                    while True:
                        chunk = await f.read(8192)
                        if not chunk:
                            break
                        hash_sha256.update(chunk)
                file_hash = hash_sha256.hexdigest()

                # 拷贝到临时目录，保持统一的后续处理流程
                import shutil
                dest_name = local_path.name if local_path.suffix.lower() == '.pdf' else f"{file_hash}.pdf"
                dest_path = temp_dir / dest_name
                if dest_path.exists():
                    # 已存在则复用
                    self.logger.info(f"Using existing cached local PDF: {dest_path}")
                else:
                    await asyncio.to_thread(shutil.copyfile, str(local_path), str(dest_path))

                if progress_callback:
                    await progress_callback("已读取本地PDF", 1.0)

                self.logger.info(f"Loaded local PDF: {local_path} -> {dest_path} (hash: {file_hash})")
                return str(dest_path), file_hash

            # 分支2：HTTP(S) 下载
            # 生成文件名
            filename = Path(parsed.path).name
            if not filename.endswith('.pdf'):
                filename = f"{hashlib.md5(url.encode()).hexdigest()}.pdf"
            file_path = temp_dir / filename

            async with self.session.get(url) as response:
                response.raise_for_status()

                # 提前缓存响应头，避免退出上下文后访问已关闭响应对象
                cached_content_type = response.headers.get('content-type', '')
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                hash_sha256 = hashlib.sha256()

                async with aiofiles.open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                        hash_sha256.update(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size > 0:
                            progress = downloaded / total_size
                            await progress_callback(f"已下载 {downloaded}/{total_size} 字节", progress)

            file_hash = hash_sha256.hexdigest()

            # 基础内容校验：Content-Type 或 PDF 魔数
            try:
                if 'pdf' not in cached_content_type.lower():
                    async with aiofiles.open(file_path, 'rb') as f:
                        header = await f.read(8)
                    if b"%PDF-" not in header:
                        raise DownloadError(f"Downloaded content is not a valid PDF (content-type={cached_content_type}).")
            except Exception as e:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass
                raise DownloadError(str(e))

            self.logger.info(f"Downloaded PDF: {url} -> {file_path} (hash: {file_hash})")
            return str(file_path), file_hash

        except Exception as e:
            self.logger.error(f"Failed to download PDF from {url}: {e}")
            raise DownloadError(f"Download failed: {e}")


class IngestionPipeline:
    """
    高精度多模态PDF摄取管道
    
    严格遵循无副作用构造函数原则，所有初始化通过initialize()方法完成
    """
    
    def __init__(self):
        """
        无副作用构造函数
        
        仅初始化空的组件容器，不执行任何I/O操作
        """
        self.config: Optional[ConfigManager] = None
        self.logger = get_logger("ingestion_pipeline")
        
        # 解析器组件
        self.nougat_parser: Optional[NougatParser] = None
        self.pymupdf_parser: Optional[PyMuPDFParser] = None
        self.hybrid_strategy: Optional[HybridParsingStrategy] = None
        
        # 组件状态
        self._initialized = False
        self._current_operations: Dict[str, asyncio.Task] = {}
    
    async def initialize(self, config: ConfigManager) -> None:
        """
        异步初始化摄取管道
        
        Args:
            config: 配置管理器实例
            
        Raises:
            ValueError: 配置无效
            RuntimeError: 组件初始化失败
        """
        if self._initialized:
            return
        
        if not config.is_initialized():
            raise ValueError("ConfigManager must be initialized")
        
        self.config = config
        
        try:
            self.logger.info("Initializing PDF ingestion pipeline")
            
            # 初始化解析器
            await self._initialize_parsers()
            
            # 初始化混合策略
            self.hybrid_strategy = HybridParsingStrategy()
            
            self._initialized = True
            self.logger.info("PDF ingestion pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ingestion pipeline: {e}")
            raise RuntimeError(f"Pipeline initialization failed: {e}")
    
    async def _initialize_parsers(self) -> None:
        """初始化所有可用的解析器"""
        parsers_info = []
        
        # 初始化Nougat解析器
        if NOUGAT_AVAILABLE:
            try:
                self.nougat_parser = NougatParser(
                    model_path=self.config.pdf_processing.nougat_model_path
                )
                await self.nougat_parser.initialize()
                parsers_info.append("Nougat (Neural)")
            except Exception as e:
                self.logger.warning(f"Nougat parser initialization failed: {e}")
                self.nougat_parser = None
        else:
            self.logger.warning("Nougat dependencies not available")
        
        # 初始化PyMuPDF解析器
        try:
            self.pymupdf_parser = PyMuPDFParser()
            await self.pymupdf_parser.initialize()
            parsers_info.append("PyMuPDF (Traditional)")
        except Exception as e:
            self.logger.error(f"PyMuPDF parser initialization failed: {e}")
            raise RuntimeError("No PDF parsers available")
        
        self.logger.info(f"Initialized parsers: {', '.join(parsers_info)}")
    
    def is_initialized(self) -> bool:
        """检查管道是否已初始化"""
        return self._initialized
    
    async def process_paper(
        self,
        url: str,
        progress_callback: Optional[Callable[[ProgressReport], None]] = None
    ) -> ProcessedPaper:
        """
        处理单篇论文
        
        Args:
            url: 论文PDF的URL
            progress_callback: 进度回调函数
            
        Returns:
            处理完成的论文对象
            
        Raises:
            PDFParsingError: PDF解析失败
            DownloadError: 下载失败
            ValidationError: 验证失败
        """
        if not self.is_initialized():
            raise RuntimeError("Pipeline not initialized")
        
        paper_id = hashlib.md5(url.encode()).hexdigest()
        start_time = datetime.now()
        
        with LoggingContextManager("paper_processing", {"paper_id": paper_id, "url": url}) as logger:
            try:
                logger.info(f"Starting paper processing: {url}")
                
                # Stage 1: 下载PDF
                await self._report_progress(
                    paper_id, ProcessingStage.DOWNLOAD, 0.0,
                    "开始下载PDF文件", start_time, progress_callback
                )
                
                file_path, file_hash = await self._download_pdf(url, paper_id, progress_callback)
                
                # 优先尝试基于文件哈希的缓存（命中则跳过后续解析流程）
                try:
                    cached = await self._load_cached_processed_paper(file_hash)
                except Exception as _e:
                    cached = None
                    self.logger.warning(f"Failed to load cached ProcessedPaper for {file_hash}: {_e}")

                if cached is not None:
                    # 使用当前会话的 paper_id 与 url 覆盖基础上下文，避免下游日志错乱
                    try:
                        cached.paper_id = paper_id
                        cached.source_url = url
                    except Exception:
                        pass

                    # 清理下载的临时PDF文件
                    await self._cleanup_temp_file(file_path, paper_id)

                    await self._report_progress(
                        paper_id, ProcessingStage.COMPLETE, 1.0,
                        "从缓存加载处理结果", start_time, progress_callback
                    )

                    logger.info(f"ProcessedPaper loaded from cache: hash={file_hash}")
                    log_paper_processing(
                        paper_id, "cache_hit", True,
                        page_count=cached.metadata.page_count,
                        quality=(cached.processing_stats or {}).get("final_quality")
                    )
                    return cached

                # Stage 2: 解析PDF（未命中缓存）
                parse_results = await self._parse_pdf(file_path, paper_id, progress_callback)
                
                # Stage 3: 提取图像
                await self._report_progress(
                    paper_id, ProcessingStage.IMAGE_EXTRACTION, 0.8,
                    "提取图像和表格", start_time, progress_callback
                )
                
                images = await self._extract_and_save_images(file_path, paper_id)
                
                # Stage 4: 合并结果
                await self._report_progress(
                    paper_id, ProcessingStage.CONTENT_MERGE, 0.9,
                    "合并解析结果", start_time, progress_callback
                )
                
                final_result = await self.hybrid_strategy.merge_results(parse_results)
                
                # Stage 5: 验证和构建最终对象
                await self._report_progress(
                    paper_id, ProcessingStage.VALIDATION, 0.95,
                    "验证处理结果", start_time, progress_callback
                )
                
                processed_paper = await self._build_processed_paper(
                    paper_id, url, file_hash, final_result, parse_results, images
                )

                # 写入缓存（容错，不影响主流程）
                try:
                    await self._save_processed_paper_cache(file_hash, processed_paper)
                except Exception as _e:
                    self.logger.warning(f"Failed to save ProcessedPaper cache for {file_hash}: {_e}")
                
                # 清理临时文件
                await self._cleanup_temp_file(file_path, paper_id)
                
                # 完成
                await self._report_progress(
                    paper_id, ProcessingStage.COMPLETE, 1.0,
                    "处理完成", start_time, progress_callback
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Paper processing completed in {processing_time:.2f}s")
                
                log_paper_processing(
                    paper_id, "complete", True,
                    page_count=processed_paper.metadata.page_count,
                    processing_time=processing_time,
                    quality=final_result.quality.value
                )
                
                return processed_paper
                
            except Exception as e:
                logger.error(f"Paper processing failed: {e}")
                log_paper_processing(paper_id, "failed", False, error_message=str(e))
                
                # 清理临时文件
                if 'file_path' in locals():
                    await self._cleanup_temp_file(file_path, paper_id)
                
                raise
    
    async def process_multiple_papers(
        self,
        urls: List[str],
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[Callable[[str, ProgressReport], None]] = None
    ) -> AsyncGenerator[Tuple[str, Union[ProcessedPaper, Exception]], None]:
        """
        并发处理多篇论文
        
        Args:
            urls: 论文URL列表
            max_concurrent: 最大并发数
            progress_callback: 进度回调函数
            
        Yields:
            Tuple[url, result_or_exception]
        """
        if not self.is_initialized():
            raise RuntimeError("Pipeline not initialized")
        
        max_concurrent = max_concurrent or self.config.performance.max_concurrent_downloads
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(url: str) -> Tuple[str, Union[ProcessedPaper, Exception]]:
            async with semaphore:
                try:
                    # 包装进度回调
                    wrapped_callback = None
                    if progress_callback:
                        wrapped_callback = lambda report: progress_callback(url, report)
                    
                    result = await self.process_paper(url, wrapped_callback)
                    return url, result
                except Exception as e:
                    return url, e
        
        # 创建所有任务
        tasks = [process_single(url) for url in urls]
        
        # 逐个返回完成的结果
        for coro in asyncio.as_completed(tasks):
            yield await coro
    
    async def _download_pdf(
        self,
        url: str,
        paper_id: str,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[str, str]:
        """下载PDF文件"""
        async def download_progress(message: str, progress: float = None):
            if progress_callback:
                await self._report_progress(
                    paper_id, ProcessingStage.DOWNLOAD, progress or 0.0,
                    message, datetime.now(), progress_callback
                )
        
        async with PDFDownloader(self.config) as downloader:
            return await downloader.download(url, download_progress)
    
    async def _parse_pdf(
        self,
        file_path: str,
        paper_id: str,
        progress_callback: Optional[Callable] = None
    ) -> List[ParseResult]:
        """使用所有可用解析器解析PDF"""
        results = []
        
        # 准备解析器列表
        parsers = []
        if self.nougat_parser and self.nougat_parser.is_available():
            parsers.append((self.nougat_parser, ProcessingStage.PARSE_NOUGAT))
        if self.pymupdf_parser and self.pymupdf_parser.is_available():
            parsers.append((self.pymupdf_parser, ProcessingStage.PARSE_PYMUPDF))
        
        if not parsers:
            raise PDFParsingError("No available parsers")
        
        # 顺序执行解析（避免资源竞争）
        for i, (parser, stage) in enumerate(parsers):
            try:
                base_progress = 0.1 + (i / len(parsers)) * 0.6  # 10%-70%
                
                await self._report_progress(
                    paper_id, stage, base_progress,
                    f"正在使用 {parser.get_name()} 解析PDF", datetime.now(), progress_callback
                )
                
                # 包装进度回调
                async def parser_progress(message: str, progress: float = None):
                    if progress_callback:
                        actual_progress = base_progress + (progress or 0) * (0.6 / len(parsers))
                        await self._report_progress(
                            paper_id, stage, actual_progress,
                            message, datetime.now(), progress_callback
                        )
                
                result = await parser.parse(file_path, parser_progress)
                results.append(result)
                
                self.logger.info(f"Parser {parser.get_name()} completed: "
                                f"success={result.success}, quality={result.quality.value}")
                
            except Exception as e:
                self.logger.error(f"Parser {parser.get_name()} failed: {e}")
                # 继续使用其他解析器
                
        if not results:
            raise PDFParsingError("All parsers failed")
        
        return results
    
    async def _extract_and_save_images(self, file_path: str, paper_id: str) -> List[ImageData]:
        """提取并保存图像"""
        def extract_images():
            images = []
            doc = fitz.open(file_path)
            
            try:
                # 创建图像保存目录
                image_dir = Path(self.config.paths.temp_downloads_folder) / f"{paper_id}_images"
                image_dir.mkdir(exist_ok=True)
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            
                            if pix.n - pix.alpha < 4:  # 确保不是CMYK
                                # 生成文件名
                                image_id = f"img_{page_num}_{img_index}"
                                image_path = image_dir / f"{image_id}.png"
                                
                                # 保存图像
                                pix.save(str(image_path))
                                
                                # 获取图像位置
                                img_rects = page.get_image_rects(xref)
                                if img_rects:
                                    rect = img_rects[0]
                                    if hasattr(rect, 'x0'):
                                        bbox = (
                                            float(rect.x0),
                                            float(rect.y0),
                                            float(rect.x1),
                                            float(rect.y1),
                                        )
                                    elif isinstance(rect, (tuple, list)) and len(rect) == 4:
                                        bbox = tuple(float(v) for v in rect)
                                    else:
                                        bbox = (0.0, 0.0, float(pix.width), float(pix.height))
                                else:
                                    bbox = (0.0, 0.0, float(pix.width), float(pix.height))
                                
                                # 计算文件大小
                                file_size = image_path.stat().st_size
                                
                                images.append(ImageData(
                                    image_id=image_id,
                                    page_number=page_num + 1,
                                    bbox=bbox,
                                    image_path=str(image_path),
                                    width=pix.width,
                                    height=pix.height,
                                    file_size=file_size
                                ))
                            
                            pix = None  # 释放内存
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                
                return images
                
            finally:
                doc.close()
        
        return await asyncio.to_thread(extract_images)
    
    async def _build_processed_paper(
        self,
        paper_id: str,
        source_url: str,
        file_hash: str,
        final_result: ParseResult,
        all_results: List[ParseResult],
        images: List[ImageData]
    ) -> ProcessedPaper:
        """构建最终的处理结果"""
        
        # 构建元数据
        metadata = final_result.metadata or PaperMetadata()
        metadata.url = source_url
        metadata.file_hash = file_hash
        
        # 提取摘要
        abstract = self._extract_abstract(final_result.content)
        
        # 提取参考文献
        references = self._extract_references(final_result.content)
        
        # 构建处理统计
        processing_stats = {
            "total_parsers_used": len(all_results),
            "successful_parsers": len([r for r in all_results if r.success]),
            "final_quality": final_result.quality.value,
            "final_confidence": final_result.confidence_score,
            "total_processing_time": sum(r.processing_time for r in all_results),
            "content_length": len(final_result.content),
            "section_count": len(final_result.sections),
            "image_count": len(images),
            "table_count": len(final_result.tables)
        }
        
        return ProcessedPaper(
            paper_id=paper_id,
            source_url=source_url,
            metadata=metadata,
            sections=final_result.sections,
            images=images,
            tables=final_result.tables,
            full_text=final_result.content,
            abstract=abstract,
            references=references,
            parse_results=all_results,
            processing_stats=processing_stats
        )

    # -------------------------
    # 缓存：ProcessedPaper <-> JSON
    # -------------------------
    def _get_cache_dir(self) -> Path:
        """返回缓存目录路径并确保存在。

        目录策略：使用 temp_downloads 的父目录作为根，在其下创建 processed_cache。
        例如：data/temp_downloads -> data/processed_cache
        """
        assert self.config is not None, "Config must be set before caching"
        temp_root = Path(self.config.paths.temp_downloads_folder).resolve()
        cache_dir = temp_root.parent / "processed_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    async def _load_cached_processed_paper(self, file_hash: str) -> Optional[ProcessedPaper]:
        """尝试从缓存中加载 ProcessedPaper。命中返回对象，未命中返回 None。"""
        cache_dir = self._get_cache_dir()
        cache_file = cache_dir / f"{file_hash}.json"
        if not cache_file.exists():
            return None

        try:
            async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                raw = await f.read()
            data = json.loads(raw)
            return self._processed_paper_from_dict(data)
        except Exception as e:
            self.logger.warning(f"Corrupted cache for {file_hash}: {e}")
            return None

    async def _save_processed_paper_cache(self, file_hash: str, processed_paper: ProcessedPaper) -> None:
        """将 ProcessedPaper 持久化为 JSON 缓存（使用文件哈希命名）。"""
        cache_dir = self._get_cache_dir()
        cache_file = cache_dir / f"{file_hash}.json"
        try:
            payload = self._processed_paper_to_dict(processed_paper)
            async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            # 仅记录，不阻断主流程
            self.logger.warning(f"Failed to write cache {cache_file}: {e}")

    def _processed_paper_to_dict(self, pp: ProcessedPaper) -> Dict[str, Any]:
        """序列化 ProcessedPaper 为可 JSON 的 dict。"""
        def dt(v: Optional[datetime]) -> Optional[str]:
            try:
                return v.isoformat() if isinstance(v, datetime) else None
            except Exception:
                return None

        def serialize_metadata(md: PaperMetadata) -> Dict[str, Any]:
            return {
                'title': md.title,
                'authors': md.authors,
                'abstract': md.abstract,
                'keywords': md.keywords,
                'doi': md.doi,
                'arxiv_id': md.arxiv_id,
                'publication_date': dt(md.publication_date),
                'journal': md.journal,
                'conference': md.conference,
                'pages': md.pages,
                'volume': md.volume,
                'issue': md.issue,
                'url': md.url,
                'file_hash': md.file_hash,
                'file_size': md.file_size,
                'page_count': md.page_count,
                'language': md.language,
                'creation_date': dt(md.creation_date),
                'modification_date': dt(md.modification_date),
                'producer': md.producer,
                'creator': md.creator,
            }

        def serialize_section(s: PaperSection) -> Dict[str, Any]:
            return {
                'section_id': s.section_id,
                'title': s.title,
                'content': s.content,
                'level': s.level,
                'page_numbers': s.page_numbers,
                'subsections': [serialize_section(ss) for ss in (s.subsections or [])],
                'references': s.references,
                'equations': s.equations,
                'images': s.images,
                'tables': s.tables,
            }

        def serialize_image(i: ImageData) -> Dict[str, Any]:
            return {
                'image_id': i.image_id,
                'page_number': i.page_number,
                'bbox': i.bbox,
                'image_path': i.image_path,
                'caption': i.caption,
                'alt_text': i.alt_text,
                'figure_number': i.figure_number,
                'image_type': i.image_type,
                'width': i.width,
                'height': i.height,
                'format': i.format,
                'file_size': i.file_size,
            }

        def serialize_table(t: TableData) -> Dict[str, Any]:
            return {
                'table_id': t.table_id,
                'page_number': t.page_number,
                'bbox': t.bbox,
                'caption': t.caption,
                'table_number': t.table_number,
                'data': t.data,
                'headers': t.headers,
                'markdown': t.markdown,
            }

        def serialize_result(r: ParseResult) -> Dict[str, Any]:
            return {
                'parser_name': r.parser_name,
                'success': r.success,
                'quality': getattr(r.quality, 'value', str(r.quality)),
                'content': r.content,
                'sections': [serialize_section(s) for s in (r.sections or [])],
                'images': [serialize_image(i) for i in (r.images or [])],
                'tables': [serialize_table(t) for t in (r.tables or [])],
                'metadata': serialize_metadata(r.metadata) if r.metadata else None,
                'processing_time': r.processing_time,
                'error_message': r.error_message,
                'confidence_score': r.confidence_score,
            }

        payload = {
            'paper_id': pp.paper_id,
            'source_url': pp.source_url,
            'metadata': serialize_metadata(pp.metadata),
            'sections': [serialize_section(s) for s in (pp.sections or [])],
            'images': [serialize_image(i) for i in (pp.images or [])],
            'tables': [serialize_table(t) for t in (pp.tables or [])],
            'full_text': pp.full_text,
            'abstract': pp.abstract,
            'references': pp.references,
            'parse_results': [serialize_result(r) for r in (pp.parse_results or [])],
            'processing_stats': pp.processing_stats,
            'created_at': dt(pp.created_at),
        }
        return payload

    def _processed_paper_from_dict(self, data: Dict[str, Any]) -> ProcessedPaper:
        """从 dict 反序列化为 ProcessedPaper（健壮处理）。"""
        def pdt(s: Optional[str]) -> Optional[datetime]:
            try:
                return datetime.fromisoformat(s) if s else None
            except Exception:
                return None

        def parse_metadata(d: Optional[Dict[str, Any]]) -> PaperMetadata:
            d = d or {}
            return PaperMetadata(
                title=d.get('title'),
                authors=d.get('authors') or [],
                abstract=d.get('abstract'),
                keywords=d.get('keywords') or [],
                doi=d.get('doi'),
                arxiv_id=d.get('arxiv_id'),
                publication_date=pdt(d.get('publication_date')),
                journal=d.get('journal'),
                conference=d.get('conference'),
                pages=d.get('pages'),
                volume=d.get('volume'),
                issue=d.get('issue'),
                url=d.get('url'),
                file_hash=d.get('file_hash'),
                file_size=int(d.get('file_size') or 0),
                page_count=int(d.get('page_count') or 0),
                language=d.get('language') or 'en',
                creation_date=pdt(d.get('creation_date')),
                modification_date=pdt(d.get('modification_date')),
                producer=d.get('producer'),
                creator=d.get('creator'),
            )

        def parse_section(d: Dict[str, Any]) -> PaperSection:
            return PaperSection(
                section_id=d.get('section_id') or '',
                title=d.get('title') or '',
                content=d.get('content') or '',
                level=int(d.get('level') or 1),
                page_numbers=d.get('page_numbers') or [],
                subsections=[parse_section(sd) for sd in (d.get('subsections') or [])],
                references=d.get('references') or [],
                equations=d.get('equations') or [],
                images=d.get('images') or [],
                tables=d.get('tables') or [],
            )

        def parse_image(d: Dict[str, Any]) -> ImageData:
            return ImageData(
                image_id=d.get('image_id') or '',
                page_number=int(d.get('page_number') or 0),
                bbox=tuple(d.get('bbox') or (0.0, 0.0, 0.0, 0.0)),
                image_path=d.get('image_path') or '',
                caption=d.get('caption'),
                alt_text=d.get('alt_text'),
                figure_number=d.get('figure_number'),
                image_type=d.get('image_type') or 'figure',
                width=int(d.get('width') or 0),
                height=int(d.get('height') or 0),
                format=d.get('format') or 'PNG',
                file_size=int(d.get('file_size') or 0),
            )

        def parse_table(d: Dict[str, Any]) -> TableData:
            return TableData(
                table_id=d.get('table_id') or '',
                page_number=int(d.get('page_number') or 0),
                bbox=tuple(d.get('bbox') or (0.0, 0.0, 0.0, 0.0)),
                caption=d.get('caption'),
                table_number=d.get('table_number'),
                data=d.get('data') or [],
                headers=d.get('headers') or [],
                markdown=d.get('markdown'),
            )

        def parse_result(d: Dict[str, Any]) -> ParseResult:
            quality_raw = d.get('quality')
            try:
                quality = ParseQuality(quality_raw) if isinstance(quality_raw, str) else ParseQuality.FAILED
            except Exception:
                quality = ParseQuality.FAILED
            return ParseResult(
                parser_name=d.get('parser_name') or '',
                success=bool(d.get('success')),
                quality=quality,
                content=d.get('content') or '',
                sections=[parse_section(s) for s in (d.get('sections') or [])],
                images=[parse_image(i) for i in (d.get('images') or [])],
                tables=[parse_table(t) for t in (d.get('tables') or [])],
                metadata=parse_metadata(d.get('metadata')) if d.get('metadata') else None,
                processing_time=float(d.get('processing_time') or 0.0),
                error_message=d.get('error_message'),
                confidence_score=float(d.get('confidence_score') or 0.0),
            )

        md = parse_metadata(data.get('metadata') or {})
        sections = [parse_section(s) for s in (data.get('sections') or [])]
        images = [parse_image(i) for i in (data.get('images') or [])]
        tables = [parse_table(t) for t in (data.get('tables') or [])]
        parse_results = [parse_result(r) for r in (data.get('parse_results') or [])]
        created_at = pdt(data.get('created_at')) or datetime.now()

        return ProcessedPaper(
            paper_id=data.get('paper_id') or '',
            source_url=data.get('source_url') or '',
            metadata=md,
            sections=sections,
            images=images,
            tables=tables,
            full_text=data.get('full_text') or '',
            abstract=data.get('abstract') or '',
            references=data.get('references') or [],
            parse_results=parse_results,
            processing_stats=data.get('processing_stats') or {},
            created_at=created_at,
        )
    
    def _extract_abstract(self, content: str) -> str:
        """提取摘要"""
        # 查找摘要部分
        abstract_patterns = [
            r'(?i)^#+\s*abstract\s*\n(.*?)(?=\n#+|\n\n[A-Z]|\Z)',
            r'(?i)abstract[:\s]*\n(.*?)(?=\n\n[A-Z]|\nKeywords|\n1\.|\Z)',
            r'(?i)\*\*abstract\*\*[:\s]*\n(.*?)(?=\n\n|\*\*|\Z)'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # 清理格式
                abstract = re.sub(r'\n+', ' ', abstract)
                abstract = re.sub(r'\s+', ' ', abstract)
                return abstract[:1000]  # 限制长度
        
        # 如果没有找到摘要，返回前几句（中文/英文均可）
        try:
            # 先按中文句末标点切分
            import re as _re
            cn_sentences = _re.split(r'[。！？]', content)
            candidates = [s.strip() for s in cn_sentences if s.strip()]
            if len(candidates) >= 2:
                return '。'.join(candidates[:3]).strip()[:500]
        except Exception:
            pass
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        return '. '.join(sentences[:3]).strip()[:500]
    
    def _extract_references(self, content: str) -> List[str]:
        """提取参考文献"""
        references = []
        
        # 查找参考文献部分
        ref_patterns = [
            r'(?i)^#+\s*references?\s*\n(.*?)(?=\n#+|\Z)',
            r'(?i)^#+\s*bibliography\s*\n(.*?)(?=\n#+|\Z)',
            r'(?i)\*\*references?\*\*\s*\n(.*?)(?=\n\*\*|\Z)'
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
            if match:
                ref_section = match.group(1)
                # 分割参考文献
                ref_lines = ref_section.split('\n')
                for line in ref_lines:
                    line = line.strip()
                    if line and (line.startswith('[') or re.match(r'^\d+\.', line)):
                        references.append(line)
                break
        
        return references[:50]  # 限制数量
    
    async def _report_progress(
        self,
        paper_id: str,
        stage: ProcessingStage,
        progress: float,
        message: str,
        started_at: datetime,
        callback: Optional[Callable]
    ) -> None:
        """报告处理进度"""
        if callback:
            report = ProgressReport(
                paper_id=paper_id,
                stage=stage,
                progress=progress,
                message=message,
                started_at=started_at,
                current_operation=message
            )
            
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(report)
                else:
                    callback(report)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    async def _cleanup_temp_file(self, file_path: str, paper_id: Optional[str] = None) -> None:
        """清理临时文件"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                    
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            "initialized": self._initialized,
            "available_parsers": [
                parser.get_name() for parser in [self.nougat_parser, self.pymupdf_parser]
                if parser and parser.is_available()
            ],
            "current_operations": len(self._current_operations),
            "nougat_available": NOUGAT_AVAILABLE,
            "layout_parser_available": LAYOUT_PARSER_AVAILABLE
        }


# 导入必要的模块用于图像处理（已在顶部导入 io）
