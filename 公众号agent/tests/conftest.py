"""
pytest配置文件
"""

import pytest
import asyncio
import sys
import tempfile
import shutil
from pathlib import Path


def pytest_configure(config):
    """pytest配置"""
    # 添加自定义标记
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_paper_metadata():
    """示例论文元数据（与实现一致）"""
    from modules.ingestion_pipeline import PaperMetadata
    from datetime import datetime, timezone
    
    return PaperMetadata(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        arxiv_id="1706.03762",
        url="https://arxiv.org/abs/1706.03762",
        publication_date=datetime(2017, 6, 12, tzinfo=timezone.utc),
        doi="10.48550/arXiv.1706.03762"
    )


@pytest.fixture
def sample_processed_paper(sample_paper_metadata):
    """示例处理后的论文（与实现一致）"""
    from modules.ingestion_pipeline import ProcessedPaper, PaperSection
    
    sections = [
        PaperSection(
            section_id="section_0",
            title="Abstract",
            content="The dominant sequence transduction models...",
            level=1,
            page_numbers=[1]
        ),
        PaperSection(
            section_id="section_1",
            title="Introduction",
            content="Recurrent neural networks...",
            level=1,
            page_numbers=[1, 2]
        )
    ]
    
    return ProcessedPaper(
        paper_id="demo",
        source_url="https://arxiv.org/abs/1706.03762",
        metadata=sample_paper_metadata,
        sections=sections,
        images=[],
        tables=[],
        full_text="The dominant sequence transduction models...\n\nRecurrent neural networks...",
        abstract="The dominant sequence transduction models...",
        references=[],
        parse_results=[],
        processing_stats={}
    )


@pytest.fixture
def sample_global_analysis_map():
    """示例全局分析映射（与实现一致字段）"""
    from modules.analysis_core import GlobalAnalysisMap
    from datetime import datetime, timezone
    
    return GlobalAnalysisMap(
        paper_id="pid-001",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        abstract_summary="Transformer 摘要总结",
        research_domain="Natural Language Processing",
        key_contributions=["Introduced the Transformer architecture", "Self-attention"],
        methodology_overview="Proposed attention-based sequence-to-sequence model",
        main_findings=["State-of-the-art results"],
        limitations=["Requires large datasets"],
        future_directions=["Scaling", "Pretraining"],
        technical_complexity=0.8,
        novelty_score=0.9,
        practical_relevance=0.8,
        related_concepts=["Self-attention", "Multi-head attention"],
        terminology_glossary={},
        mathematical_concepts=["Scaled Dot-Product Attention", "Multi-Head Attention"],
        created_at=datetime.now(timezone.utc),
    )
