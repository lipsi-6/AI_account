#!/usr/bin/env python3
"""
简化的PDF测试脚本
直接测试PDF处理和分析功能，不涉及论文发现
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目路径到sys.path
sys.path.insert(0, str(Path(__file__).parent))

from modules.config_manager import ConfigManager
from modules.logger import setup_logging, get_logger
from modules.llm_provider import LLMProviderService
from modules.embedding_provider import EmbeddingProviderService
from modules.memory_system import HybridMemorySystem
from modules.ingestion_pipeline import IngestionPipeline
from modules.analysis_core import DeepAnalysisCore, AnalysisRequest
from modules.synthesis_engine import SynthesisEngine

async def test_pdf_processing():
    """测试PDF处理功能"""
    print("🧪 开始测试PDF处理功能...")
    
    # 检查测试PDF是否存在
    pdf_path = Path("test.pdf")
    if not pdf_path.exists():
        print("❌ 测试PDF文件不存在！")
        return False
    
    print(f"✓ 找到测试PDF: {pdf_path}")
    
    try:
        # 1. 初始化配置管理器
        config = ConfigManager()
        await config.initialize("config.yaml")
        print("✓ 配置管理器初始化完成")
        
        # 2. 设置日志系统
        await setup_logging(config)
        logger = get_logger("pdf_test")
        print("✓ 日志系统初始化完成")
        
        # 3. 初始化LLM提供商服务
        llm_provider = LLMProviderService(config)
        print("✓ LLM服务初始化完成")
        
        # 4. 初始化嵌入提供商服务
        embedding_provider = EmbeddingProviderService(config)
        print("✓ 嵌入服务初始化完成")
        
        # 5. 初始化记忆系统
        memory_system = HybridMemorySystem(config, embedding_provider)
        await memory_system.initialize()
        print("✓ 记忆系统初始化完成")
        
        # 6. 初始化消化管道
        ingestion_pipeline = IngestionPipeline()
        await ingestion_pipeline.initialize(config)
        print("✓ 消化管道初始化完成")
        
        # 7. 初始化分析核心
        analysis_core = DeepAnalysisCore(
            config,
            llm_provider,
            memory_system,
            embedding_provider,
        )
        await analysis_core.initialize()
        print("✓ 分析核心初始化完成")
        
        # 8. 初始化合成引擎
        synthesis_engine = SynthesisEngine(config, llm_provider, memory_system)
        await synthesis_engine.initialize()
        print("✓ 合成引擎初始化完成")
        
        print("\n📄 开始处理PDF...")
        
        # 处理PDF
        processed_paper = await ingestion_pipeline.process_paper(str(pdf_path.absolute()))
        
        if not processed_paper:
            print("❌ PDF处理失败")
            return False
            
        print(f"✓ PDF解析完成，提取 {len(processed_paper.sections)} 个章节")
        
        # 进行深度分析
        print("🔍 开始深度分析...")
        request = AnalysisRequest(
            paper_content=processed_paper.full_text,
            paper_title=processed_paper.metadata.title or "测试论文",
            paper_authors=processed_paper.metadata.authors or [],
            analysis_depth=2,  # 降低分析深度以加快测试
            include_mathematical_details=True,
            include_historical_context=False,  # 简化分析
            custom_focus_areas=None,
            output_language="zh-CN",
        )
        analysis_result = await analysis_core.analyze_paper(request)
        print(f"✓ 分析完成，生成 {len(analysis_result.chunk_explanations)} 个意群解释")
        
        # 生成最终文章
        print("📝 开始合成文章...")
        from modules.synthesis_engine import (
            AnalysisPair,
            GlobalAnalysisMap,
            PaperMetadata,
            FigureReference,
        )
        
        # 构建分析对
        chunk_map = {c.chunk_id: c for c in analysis_result.semantic_chunks}
        analysis_pairs = []
        for exp in analysis_result.chunk_explanations:
            chunk = chunk_map.get(exp.chunk_id)
            if chunk:
                pair = AnalysisPair(
                    original=chunk.content,
                    analysis=exp.explanation,
                    section_id=chunk.section_type,
                    confidence_score=exp.confidence_score,
                )
                analysis_pairs.append(pair)
        
        # 构建全局分析映射
        global_map_src = analysis_result.global_map
        global_analysis = GlobalAnalysisMap(
            summary=analysis_result.overall_summary,
            key_insights=analysis_result.research_insights[:5],
            technical_terms=global_map_src.terminology_glossary,
            methodology_analysis=global_map_src.methodology_overview,
            significance_assessment=", ".join(analysis_result.practical_implications[:3])
        )
        
        # 构建论文元数据
        paper_metadata = PaperMetadata(
            title=processed_paper.metadata.title or "测试论文",
            authors=processed_paper.metadata.authors or [],
            institutions=None,
            publication_date=None,
            arxiv_id=None,
            doi=None,
            abstract=processed_paper.abstract,
        )
        
        # 合成文章
        article = await synthesis_engine.synthesize_article(
            analysis_pairs=analysis_pairs,
            global_analysis=global_analysis,
            paper_metadata=paper_metadata,
            figures=[],
        )
        
        print(f"✅ 测试完成！文章已生成: {article.title}")
        if hasattr(article, 'file_path') and article.file_path:
            print(f"📄 输出文件: {article.file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️ 警告: 未设置API密钥环境变量 (OPENAI_API_KEY 或 DEEPSEEK_API_KEY)")
        print("请设置你的DeepSeek API密钥:")
        print("export OPENAI_API_KEY='你的deepseek_api_key'")
        
    # 运行测试
    success = asyncio.run(test_pdf_processing())
    
    if success:
        print("\n🎉 PDF处理测试成功！")
        sys.exit(0)
    else:
        print("\n💥 PDF处理测试失败！")
        sys.exit(1)
