#!/usr/bin/env python3
"""
简化的PDF测试脚本
直接测试PDF处理和分析功能，不涉及论文发现
"""

import asyncio
import sys
import os
import hashlib
import json
from datetime import datetime, timezone
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
import aiofiles
import shutil
from dataclasses import asdict

async def test_pdf_processing():
    """测试PDF处理功能"""
    print("🧪 开始测试PDF处理功能...")
    
    # 检查测试PDF是否存在
    pdf_path = Path("test_simple.pdf")
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

        # ====== 中间结果持久化（按PDF哈希命名）======
        file_hash = (
            getattr(getattr(processed_paper, 'metadata', None), 'file_hash', None)
            or hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        )
        diag_root = Path("output") / "diagnostics" / file_hash
        diag_root.mkdir(parents=True, exist_ok=True)

        async def save_text(rel: str, text: str) -> None:
            p = diag_root / rel
            async with aiofiles.open(p, 'w', encoding='utf-8') as f:
                await f.write(text or "")
            print(f"   ↳ 写入 {p}")

        async def save_json(rel: str, obj) -> None:
            p = diag_root / rel
            async with aiofiles.open(p, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(obj, ensure_ascii=False, indent=2))
            print(f"   ↳ 写入 {p}")

        # 摄取阶段结果
        print("💾 保存摄取阶段（ingestion）中间结果...")
        await save_text("ingestion_full_text.md", processed_paper.full_text)
        await save_json(
            "ingestion_sections.json",
            [
                {
                    "section_id": s.section_id,
                    "title": s.title,
                    "level": s.level,
                    "content_preview": (s.content or "")[:800]
                }
                for s in (processed_paper.sections or [])
            ],
        )
        await save_json(
            "ingestion_images.json",
            [
                {
                    "image_id": i.image_id,
                    "page": i.page_number,
                    "bbox": i.bbox,
                    "path": i.image_path,
                    "size": i.file_size,
                }
                for i in (processed_paper.images or [])
            ],
        )
        await save_json("ingestion_stats.json", processed_paper.processing_stats)
        await save_json(
            "ingestion_parse_results.json",
            [
                {
                    "parser": r.parser_name,
                    "success": r.success,
                    "quality": getattr(getattr(r, 'quality', None), 'value', str(getattr(r, 'quality', None))),
                    "content_length": len(r.content or ""),
                    "sections": len(r.sections or []),
                    "confidence": r.confidence_score,
                    "error": r.error_message,
                }
                for r in (processed_paper.parse_results or [])
            ],
        )

        # 记录 LLM 的输入提示词（猴子补丁 llm_provider.generate）
        print("🧾 启用提示词记录器：将保存所有 LLM 输入提示到 diagnostics 目录")
        llm_prompts_dir = diag_root / "llm_prompts"
        llm_prompts_dir.mkdir(exist_ok=True)
        _llm_call_counter = {"n": 0}
        _original_generate = llm_provider.generate

        async def _wrapped_generate(
            messages,
            model=None,
            temperature=None,
            max_tokens=None,
            system_prompt=None,
            use_insight_model=False,
            **kwargs,
        ):
            try:
                _llm_call_counter["n"] += 1
                idx = _llm_call_counter["n"]
                payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model
                    or (
                        getattr(config.llm_settings, "insight_model", None)
                        if use_insight_model
                        else getattr(config.llm_settings, "primary_model", None)
                    ),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "system_prompt": system_prompt,
                    "use_insight_model": use_insight_model,
                    "messages": messages,
                    "additional_params": kwargs,
                }
                async with aiofiles.open(
                    llm_prompts_dir / f"{idx:03d}_prompt.json", "w", encoding="utf-8"
                ) as f:
                    await f.write(json.dumps(payload, ensure_ascii=False, indent=2))
            except Exception as _e:
                try:
                    print(f"⚠️ LLM 提示词记录失败: {_e}")
                except Exception:
                    pass
            return await _original_generate(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                use_insight_model=use_insight_model,
                **kwargs,
            )

        llm_provider.generate = _wrapped_generate
        
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

        # 分析阶段中间结果
        print("💾 保存分析阶段（analysis）中间结果...")
        gm = analysis_result.global_map
        gm_payload = {
            "paper_id": gm.paper_id,
            "title": gm.title,
            "authors": gm.authors,
            "abstract_summary": gm.abstract_summary,
            "research_domain": gm.research_domain,
            "key_contributions": gm.key_contributions,
            "methodology_overview": gm.methodology_overview,
            "main_findings": gm.main_findings,
            "limitations": gm.limitations,
            "future_directions": gm.future_directions,
            "technical_complexity": gm.technical_complexity,
            "novelty_score": gm.novelty_score,
            "practical_relevance": gm.practical_relevance,
            "related_concepts": gm.related_concepts,
            "terminology_glossary": gm.terminology_glossary,
            "mathematical_concepts": gm.mathematical_concepts,
            "created_at": gm.created_at.isoformat() if getattr(gm, 'created_at', None) else None,
        }
        await save_json("analysis_global_map.json", gm_payload)
        await save_json(
            "analysis_chunks.json",
            [
                {
                    "chunk_id": c.chunk_id,
                    "section_type": c.section_type,
                    "start": c.start_position,
                    "end": c.end_position,
                    "topic": c.semantic_topic,
                    "importance": c.importance_score,
                    "complexity": c.complexity_level,
                    "content_preview": (c.content or "")[:800],
                }
                for c in (analysis_result.semantic_chunks or [])
            ],
        )
        await save_json(
            "analysis_chunk_explanations.json",
            [
                {
                    "chunk_id": e.chunk_id,
                    "confidence": e.confidence_score,
                    "key_insights": e.key_insights,
                    "terminology": e.terminology_explained,
                    "math": e.mathematical_explanations,
                    "llm_model": e.llm_model_used,
                    "explanation": e.explanation,
                }
                for e in (analysis_result.chunk_explanations or [])
            ],
        )
        await save_text("analysis_overall_summary.md", analysis_result.overall_summary or "")
        await save_json("analysis_research_insights.json", analysis_result.research_insights)
        await save_json("analysis_practical_implications.json", analysis_result.practical_implications)
        await save_json("analysis_quality_metrics.json", analysis_result.quality_metrics)
        
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
        
        # 合成阶段输入保存
        print("💾 保存合成阶段（synthesis）输入...")
        await save_json(
            "synthesis_analysis_pairs.json",
            [
                {
                    "section_id": p.section_id,
                    "confidence": p.confidence_score,
                    "original_preview": (p.original or "")[:600],
                    "analysis_preview": (p.analysis or "")[:600],
                }
                for p in analysis_pairs
            ],
        )
        await save_json("synthesis_global_analysis.json", asdict(global_analysis))
        await save_json("synthesis_paper_metadata.json", asdict(paper_metadata))

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
            # 复制一份到诊断目录，便于单点查看
            try:
                target_md = diag_root / "final_article.md"
                src = Path(article.file_path)
                if src.exists():
                    async with aiofiles.open(src, 'r', encoding='utf-8') as rf:
                        content = await rf.read()
                    await save_text("final_article.md", content)
                # 记录原始路径
                await save_text("final_article_path.txt", article.file_path)
            except Exception as _e:
                logger.warning(f"复制最终文章到诊断目录失败: {_e}")

        # 清理 data/temp_downloads 中的临时文件（本次测试不再需要）
        try:
            temp_dir = Path(config.paths.temp_downloads_folder)
            if temp_dir.exists():
                for item in temp_dir.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                        else:
                            shutil.rmtree(item, ignore_errors=True)
                    except Exception as _e:
                        logger.warning(f"清理临时文件失败: {item} -> {_e}")
                print("✓ 已清理 data/temp_downloads 目录")
        except Exception as _e:
            print(f"⚠️ 清理临时目录失败: {_e}")

        # 清理老旧日志（仅保留当前主日志文件）
        try:
            log_file = Path(config.logging.file_path)
            logs_dir = log_file.parent
            if logs_dir.exists():
                for lf in logs_dir.glob("*.log.*"):
                    try:
                        lf.unlink()
                    except Exception:
                        pass
            print(f"🗑️ 已清理旧日志，当前日志: {log_file}")
        except Exception as _e:
            print(f"⚠️ 日志清理失败: {_e}")
        
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
