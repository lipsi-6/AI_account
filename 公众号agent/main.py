"""
Deep Scholar AI - 主应用入口与协调器

严格遵循依赖注入原则：
- 系统的唯一入口点
- 负责解析命令行参数、加载配置、初始化所有核心服务
- 协调各个模块的工作流，实现完整的论文发现到文章生成流程
"""

import asyncio
import argparse
import sys
import signal
from pathlib import Path
from typing import Optional, List
import traceback

# 导入所有模块
from modules.config_manager import ConfigManager
from modules.logger import setup_logging, get_logger
from modules.llm_provider import LLMProviderService
from modules.embedding_provider import EmbeddingProviderService
from modules.memory_system import HybridMemorySystem
from modules.discovery_engine import DiscoveryEngine
from modules.ingestion_pipeline import IngestionPipeline
from modules.analysis_core import DeepAnalysisCore, AnalysisRequest
from modules.synthesis_engine import (
    SynthesisEngine,
    AnalysisPair as SynthesisAnalysisPair,
    GlobalAnalysisMap as SynthesisGlobalAnalysisMap,
    PaperMetadata as SynthesisPaperMetadata,
    FigureReference as SynthesisFigureReference,
)
from modules.learning_loop import LearningLoop


class DeepScholarAI:
    """
    Deep Scholar AI 主应用类
    
    协调所有模块的工作流，实现完整的论文研究到文章生成的流程
    """
    
    def __init__(self):
        """无副作用构造函数"""
        self.logger = None
        self.config: Optional[ConfigManager] = None
        self.llm_provider: Optional[LLMProviderService] = None
        self.embedding_provider: Optional[EmbeddingProviderService] = None
        self.memory_system: Optional[HybridMemorySystem] = None
        self.discovery_engine: Optional[DiscoveryEngine] = None
        self.ingestion_pipeline: Optional[IngestionPipeline] = None
        self.analysis_core: Optional[DeepAnalysisCore] = None
        self.synthesis_engine: Optional[SynthesisEngine] = None
        self.learning_loop: Optional[LearningLoop] = None
        
        self._initialized = False
        self._shutdown_event = asyncio.Event()
        self._interactive_last_discovered = []  # [(paper_id, title)] for feedback mapping
        
        # 注册信号处理器
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """设置信号处理器以优雅关闭"""
        def signal_handler(signum, frame):
            print(f"\n接收到信号 {signum}，开始优雅关闭...")
            try:
                loop = asyncio.get_running_loop()
                # 在正确的事件循环线程上调度关闭任务
                loop.call_soon_threadsafe(asyncio.create_task, self.shutdown())
            except RuntimeError:
                # 退路：无运行中事件循环时，同步创建并运行一个临时循环关闭
                try:
                    asyncio.run(self.shutdown())
                except Exception:
                    pass
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self, config_path: str) -> None:
        """
        异步初始化所有服务
        
        Args:
            config_path: 配置文件路径
        """
        try:
            print("正在初始化 Deep Scholar AI...")
            
            # 1. 初始化配置管理器
            self.config = ConfigManager()
            await self.config.initialize(config_path)
            print("✓ 配置管理器初始化完成")
            
            # 2. 设置日志系统
            await setup_logging(self.config)
            self.logger = get_logger("deep_scholar_ai")
            self.logger.info("Deep Scholar AI 开始初始化")
            print("✓ 日志系统初始化完成")
            
            # 3. 初始化LLM提供商服务
            self.llm_provider = LLMProviderService(self.config)
            self.logger.info("LLM提供商服务初始化完成")
            print("✓ LLM服务初始化完成")
            
            # 4. 初始化嵌入提供商服务
            self.embedding_provider = EmbeddingProviderService(self.config)
            self.logger.info("嵌入提供商服务初始化完成")
            print("✓ 嵌入服务初始化完成")
            
            # 5. 初始化记忆系统
            self.memory_system = HybridMemorySystem(self.config, self.embedding_provider)
            await self.memory_system.initialize()
            self.logger.info("记忆系统初始化完成")
            print("✓ 记忆系统初始化完成")
            
            # 6. 初始化发现引擎
            self.discovery_engine = DiscoveryEngine(
                self.config,
                self.embedding_provider,
                self.llm_provider,
                self.memory_system,
            )
            await self.discovery_engine.initialize()
            self.logger.info("发现引擎初始化完成")
            print("✓ 发现引擎初始化完成")
            
            # 7. 初始化消化管道
            self.ingestion_pipeline = IngestionPipeline()
            await self.ingestion_pipeline.initialize(self.config)
            self.logger.info("消化管道初始化完成")
            print("✓ 消化管道初始化完成")
            
            # 8. 初始化分析核心
            self.analysis_core = DeepAnalysisCore(
                self.config,
                self.llm_provider,
                self.memory_system,
                self.embedding_provider,
            )
            await self.analysis_core.initialize()
            self.logger.info("分析核心初始化完成")
            print("✓ 分析核心初始化完成")
            
            # 9. 初始化合成引擎
            self.synthesis_engine = SynthesisEngine(self.config, self.llm_provider, self.memory_system)
            await self.synthesis_engine.initialize()
            self.logger.info("合成引擎初始化完成")
            print("✓ 合成引擎初始化完成")

            # 10. 初始化学习循环（持续进化回路）
            try:
                self.learning_loop = LearningLoop(
                    config=self.config,
                    memory_system=self.memory_system,
                    embedding_service=self.embedding_provider,
                )
                await self.learning_loop.initialize()
                # 注入到 DiscoveryEngine 与 AnalysisCore
                try:
                    if self.discovery_engine:
                        self.discovery_engine.learning_loop = self.learning_loop
                        # 若 TasteEngine 已可用，补全注入
                        if getattr(self.discovery_engine, "taste_engine", None):
                            self.learning_loop.set_taste_engine(self.discovery_engine.taste_engine)
                    if self.analysis_core:
                        self.analysis_core.learning_loop = self.learning_loop
                except Exception as _e:
                    self.logger.warning(f"LearningLoop cross-injection failed: {_e}")
            except Exception as e:
                self.logger.warning(f"LearningLoop 初始化失败（非致命）：{e}")
            
            self._initialized = True
            self.logger.info("Deep Scholar AI 全部服务初始化完成")
            print("🚀 Deep Scholar AI 初始化完成，系统就绪！")
            
        except Exception as e:
            error_msg = f"初始化失败: {e}"
            if self.logger:
                self.logger.error(error_msg, exc_info=True)
            else:
                print(f"❌ {error_msg}")
                traceback.print_exc()
            raise
    
    async def run_discovery_cycle(self) -> None:
        """运行论文发现周期"""
        if not self._initialized:
            raise RuntimeError("系统未初始化")
        
        self.logger.info("开始论文发现周期")
        
        try:
            # 发现新论文
            papers = await self.discovery_engine.discover_papers()
            self.logger.info(f"发现 {len(papers)} 篇论文")
            
            if not papers:
                self.logger.info("未发现新论文，跳过此周期")
                return
            
            # 根据发现引擎结果筛选（discover_papers 已包含评分与排序时可省略再次排序）
            # 若需额外排序，可启用 rank_papers；此处为避免重复评分，直接切片
            top_papers = papers[: self.config.preferences.daily_paper_limit]
            
            self.logger.info(f"筛选出 {len(top_papers)} 篇高质量论文")
            
            return top_papers
            
        except Exception as e:
            self.logger.error(f"论文发现周期失败: {e}", exc_info=True)
            raise
    
    async def process_paper(self, paper_url: str) -> Optional[str]:
        """
        处理单篇论文，从下载到最终文章生成
        
        Args:
            paper_url: 论文URL
            
        Returns:
            生成的文章文件路径，如果处理失败则返回None
        """
        if not self._initialized:
            raise RuntimeError("系统未初始化")
        
        self.logger.info(f"开始处理论文: {paper_url}")
        
        try:
            # 1. 下载和解析PDF
            self.logger.info("步骤 1/4: 下载和解析PDF")
            processed_paper = await self.ingestion_pipeline.process_paper(paper_url)
            
            if not processed_paper:
                self.logger.warning(f"论文处理失败: {paper_url}")
                return None
            
            self.logger.info(f"PDF解析完成，提取 {len(processed_paper.sections)} 个章节")
            
            # 2. 深度分析
            self.logger.info("步骤 2/4: 深度分析论文内容")
            request = AnalysisRequest(
                paper_content=processed_paper.full_text,
                paper_title=processed_paper.metadata.title or "",
                paper_authors=processed_paper.metadata.authors or [],
                analysis_depth=3,
                include_mathematical_details=True,
                include_historical_context=True,
                custom_focus_areas=None,
                output_language="zh-CN",
            )
            analysis_result = await self.analysis_core.analyze_paper(request)

            self.logger.info(
                f"分析完成，生成 {len(analysis_result.chunk_explanations)} 个意群解释"
            )
            
            # 3. 存储到记忆系统
            self.logger.info("步骤 3/4: 存储到记忆系统")
            # 将分析结果存入混合记忆系统（真实实现）
            try:
                await self.memory_system.store_paper_analysis(
                    processed_paper.metadata,
                    analysis_result,
                )
            except Exception as e:
                self.logger.warning(f"store_paper_analysis failed: {e}")
            
            # 4. 合成文章
            self.logger.info("步骤 4/4: 合成最终文章")

            # 将 DeepAnalysisCore 的结果转换为合成引擎的数据结构
            # 4.1 构建分析对（original=chunk内容, analysis=解释）
            chunk_map = {c.chunk_id: c for c in analysis_result.semantic_chunks}
            analysis_pairs: List[SynthesisAnalysisPair] = []
            for exp in analysis_result.chunk_explanations:
                chunk = chunk_map.get(exp.chunk_id)
                if not chunk:
                    continue
                pair = SynthesisAnalysisPair(
                    original=chunk.content,
                    analysis=exp.explanation,
                    section_id=chunk.section_type,
                    confidence_score=exp.confidence_score,
                )
                analysis_pairs.append(pair)

            # 4.2 构建全局分析映射（转换结构）
            global_map_src = analysis_result.global_map
            significance = ", ".join(analysis_result.practical_implications[:5])
            s_global = SynthesisGlobalAnalysisMap(
                summary=analysis_result.overall_summary,
                key_insights=analysis_result.research_insights[:10],
                technical_terms=global_map_src.terminology_glossary,
                methodology_analysis=global_map_src.methodology_overview,
                significance_assessment=significance or ""
            )

            # 4.3 构建论文元数据
            s_meta = SynthesisPaperMetadata(
                title=processed_paper.metadata.title,
                authors=processed_paper.metadata.authors,
                institutions=None,
                publication_date=(
                    processed_paper.metadata.publication_date.isoformat()
                    if processed_paper.metadata.publication_date
                    else None
                ),
                arxiv_id=processed_paper.metadata.arxiv_id,
                doi=processed_paper.metadata.doi,
                abstract=processed_paper.abstract,
            )

            # 4.4 将摄取阶段的图片映射为合成引擎图片引用，并生成语义上下文
            def _build_figure_context(page_number: int) -> str:
                try:
                    # 优先使用包含该页的章节标题与内容片段作为上下文
                    for sec in processed_paper.sections:
                        if sec and getattr(sec, 'page_numbers', None) and page_number in (sec.page_numbers or []):
                            title = (sec.title or '').strip()
                            excerpt = (sec.content or '')[:500].strip()
                            ctx_parts = [p for p in [title, excerpt] if p]
                            if ctx_parts:
                                return ' '.join(ctx_parts)
                    # 回退到摘要或全文前缀
                    if processed_paper.abstract:
                        return processed_paper.abstract[:500]
                    return processed_paper.full_text[:500]
                except Exception:
                    return processed_paper.abstract or processed_paper.full_text[:500]

            fig_refs: List[SynthesisFigureReference] = []
            try:
                for img in getattr(processed_paper, 'images', []) or []:
                    caption = (getattr(img, 'caption', None) or getattr(img, 'alt_text', None) or f"Figure (p.{getattr(img, 'page_number', '?')})").strip()
                    context = _build_figure_context(getattr(img, 'page_number', 0))
                    file_path = getattr(img, 'image_path', '') or ''
                    if file_path:
                        fig_refs.append(
                            SynthesisFigureReference(
                                figure_id=getattr(img, 'image_id', ''),
                                caption=caption,
                                file_path=file_path,
                                context=context,
                                relevance_score=0.0,
                            )
                        )
            except Exception as _e:
                self.logger.warning(f"构建图片引用失败，将继续不带图片: {_e}")

            article = await self.synthesis_engine.synthesize_article(
                analysis_pairs=analysis_pairs,
                global_analysis=s_global,
                paper_metadata=s_meta,
                figures=fig_refs,
            )
            self.logger.info(f"文章生成完成: {article.title}")
            
            # 更新品味引擎（正面反馈）
            await self.discovery_engine.update_taste_from_paper(
                processed_paper.metadata, positive=True
            )
            
            # 返回保存路径
            return getattr(article, 'file_path', None)
            
        except Exception as e:
            self.logger.error(f"论文处理失败 {paper_url}: {e}", exc_info=True)
            
            # 更新品味引擎（负面反馈）
            try:
                if 'processed_paper' in locals():
                    await self.discovery_engine.update_taste_from_paper(
                        processed_paper.metadata, positive=False
                    )
            except:
                pass
            
            return None
    
    async def run_daily_workflow(self) -> None:
        """运行每日工作流"""
        if not self._initialized:
            raise RuntimeError("系统未初始化")
        
        self.logger.info("开始每日工作流")
        
        try:
            # 1. 发现论文
            top_papers = await self.run_discovery_cycle()
            
            if not top_papers:
                self.logger.info("今日无高质量论文发现")
                return
            
            # 2. 选择最值得深度解读的论文
            selected_paper = top_papers[0]  # 选择排名最高的论文
            paper_title = getattr(selected_paper.metadata, 'title', None) or 'Unknown Paper'
            self.logger.info(f"选择论文进行深度解读: {paper_title}")
            
            # 3. 处理论文
            paper_url = (
                getattr(selected_paper.metadata, 'pdf_url', None)
                or getattr(selected_paper.metadata, 'html_url', None)
                or getattr(selected_paper.metadata, 'source_url', None)
            )
            article_path = await self.process_paper(paper_url) if paper_url else None
            
            if article_path:
                self.logger.info(f"每日工作流完成，文章已生成: {article_path}")
                print(f"✅ 今日文章已生成: {article_path}")
                # 生成学习进度报告（非阻塞可忽略错误）
                try:
                    if self.learning_loop:
                        await self.learning_loop.generate_progress_report()
                except Exception as _e:
                    self.logger.warning(f"生成学习进度报告失败: {_e}")
            else:
                self.logger.warning("每日工作流完成，但未能生成文章")
                print("⚠️ 今日论文处理失败，未能生成文章")
            
        except Exception as e:
            self.logger.error(f"每日工作流失败: {e}", exc_info=True)
            print(f"❌ 每日工作流失败: {e}")
    
    async def run_interactive_mode(self) -> None:
        """运行交互模式"""
        if not self._initialized:
            raise RuntimeError("系统未初始化")
        
        self.logger.info("进入交互模式")
        print("\n📚 Deep Scholar AI 交互模式")
        print("可用命令:")
        print("  1. discover - 运行论文发现")
        print("  2. process <URL> - 处理指定论文")
        print("  3. daily - 运行每日工作流")
        print("  4. stats - 显示系统统计")
        print("  5. quit - 退出")
        print("  6. report - 生成学习进度报告")
        print("  7. feedback <index> <+|-> - 对最近发现的论文进行反馈")
        print("  8. learn - 查看学习状态摘要")
        
        while not self._shutdown_event.is_set():
            try:
                # 使用非阻塞方式读取输入，避免阻塞事件循环
                command = (await asyncio.to_thread(input, "\n请输入命令: ")).strip().lower()
                
                if command == "quit" or command == "q":
                    break
                elif command == "discover":
                    papers = await self.run_discovery_cycle()
                    if papers:
                        print(f"发现 {len(papers)} 篇论文:")
                        self._interactive_last_discovered = []
                        for i, paper in enumerate(papers[:10], 1):
                            title = getattr(getattr(paper, 'metadata', None), 'title', None) or 'Unknown Paper'
                            pid = getattr(paper, 'id', None) or f"idx-{i}"
                            self._interactive_last_discovered.append((pid, title))
                            short = pid[:8] if isinstance(pid, str) else str(pid)
                            print(f"  {i}. {title} [id:{short}]")
                elif command.startswith("process "):
                    url = command[8:].strip()
                    if url:
                        result = await self.process_paper(url)
                        if result:
                            print(f"文章已生成: {result}")
                        else:
                            print("论文处理失败")
                    else:
                        print("请提供论文URL")
                elif command == "daily":
                    await self.run_daily_workflow()
                elif command == "stats":
                    await self._show_stats()
                elif command == "report":
                    try:
                        if self.learning_loop:
                            report = await self.learning_loop.generate_progress_report()
                            print("\n🧠 学习进度报告（已写入记忆）：")
                            print(f"  keys: {', '.join(report.keys())}")
                        else:
                            print("学习循环未初始化")
                    except Exception as _e:
                        print(f"生成报告失败: {_e}")
                elif command == "learn":
                    try:
                        if self.learning_loop:
                            summary = await self.learning_loop.get_learning_summary()
                            print("\n🧠 学习状态摘要：")
                            for k, v in summary.items():
                                print(f"  {k}: {v}")
                        else:
                            print("学习循环未初始化")
                    except Exception as _e:
                        print(f"获取学习状态失败: {_e}")
                elif command.startswith("feedback "):
                    try:
                        args = command.split()
                        if len(args) != 3 or args[2] not in ['+', '-']:
                            print("用法: feedback <index> <+|->  例如: feedback 1 +")
                        else:
                            idx = int(args[1])
                            sign = args[2]
                            if 1 <= idx <= len(self._interactive_last_discovered):
                                paper_id, title = self._interactive_last_discovered[idx - 1]
                                ok = await self.discovery_engine.provide_feedback(paper_id, feedback=(sign=='+'))
                                if ok:
                                    print(f"已记录反馈: {'正面' if sign=='+' else '负面'} -> {title}")
                                else:
                                    print("反馈失败")
                            else:
                                print("索引超出范围，请先运行 discover 查看列表")
                    except Exception as _e:
                        print(f"反馈命令失败: {_e}")
                else:
                    print("未知命令，请重试")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"交互模式错误: {e}", exc_info=True)
                print(f"执行错误: {e}")
        
        self.logger.info("退出交互模式")
    
    async def _show_stats(self) -> None:
        """显示系统统计信息"""
        try:
            stats = {
                "记忆系统": self.memory_system.get_memory_stats(),
                "发现引擎": await self.discovery_engine.get_discovery_insights(),
                "嵌入服务": self.embedding_provider.get_cache_stats(),
            }
            
            print("\n📊 系统统计信息:")
            for service, service_stats in stats.items():
                print(f"\n{service}:")
                for key, value in service_stats.items():
                    print(f"  {key}: {value}")
                    
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}", exc_info=True)
            print(f"获取统计信息失败: {e}")
    
    async def shutdown(self) -> None:
        """优雅关闭所有服务"""
        self.logger.info("开始关闭 Deep Scholar AI")
        
        try:
            # 设置关闭事件
            self._shutdown_event.set()
            
            # 关闭各个服务
            services = [
                ("学习循环", self.learning_loop),
                ("合成引擎", self.synthesis_engine),
                ("分析核心", self.analysis_core),
                ("消化管道", self.ingestion_pipeline),
                ("发现引擎", self.discovery_engine),
                ("记忆系统", self.memory_system),
                ("嵌入服务", self.embedding_provider),
                ("LLM服务", self.llm_provider),
            ]
            
            for service_name, service in services:
                if service and hasattr(service, 'close'):
                    try:
                        await service.close()
                        self.logger.info(f"{service_name}已关闭")
                    except Exception as e:
                        self.logger.error(f"关闭{service_name}失败: {e}")
            
            self.logger.info("Deep Scholar AI 已优雅关闭")
            print("👋 Deep Scholar AI 已安全关闭")
            
        except Exception as e:
            print(f"关闭过程中出现错误: {e}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Deep Scholar AI - 虚拟AI研究员")
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )
    parser.add_argument(
        "--mode",
        choices=["daily", "interactive", "process"],
        default="interactive",
        help="运行模式 (默认: interactive)"
    )
    parser.add_argument(
        "--paper-url",
        help="指定要处理的论文URL (仅在process模式下使用)"
    )
    
    args = parser.parse_args()
    
    # 检查配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        print("请创建配置文件或使用 --config 参数指定其他路径")
        sys.exit(1)
    
    # 创建并初始化应用
    app = DeepScholarAI()
    
    try:
        await app.initialize(str(config_path))
        
        if args.mode == "daily":
            await app.run_daily_workflow()
        elif args.mode == "process":
            if not args.paper_url:
                print("❌ process模式需要提供 --paper-url 参数")
                sys.exit(1)
            result = await app.process_paper(args.paper_url)
            if result:
                print(f"✅ 文章已生成: {result}")
            else:
                print("❌ 论文处理失败")
                sys.exit(1)
        else:  # interactive
            await app.run_interactive_mode()
            
    except KeyboardInterrupt:
        print("\n接收到中断信号")
    except Exception as e:
        print(f"❌ 应用运行失败: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        await app.shutdown()


if __name__ == "__main__":
    # 设置事件循环策略（Windows兼容性）
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 运行主函数
    asyncio.run(main())
