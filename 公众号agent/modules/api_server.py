"""
HTTP API Server (FastAPI)

Production-grade, dependency-injected service exposing the agent's core
capabilities for a frontend client:

- POST /initialize                → boot services using config.yaml
- GET  /stats                     → system stats snapshot
- POST /discover                  → run discovery cycle
- POST /process                   → process a single paper by URL (http/file)
- GET  /drafts                    → list generated drafts
- GET  /drafts/{article_id}       → read a specific draft content
- GET  /memory/search             → hybrid memory search

Minimal surface; no speculative features. All long operations are async
and non-blocking. Errors are returned with structured messages.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi import Response
import io
import re
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config_manager import ConfigManager
from .logger import setup_logging, get_logger
from .llm_provider import LLMProviderService
from .embedding_provider import EmbeddingProviderService
from .memory_system import HybridMemorySystem, MemoryType
from .discovery_engine import DiscoveryEngine
from .ingestion_pipeline import IngestionPipeline
from .analysis_core import DeepAnalysisCore, AnalysisRequest
from .synthesis_engine import (
    SynthesisEngine,
    AnalysisPair as SynthesisAnalysisPair,
    GlobalAnalysisMap as SynthesisGlobalAnalysisMap,
    PaperMetadata as SynthesisPaperMetadata,
    FigureReference as SynthesisFigureReference,
)


class InitRequest(BaseModel):
    config_path: str = "config.yaml"


class ProcessRequest(BaseModel):
    paper_url: str


class MemorySearchResponse(BaseModel):
    results: List[Dict[str, Any]]


def create_app() -> FastAPI:
    app = FastAPI(title="Deep Scholar AI API", version="1.0.0")

    # CORS for local dev UI; constrain in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: Dict[str, Any] = {
        "initialized": False,
        "config": None,
        "logger": None,
        "llm": None,
        "embed": None,
        "memory": None,
        "discovery": None,
        "ingest": None,
        "analysis": None,
        "synthesis": None,
    }

    logger = get_logger("api_server")

    async def ensure_initialized():
        if not state["initialized"]:
            raise HTTPException(status_code=400, detail="Server not initialized. POST /initialize first.")

    @app.post("/initialize")
    async def initialize(req: InitRequest):
        if state["initialized"]:
            return {"status": "ok", "message": "already initialized"}
        try:
            cfg = ConfigManager()
            await cfg.initialize(req.config_path)
            await setup_logging(cfg)
            log = get_logger("api")
            llm = LLMProviderService(cfg)
            embed = EmbeddingProviderService(cfg)
            memory = HybridMemorySystem(cfg, embed)
            await memory.initialize()
            discovery = DiscoveryEngine(cfg, embed, llm, memory)
            await discovery.initialize()
            ingest = IngestionPipeline()
            await ingest.initialize(cfg)
            analysis = DeepAnalysisCore(cfg, llm, memory, embed)
            await analysis.initialize()
            synthesis = SynthesisEngine(cfg, llm, memory)
            await synthesis.initialize()

            state.update(
                initialized=True,
                config=cfg,
                logger=log,
                llm=llm,
                embed=embed,
                memory=memory,
                discovery=discovery,
                ingest=ingest,
                analysis=analysis,
                synthesis=synthesis,
            )
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"initialize failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stats")
    async def stats():
        await ensure_initialized()
        try:
            mem_stats = state["memory"].get_memory_stats()
            disc_insights = await state["discovery"].get_discovery_insights()
            embed_info = state["embed"].get_provider_info()
            return {
                "memory": mem_stats,
                "discovery": disc_insights,
                "embedding": embed_info,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/discover")
    async def discover(limit: int = 10):
        await ensure_initialized()
        try:
            papers = await state["discovery"].discover_papers(limit=limit)
            out = []
            for p in papers:
                m = p
                md = getattr(m, "metadata", None)
                out.append(
                    {
                        "id": getattr(m, "id", None),
                        "title": getattr(md, "title", None),
                        "authors": [getattr(a, "name", None) for a in (getattr(md, "authors", []) or [])],
                        "arxiv_id": getattr(md, "arxiv_id", None),
                        "pdf_url": getattr(md, "pdf_url", None),
                        "html_url": getattr(md, "html_url", None),
                        "relevance_score": getattr(m, "relevance_score", None),
                    }
                )
            return {"papers": out}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sse/discover")
    async def sse_discover(limit: int = 20):
        await ensure_initialized()

        async def event_gen():
            import json
            try:
                yield f"event: start\ndata: {json.dumps({'message': 'discover_started'})}\n\n"
                papers = await state["discovery"].discover_papers(limit=limit)
                # stream per paper for progress feel
                for p in papers:
                    md = getattr(p, "metadata", None)
                    item = {
                        "id": getattr(p, "id", None),
                        "title": getattr(md, "title", None),
                        "authors": [getattr(a, "name", None) for a in (getattr(md, "authors", []) or [])],
                        "arxiv_id": getattr(md, "arxiv_id", None),
                        "pdf_url": getattr(md, "pdf_url", None),
                        "html_url": getattr(md, "html_url", None),
                        "relevance_score": getattr(p, "relevance_score", None),
                    }
                    yield f"event: paper\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
                yield f"event: complete\ndata: {json.dumps({'count': len(papers)})}\n\n"
            except Exception as e:
                payload = {"error": str(e)}
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    @app.post("/process")
    async def process(req: ProcessRequest):
        await ensure_initialized()
        try:
            # Reuse main.py pipeline
            ingest = state["ingest"]
            analysis = state["analysis"]
            memory = state["memory"]
            synthesis = state["synthesis"]

            processed = await ingest.process_paper(req.paper_url)
            if not processed:
                raise HTTPException(status_code=400, detail="Ingestion failed")

            request = AnalysisRequest(
                paper_content=processed.full_text,
                paper_title=processed.metadata.title or "",
                paper_authors=processed.metadata.authors or [],
                analysis_depth=3,
                include_mathematical_details=True,
                include_historical_context=True,
                custom_focus_areas=None,
                output_language="zh-CN",
            )
            result = await analysis.analyze_paper(request)

            # store
            await memory.store_paper_analysis(processed.metadata, result)

            # map for synthesis
            chunk_map = {c.chunk_id: c for c in result.semantic_chunks}
            pairs: List[SynthesisAnalysisPair] = []
            for exp in result.chunk_explanations:
                chunk = chunk_map.get(exp.chunk_id)
                if not chunk:
                    continue
                pairs.append(
                    SynthesisAnalysisPair(
                        original=chunk.content,
                        analysis=exp.explanation,
                        section_id=chunk.section_type,
                        confidence_score=exp.confidence_score,
                    )
                )

            s_global = SynthesisAnalysisGlobalAdapter(result)
            s_meta = SynthesisPaperMetadata(
                title=processed.metadata.title,
                authors=processed.metadata.authors,
                institutions=None,
                publication_date=(
                    processed.metadata.publication_date.isoformat() if processed.metadata.publication_date else None
                ),
                arxiv_id=processed.metadata.arxiv_id,
                doi=processed.metadata.doi,
                abstract=processed.abstract,
            )

            # Map ingestion images -> synthesis figure references with minimal context
            def _build_figure_context(page_number: int) -> str:
                try:
                    for sec in processed.sections:
                        if getattr(sec, 'page_numbers', None) and page_number in (sec.page_numbers or []):
                            title = (sec.title or '').strip()
                            excerpt = (sec.content or '')[:500].strip()
                            ctx = ' '.join([p for p in [title, excerpt] if p])
                            if ctx:
                                return ctx
                    if processed.abstract:
                        return processed.abstract[:500]
                    return processed.full_text[:500]
                except Exception:
                    return processed.abstract or processed.full_text[:500]

            fig_refs: List[SynthesisFigureReference] = []
            try:
                for img in getattr(processed, 'images', []) or []:
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
            except Exception:
                fig_refs = []

            article = await synthesis.synthesize_article(
                analysis_pairs=pairs,
                global_analysis=s_global,
                paper_metadata=s_meta,
                figures=fig_refs,
            )

            return {"article_path": article.file_path, "title": article.title}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sse/process")
    async def sse_process(paper_url: str):
        await ensure_initialized()

        async def event_gen():
            import json
            try:
                ingest = state["ingest"]
                analysis = state["analysis"]
                memory = state["memory"]
                synthesis = state["synthesis"]

                # bridge progress from ingestion
                async def cb(report):
                    try:
                        payload = {
                            "stage": getattr(report, "stage", None).value if getattr(report, "stage", None) else None,
                            "progress": getattr(report, "progress", None),
                            "message": getattr(report, "message", None),
                        }
                        yield_line = "event: progress\n" + f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        yield yield_line  # generator injection pattern below
                    except Exception:
                        pass

                # To push cb events inside generator, we create a queue
                queue: asyncio.Queue[str] = asyncio.Queue()

                async def enqueue_progress(report):
                    import json as _json
                    try:
                        payload = {
                            "stage": getattr(report, "stage", None).value if getattr(report, "stage", None) else None,
                            "progress": getattr(report, "progress", None),
                            "message": getattr(report, "message", None),
                        }
                        await queue.put("event: progress\n" + f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n")
                    except Exception:
                        pass

                # run pipeline in background task
                async def run_pipeline():
                    try:
                        await queue.put("event: start\n" + "data: {\"message\": \"process_started\"}\n\n")
                        processed = await ingest.process_paper(paper_url, enqueue_progress)
                        if not processed:
                            await queue.put("event: error\n" + "data: {\"message\": \"ingestion_failed\"}\n\n")
                            return
                        await queue.put("event: stage\n" + "data: {\"name\": \"analysis_start\"}\n\n")
                        request = AnalysisRequest(
                            paper_content=processed.full_text,
                            paper_title=processed.metadata.title or "",
                            paper_authors=processed.metadata.authors or [],
                            analysis_depth=3,
                            include_mathematical_details=True,
                            include_historical_context=True,
                            custom_focus_areas=None,
                            output_language="zh-CN",
                        )
                        result = await analysis.analyze_paper(request)
                        await queue.put("event: stage\n" + "data: {\"name\": \"memory_store\"}\n\n")
                        try:
                            await memory.store_paper_analysis(processed.metadata, result)
                        except Exception:
                            pass

                        # map figures
                        def _build_figure_context(page_number: int) -> str:
                            try:
                                for sec in processed.sections:
                                    if getattr(sec, 'page_numbers', None) and page_number in (sec.page_numbers or []):
                                        title = (sec.title or '').strip()
                                        excerpt = (sec.content or '')[:500].strip()
                                        ctx = ' '.join([p for p in [title, excerpt] if p])
                                        if ctx:
                                            return ctx
                                if processed.abstract:
                                    return processed.abstract[:500]
                                return processed.full_text[:500]
                            except Exception:
                                return processed.abstract or processed.full_text[:500]

                        fig_refs: List[SynthesisFigureReference] = []
                        try:
                            for img in getattr(processed, 'images', []) or []:
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
                        except Exception:
                            fig_refs = []

                        await queue.put("event: stage\n" + "data: {\"name\": \"synthesis_start\"}\n\n")
                        # build pairs
                        chunk_map = {c.chunk_id: c for c in result.semantic_chunks}
                        pairs: List[SynthesisAnalysisPair] = []
                        for exp in result.chunk_explanations:
                            chunk = chunk_map.get(exp.chunk_id)
                            if not chunk:
                                continue
                            pairs.append(
                                SynthesisAnalysisPair(
                                    original=chunk.content,
                                    analysis=exp.explanation,
                                    section_id=chunk.section_type,
                                    confidence_score=exp.confidence_score,
                                )
                            )

                        s_global = SynthesisAnalysisGlobalAdapter(result)
                        s_meta = SynthesisPaperMetadata(
                            title=processed.metadata.title,
                            authors=processed.metadata.authors,
                            institutions=None,
                            publication_date=(processed.metadata.publication_date.isoformat() if processed.metadata.publication_date else None),
                            arxiv_id=processed.metadata.arxiv_id,
                            doi=processed.metadata.doi,
                            abstract=processed.abstract,
                        )

                        article = await synthesis.synthesize_article(
                            analysis_pairs=pairs,
                            global_analysis=s_global,
                            paper_metadata=s_meta,
                            figures=fig_refs,
                        )
                        await queue.put("event: complete\n" + f"data: {json.dumps({'article_path': article.file_path, 'title': article.title})}\n\n")
                    except Exception as e:
                        await queue.put("event: error\n" + f"data: {json.dumps({'error': str(e)})}\n\n")
                    finally:
                        await queue.put("[DONE]")

                # start pipeline
                task = asyncio.create_task(run_pipeline())

                # drain queue
                while True:
                    line = await queue.get()
                    if line == "[DONE]":
                        break
                    yield line
                await task
            except Exception as e:
                import json
                yield "event: error\n" + f"data: {json.dumps({'error': str(e)})}\n\n"

        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)

    @app.get("/drafts")
    async def list_drafts():
        await ensure_initialized()
        drafts_dir = Path(state["config"].paths.drafts_folder)
        drafts_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(drafts_dir.glob("*.md"), reverse=True)
        items = []
        for f in files[:200]:
            items.append({"name": f.name, "path": str(f), "article_id_hint": f.stem.split("_")[-1]})
        return {"items": items}

    @app.get("/drafts/{filename}")
    async def get_draft(filename: str):
        await ensure_initialized()
        path = Path(state["config"].paths.drafts_folder) / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="not found")
        try:
            text = await asyncio.to_thread(path.read_text, "utf-8")
            return {"name": filename, "content": text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Export HTML (basic markdown-to-HTML, headings/paragraphs/images)
    @app.get("/drafts/{filename}/export/html")
    async def export_html(filename: str):
        await ensure_initialized()
        md_path = Path(state["config"].paths.drafts_folder) / filename
        if not md_path.exists():
            raise HTTPException(status_code=404, detail="not found")
        text = await asyncio.to_thread(md_path.read_text, "utf-8")
        # very conservative markdown → html (headings, paragraphs, images, emphasis)
        html = text
        # strip front-matter
        html = re.sub(r"^---[\s\S]*?---\n", "", html)
        # images ![alt](src)
        html = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<figure><img src="\2" alt="\1"/><figcaption>\1</figcaption></figure>', html)
        # headings
        html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        # bold **text**
        html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)
        # paragraphs
        html = "\n".join(f"<p>{p}</p>" if not p.strip().startswith("<h") and not p.strip().startswith("<figure") else p for p in html.split("\n\n"))
        doc = (
            "<!doctype html><html><head><meta charset='utf-8'><title>"
            + filename
            + "</title><style>"
            + "body{font-family:-apple-system,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:760px;margin:40px auto;line-height:1.6;padding:0 16px}"
            + "img{max-width:100%}figure{margin:16px 0;text-align:center}figcaption{font-size:12px;color:#555}"
            + "</style></head><body>"
            + html
            + "</body></html>"
        )
        return Response(content=doc, media_type="text/html")

    # Export PDF (simple headless HTML→PDF using reportlab fallback if playwright unavailable)
    @app.get("/drafts/{filename}/export/pdf")
    async def export_pdf(filename: str):
        await ensure_initialized()
        md_path = Path(state["config"].paths.drafts_folder) / filename
        if not md_path.exists():
            raise HTTPException(status_code=404, detail="not found")
        text = await asyncio.to_thread(md_path.read_text, "utf-8")
        # naive plain-text PDF fallback
        try:
            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.pdfgen import canvas
                from reportlab.lib.units import mm
            except ImportError:
                # Graceful degradation when optional dependency missing
                raise HTTPException(status_code=501, detail="PDF export requires optional dependency 'reportlab'. Install with: pip install reportlab")

            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            width, height = A4
            x, y = 20*mm, height - 20*mm
            for line in text.splitlines():
                if y < 20*mm:
                    c.showPage(); y = height - 20*mm
                c.setFont("Helvetica", 10)
                c.drawString(x, y, line[:1200])
                y -= 12
            c.save()
            buf.seek(0)
            headers = {"Content-Disposition": f"attachment; filename={filename.replace('.md','')}.pdf"}
            return Response(content=buf.read(), media_type="application/pdf", headers=headers)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF export failed: {e}")

    @app.get("/memory/search", response_model=MemorySearchResponse)
    async def memory_search(q: str = Query(..., min_length=1), limit: int = 10):
        await ensure_initialized()
        try:
            results = await state["memory"].search_memory(
                query=q,
                memory_types=[MemoryType.EPISODIC, MemoryType.CONCEPTUAL],
                limit=limit,
                similarity_threshold=0.6,
                include_related=True,
                search_depth=2,
            )
            out = []
            for r in results:
                item = {
                    "memory_type": r.memory_type.value,
                    "similarity_score": r.similarity_score,
                    "explanation": r.explanation,
                }
                if r.memory_type == MemoryType.EPISODIC:
                    e = r.entry
                    item.update(
                        {
                            "id": e.id,
                            "content": e.content,
                            "source": e.source,
                            "context": e.context,
                            "tags": e.tags,
                            "timestamp": e.timestamp.isoformat(),
                        }
                    )
                else:
                    e = r.entry
                    item.update(
                        {
                            "id": e.id,
                            "name": e.name,
                            "node_type": e.node_type.value,
                            "attributes": e.attributes,
                        }
                    )
                out.append(item)
            return MemorySearchResponse(results=out)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    class SynthesisAnalysisGlobalAdapter(SynthesisGlobalAnalysisMap):
        def __init__(self, analysis_result: Any):
            super().__init__(
                summary=getattr(analysis_result, "overall_summary", ""),
                key_insights=list(getattr(analysis_result, "research_insights", []) or []),
                technical_terms=dict(getattr(analysis_result.global_map, "terminology_glossary", {}) or {}),
                methodology_analysis=getattr(analysis_result.global_map, "methodology_overview", ""),
                significance_assessment=", ".join(getattr(analysis_result, "practical_implications", [])[:5]) or "",
            )

    return app


app = create_app()


