"""
Learning Loop - 持续进化与品味强化引擎

职责（最小侵入、生产可用）：
- 接收两类反馈：
  1) 人类反馈（DiscoveryEngine.provide_feedback / update_taste_from_paper 调用链）
  2) 自反馈（基于 AnalysisResult 的质量指标）
- 将反馈事件标准化落库为情节记忆（episodic），并按需补充概念记忆节点（conceptual）
- 以保守、可解释的策略更新 TasteEngine（关键词/作者权重与探索因子 exploration_factor）
- 生成阶段性“进化报告”并写入情节记忆，作为持续学习的可核查证据

严格遵循：无副作用构造函数；所有 I/O 在 initialize/方法中；
不修改现有类签名，仅在上层注入和调用，避免破坏现有流程。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from .config_manager import ConfigManager
from .embedding_provider import EmbeddingProviderService
from .memory_system import HybridMemorySystem, MemoryType
from .logger import get_logger

# 轻耦合：避免在类型层面引入循环依赖；运行时进行鸭子类型调用
# from .discovery_engine import TasteEngine, DiscoveredPaper  # 不强制导入


@dataclass
class LearningState:
    """在磁盘上持久化的学习状态（轻量、可解释）。"""
    human_positive: int = 0
    human_negative: int = 0
    self_high_quality: int = 0
    self_low_quality: int = 0
    recent_scores: List[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "human_positive": self.human_positive,
            "human_negative": self.human_negative,
            "self_high_quality": self.self_high_quality,
            "self_low_quality": self.self_low_quality,
            "recent_scores": list(self.recent_scores or []),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LearningState":
        st = LearningState()
        st.human_positive = int(data.get("human_positive", 0))
        st.human_negative = int(data.get("human_negative", 0))
        st.self_high_quality = int(data.get("self_high_quality", 0))
        st.self_low_quality = int(data.get("self_low_quality", 0))
        rs = data.get("recent_scores")
        st.recent_scores = list(rs) if isinstance(rs, list) else []
        return st


class LearningLoop:
    """
    学习循环：统一接收反馈 → 写入记忆 → 更新 TasteEngine → 调整探索因子 → 生成进化报告。

    依赖：
    - config: ConfigManager（反馈权重、日志路径等）
    - memory_system: HybridMemorySystem（持久化学习事件/概念）
    - embedding_service: EmbeddingProviderService（必要时生成概念节点嵌入）
    - taste_engine: DiscoveryEngine 内部的 TasteEngine（可为 None，运行时再赋值）
    """

    def __init__(
        self,
        config: ConfigManager,
        memory_system: HybridMemorySystem,
        embedding_service: EmbeddingProviderService,
        taste_engine: Optional[object] = None,
    ) -> None:
        if not config.is_initialized():
            raise ValueError("ConfigManager must be initialized")

        self.config = config
        self.memory_system = memory_system
        self.embedding_service = embedding_service
        self.taste_engine = taste_engine  # 运行时注入 DiscoveryEngine.taste_engine

        self.logger = get_logger("learning_loop")
        self._initialized = False
        self._state: LearningState = LearningState()
        self._state_file: Path = (
            Path(self.config.paths.logs_folder) / "learning_state.json"
        )
        # 是否启用学习闭环
        self._enabled: bool = True

        # 学习参数（可由 config.learning 覆盖）
        try:
            lc = getattr(self.config, "learning", None)
        except Exception:
            lc = None
        self._epsilon_min = float(getattr(lc, "epsilon_min", 0.05) or 0.05)
        self._epsilon_max = float(getattr(lc, "epsilon_max", 0.30) or 0.30)
        self._epsilon_step = float(getattr(lc, "epsilon_step", 0.02) or 0.02)
        self._quality_threshold_good = float(getattr(lc, "quality_threshold_good", 0.70) or 0.70)
        self._recent_window = int(getattr(lc, "recent_window", 100) or 100)

    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            await self._load_state()
            try:
                lc = getattr(self.config, "learning", None)
                if lc is not None and hasattr(lc, "enable"):
                    self._enabled = bool(getattr(lc, "enable"))
            except Exception:
                self._enabled = True
            self._initialized = True
            self.logger.info("LearningLoop initialized")
        except Exception as e:
            self.logger.error(f"LearningLoop initialize failed: {e}")
            raise

    async def _load_state(self) -> None:
        try:
            if self._state_file.exists():
                content = await asyncio.to_thread(self._state_file.read_text, "utf-8")
                data = json.loads(content)
                self._state = LearningState.from_dict(data)
            else:
                self._state.recent_scores = []
        except Exception as e:
            self.logger.warning(f"Failed to load learning state, using defaults: {e}")
            self._state = LearningState()
            self._state.recent_scores = []

    async def _save_state(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self._state.to_dict(), ensure_ascii=False, indent=2)
            # 原子写入：写入临时文件后原子替换
            tmp_path = self._state_file.with_suffix(self._state_file.suffix + ".tmp")
            await asyncio.to_thread(tmp_path.write_text, payload, "utf-8")
            await asyncio.to_thread(tmp_path.replace, self._state_file)
        except Exception as e:
            self.logger.warning(f"Failed to save learning state: {e}")

    def set_taste_engine(self, taste_engine: object) -> None:
        """在 DiscoveryEngine 初始化后注入 TasteEngine。"""
        self.taste_engine = taste_engine

    # -------------------- 人类反馈路径 --------------------
    async def record_human_feedback(self, paper: object, positive: bool) -> None:
        """
        记录人类反馈事件：
        - 将事件写入情节记忆（含作者与基本元信息）
        - 温和地调整探索因子（正反馈略降探索，负反馈略升探索）
        - 关键字/作者权重的强力更新交由 TasteEngine.learn_from_feedback 完成；
          此函数只做附加的持久化与探索因子调节，避免重复强化。
        """
        try:
            if not self._enabled:
                return
            # 1) 写入记忆（尽量从 DiscoveredPaper 结构读取字段，鸭子类型）
            title = self._safe_get(paper, ["metadata", "title"]) or "Unknown Paper"
            arxiv_id = self._safe_get(paper, ["metadata", "arxiv_id"]) or None
            doi = self._safe_get(paper, ["metadata", "doi"]) or None
            authors = []
            authors_obj = self._safe_get(paper, ["metadata", "authors"]) or []
            try:
                for a in authors_obj:
                    name = getattr(a, "name", None) if not isinstance(a, str) else a
                    if name:
                        authors.append(name)
            except Exception:
                pass

            await self.memory_system.store_episodic_memory(
                content=f"Human feedback: {'positive' if positive else 'negative'} for '{title}'",
                source="human_feedback",
                context="discovery_feedback",
                tags=["feedback_event", "human", "paper"],
                metadata={
                    "title": title,
                    "arxiv_id": arxiv_id,
                    "doi": doi,
                    "authors": authors,
                    "feedback_positive": positive,
                },
                importance_score=1.0 if positive else 0.6,
            )

            # 2) 更新内部学习状态与探索因子
            if positive:
                self._state.human_positive += 1
                await self._adjust_exploration(decrease=True)
            else:
                self._state.human_negative += 1
                await self._adjust_exploration(decrease=False)

            await self._save_state()

        except Exception as e:
            self.logger.warning(f"record_human_feedback failed: {e}")

    # -------------------- 自反馈路径（基于分析质量） --------------------
    async def record_self_evaluation_from_analysis(self, analysis_result: object, authors: Optional[List[str]] = None) -> None:
        """
        基于 AnalysisResult 的质量指标进行自反馈学习：
        - 将质量评分写入情节记忆
        - 根据质量好坏调整 TasteEngine 的关键词/作者权重（保守幅度）
        - 动态调整探索因子（好→降探索，差→升探索）
        """
        try:
            if not self._enabled:
                return
            metrics = getattr(analysis_result, "quality_metrics", {}) or {}
            global_map = getattr(analysis_result, "global_map", None)
            overall_quality = float(metrics.get("overall_quality", 0.0) or 0.0)
            confidence = float(metrics.get("confidence", 0.0) or 0.0)

            related_concepts: List[str] = []
            try:
                related_concepts = list(getattr(global_map, "related_concepts", []) or [])
            except Exception:
                related_concepts = []

            title = getattr(global_map, "title", "Unknown Paper") if global_map else "Unknown Paper"
            author_list = authors or (getattr(global_map, "authors", None) or [])

            await self.memory_system.store_episodic_memory(
                content=(
                    f"Self-evaluation for '{title}': overall_quality={overall_quality:.3f},"
                    f" confidence={confidence:.3f}"
                ),
                source="self_evaluation",
                context=f"paper_id:{getattr(analysis_result, 'paper_id', '')}",
                tags=["feedback_event", "self", "analysis_quality"],
                metadata={
                    "overall_quality": overall_quality,
                    "confidence": confidence,
                    "related_concepts": related_concepts[:10],
                    "authors": author_list[:10] if isinstance(author_list, list) else author_list,
                },
                importance_score=max(overall_quality, 0.5),
            )

            # 更新内部状态
            if self._state.recent_scores is None:
                self._state.recent_scores = []
            self._state.recent_scores.append(overall_quality)
            if len(self._state.recent_scores) > self._recent_window:
                self._state.recent_scores = self._state.recent_scores[-self._recent_window :]

            is_good = overall_quality >= self._quality_threshold_good
            if is_good:
                self._state.self_high_quality += 1
            else:
                self._state.self_low_quality += 1

            # 关键词/作者权重的保守更新
            await self._reinforce_taste_from_self_eval(
                related_concepts=related_concepts,
                authors=author_list,
                positive=is_good,
            )

            # 调整探索因子
            await self._adjust_exploration(decrease=is_good)
            await self._save_state()
        except Exception as e:
            self.logger.warning(f"record_self_evaluation_from_analysis failed: {e}")

    # -------------------- 报告与导出 --------------------
    async def generate_progress_report(self) -> Dict[str, Any]:
        """生成进化报告并写入情节记忆，返回报告字典。"""
        try:
            if not self._enabled:
                return {"disabled": True}
            taste_summary = None
            try:
                if self.taste_engine and hasattr(self.taste_engine, "get_taste_summary"):
                    taste_summary = await self.taste_engine.get_taste_summary()
            except Exception:
                taste_summary = None

            memory_stats = self.memory_system.get_memory_stats() if self.memory_system else {}

            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "learning_state": self._state.to_dict(),
                "taste_summary": taste_summary,
                "memory_stats": memory_stats,
            }

            await self.memory_system.store_episodic_memory(
                content="Weekly learning progress report",
                source="learning_report",
                context="progress_report",
                tags=["learning", "report"],
                metadata=report,
                importance_score=0.9,
            )
            return report
        except Exception as e:
            self.logger.warning(f"generate_progress_report failed: {e}")
            return {"error": str(e)}

    async def get_learning_summary(self) -> Dict[str, Any]:
        """返回学习状态摘要（不落库）。"""
        try:
            tv = getattr(self.taste_engine, "taste_vector", None) if self.taste_engine else None
            epsilon = getattr(tv, "exploration_factor", None) if tv else None
            return {
                "enabled": self._enabled,
                "human_positive": self._state.human_positive,
                "human_negative": self._state.human_negative,
                "self_high_quality": self._state.self_high_quality,
                "self_low_quality": self._state.self_low_quality,
                "recent_scores_count": len(self._state.recent_scores or []),
                "exploration_factor": epsilon,
            }
        except Exception as e:
            return {"error": str(e)}

    # -------------------- 内部工具 --------------------
    async def _reinforce_taste_from_self_eval(
        self,
        related_concepts: List[str],
        authors: Optional[List[str]],
        positive: bool,
    ) -> None:
        """
        基于自评质量对关键词/作者进行温和的权重调整：
        - 正向：related_concepts 与 authors 权重 +positive_weight
        - 负向：权重 -negative_weight（下限 0.1）
        - 将 concepts 合理加入 keywords 列表（若不存在）
        """
        try:
            if not self.taste_engine:
                return
            tv = getattr(self.taste_engine, "taste_vector", None)
            if not tv:
                return

            pos_delta = float(getattr(self.config.feedback, "positive_weight_increase", 0.2))
            neg_delta = float(getattr(self.config.feedback, "negative_weight_decrease", 0.05))

            # 关键词（来自 concepts）
            for kw in (related_concepts or [])[:10]:
                if not isinstance(kw, str) or not kw.strip():
                    continue
                if kw not in tv.keywords:
                    tv.keywords.append(kw)
                    tv.weights[kw] = 1.0
                else:
                    current = tv.weights.get(kw, 1.0)
                    tv.weights[kw] = max(0.1, current + (pos_delta if positive else -neg_delta))

            # 作者偏好
            if isinstance(authors, list):
                for name in authors[:8]:
                    if not name:
                        continue
                    key = f"author:{name}"
                    current = tv.weights.get(key, 1.0)
                    tv.weights[key] = max(0.1, current + (pos_delta if positive else -neg_delta))

            # 持久化 TasteVector（使用公共接口 save()）
            try:
                if hasattr(self.taste_engine, "save") and callable(getattr(self.taste_engine, "save")):
                    await self.taste_engine.save()  # type: ignore[attr-defined]
                elif hasattr(self.taste_engine, "_save_taste_vector"):
                    # 兼容旧实现
                    await self.taste_engine._save_taste_vector()  # type: ignore[attr-defined]
            except Exception as _e:
                self.logger.warning(f"TasteEngine save failed: {_e}")
        except Exception as e:
            self.logger.warning(f"_reinforce_taste_from_self_eval failed: {e}")

    async def _adjust_exploration(self, decrease: bool) -> None:
        """
        按反馈信号调节探索因子 epsilon：
        - 正反馈：epsilon 下降一个 step（保底 _epsilon_min）
        - 负反馈：epsilon 上升一个 step（封顶 _epsilon_max）
        """
        try:
            if not self.taste_engine:
                return
            tv = getattr(self.taste_engine, "taste_vector", None)
            if not tv:
                return
            epsilon = float(getattr(tv, "exploration_factor", 0.1) or 0.1)
            if decrease:
                epsilon = max(self._epsilon_min, epsilon - self._epsilon_step)
            else:
                epsilon = min(self._epsilon_max, epsilon + self._epsilon_step)
            tv.exploration_factor = epsilon
            try:
                if hasattr(self.taste_engine, "save") and callable(getattr(self.taste_engine, "save")):
                    await self.taste_engine.save()  # type: ignore[attr-defined]
                elif hasattr(self.taste_engine, "_save_taste_vector"):
                    await self.taste_engine._save_taste_vector()  # type: ignore[attr-defined]
            except Exception as _e:
                self.logger.warning(f"TasteEngine save failed: {_e}")
        except Exception as e:
            self.logger.warning(f"_adjust_exploration failed: {e}")

    @staticmethod
    def _safe_get(obj: object, path: List[str]) -> Any:
        cur = obj
        try:
            for key in path:
                if cur is None:
                    return None
                if isinstance(cur, dict):
                    cur = cur.get(key)
                else:
                    cur = getattr(cur, key)
            return cur
        except Exception:
            return None

    async def close(self) -> None:
        try:
            await self._save_state()
            self.logger.info("LearningLoop closed")
        except Exception:
            pass


