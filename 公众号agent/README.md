## Deep Scholar AI（微信公众号学术助手）——实事求是的 README（2025）

本仓库是一个可本地运行的“虚拟 AI 研究员”原型：从公开来源发现论文、解析 PDF、进行语义分析与文章合成，并输出适配微信公众号风格的 Markdown 草稿。提供命令行工作流、可选的 HTTP API 以及一个本地前端开发面板（便于演示与调试）。

本文档如实说明能力与限制，帮助你可靠跑通并进行二次开发。

---

## 架构与能力（实际可用）

- 后端核心模块（`modules/`）
  - 发现引擎：ArXiv/会议页/Twitter(X)/本地文件夹监听（缺失依赖时自动禁用对应源）
  - PDF 摄取：PyMuPDF（默认）与 Nougat（可选）双路径解析，混合策略合并文本/结构/图片
  - 深度分析：生成 GlobalAnalysisMap，语义切分（SOTA 嵌入 + 层次聚类），逐意群 JSON 解释
  - 记忆系统：ChromaDB（情节记忆）+ NetworkX（概念记忆，多重边保真 GPickle 主存，GraphML 可选导出）
  - 文章合成：Jinja2 模板 + LLM；提供语义驱动的图片插入与微信公众号格式化器
- HTTP API（FastAPI，见 `modules/api_server.py`）
  - 初始化、发现、处理单篇、草稿列表/导出（HTML/PDF）、混合记忆检索（详见“API 端点”）
- 前端开发面板（Vite + React，`frontend/`）
  - 通过本地代理调用后端 API：初始化 → 发现 → 处理 → 预览/导出 → 记忆检索
- CLI 工作流（`main.py`）
  - 交互模式、单篇处理、每日工作流；带完善日志与异常兜底

---

## 不是什么（避免误解）

- 非“完全无人值守”的生产系统：上游网页结构与供应商 API 变化会影响稳定性，需要根据日志调参
- 不保证学术正确性/时效性：LLM 会产生错误，发布前请审稿
- 非零成本：外部 API 会产生费用；并发/重试已可配，但不对供应商配额作保证

---

## 环境与依赖

- Python 3.8+，推荐 macOS/Linux（Windows 需留意事件循环与依赖兼容）
- 必需：`openai`（或切换到本地嵌入/LLM 方案）、`chromadb`、`networkx`、`fastapi`、`uvicorn`
- 可选：
  - Nougat OCR（`transformers`, `torch`）用于高质量 PDF 解析
  - `watchdog`（手动摄取源）
  - `reportlab`（PDF 导出；未安装时 API 返回 501）

安装：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

API Key（建议环境变量）：

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."      # 可选
export GOOGLE_GEMINI_API_KEY="..."  # 可选
export TWITTER_BEARER_TOKEN="..."   # 若需社媒源
```

---

## 配置（`config.yaml` 关键项）

- `llm_settings.provider` 与 `primary_model`/`insight_model`
- `embedding_settings.provider` 与 `model`
- `vector_db.path`（ChromaDB 持久化目录）
- `knowledge_graph.file_path`（将生成 `<file>.gpickle` 主存，GraphML 可选导出）
- `paths.*` 输出、日志、临时下载、模板目录
- `performance.*` 并发/重试；`learning.*` 学习回路；`logging.*` 结构化日志

---

## 运行方式

1) 命令行工作流（后端一体化）

```bash
python main.py --config config.yaml --mode interactive
python main.py --config config.yaml --mode process --paper-url "https://arxiv.org/abs/1706.03762"
python main.py --config config.yaml --mode process --paper-url "file:///abs/path/to/paper.pdf"
python main.py --config config.yaml --mode daily
```

2) 启动 HTTP API（默认 127.0.0.1:8000）

```bash
python run_server.py
# 初始化：POST /initialize {"config_path":"config.yaml"}
```

3) 前端开发面板（Vite Dev Server，端口 5173，已配置反向代理 `/api` → 后端）

```bash
cd frontend
npm install
npm run dev
# 浏览器打开 http://localhost:5173
```

---

## API 端点（简要）

- POST `/initialize` 初始化服务
- GET  `/stats` 系统统计（记忆/发现/嵌入提供商信息）
- POST `/discover?limit=10` 发现论文（ArXiv/会议/社媒/本地，返回元数据）
- GET  `/sse/discover` 发现任务 SSE 流
- POST `/process` `{ paper_url }` 处理单篇（下载→解析→分析→记忆→合成）
- GET  `/sse/process?paper_url=...` 单篇处理 SSE 流
- GET  `/drafts` 草稿列表
- GET  `/drafts/{name}` 读取草稿内容
- GET  `/drafts/{name}/export/html` 导出 HTML（轻量 Markdown→HTML）
- GET  `/drafts/{name}/export/pdf` 导出 PDF（需安装 `reportlab`，否则返回 501）
- GET  `/memory/search?q=...&limit=10` 混合记忆检索（情节+概念，含相似度与简要说明）

注：HTML 导出使用了保守的 Markdown 正则转换（标题/加粗/图片/段落），避免复杂依赖；仅覆盖展示需要。

---

## 数据与持久化

- 草稿：`output/drafts/`（文件名含时间戳与随机 ID），配图复制到相邻文件夹
- 日志：`logs/deep_scholar.log`（结构化日志，易过滤）
- 临时下载：`data/temp_downloads/`
- 记忆系统：
  - ChromaDB：`vector_db.path` 下持久化（collection 名见配置）
  - NetworkX（知识图谱）：`<file>.gpickle` 主存 + 可选 `<file>.graphml` 导出；节点嵌入缓存 `<file>.embeddings.pkl`

---

## 质量与测试

```bash
pytest -v
pytest --cov=modules --cov-report=term-missing
black modules/ tests/ main.py
flake8 modules/ tests/ main.py
```

测试覆盖核心路径：配置加载、LLM/Embedding 封装、集成工作流与若干数据结构/格式化逻辑。

---

## 限制与回退策略（重要）

- 会议页解析受网页结构影响；失败不影响其他源
- Nougat 模型耗时与显存较高；环境不合适时仅启用 PyMuPDF 路径
- LLM JSON 解析：先剥离 ``` 围栏，宽松解析失败则降级为保守结果并记录日志
- 嵌入/检索：优先使用 cosine；版本差异较大时使用兼容分支（如 AgglomerativeClustering 的 `precomputed` 回退）
- PDF 导出：未安装 `reportlab` 时返回 501，避免误导

---

## 变更与修复（本次更新要点）

- 修复：API HTML 导出中的 Markdown 正则转义错误，确保图片与加粗识别正确
- 加强：PDF 导出在缺失 `reportlab` 时返回 501（明确可选依赖），而不是笼统 500
- 校准：README 修正为“提供 HTTP API 与前端开发面板”，与代码现状一致

---

## 许可证

MIT（如仓库没有 LICENSE 文件，请在使用前确认条款）。

致谢：OpenAI/Anthropic/Gemini SDK、ChromaDB、NetworkX、PyMuPDF、Nougat/transformers、scikit-learn、BeautifulSoup、watchdog、feedparser、Vite/React 等。

如遇问题，优先查看日志与配置；若为第三方 API/网页结构变更，请调整对应模块实现。
