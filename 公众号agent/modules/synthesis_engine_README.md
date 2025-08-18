# SynthesisEngine 模块文档

## 概述

`synthesis_engine.py` 是Deep Scholar AI系统的核心合成引擎，负责将AnalysisCore输出的解读内容合成为完整的微信公众号文章。

## 核心功能

### 1. 内容聚合
- 将AnalysisCore输出的"讲解对"按原文顺序组合
- 集成GlobalAnalysisMap和长期记忆
- 智能关联相关研究背景

### 2. 结构化生成
- **标题生成**: 基于GlobalAnalysisMap生成3-5个高水平标题
- **引言撰写**: 结合全局分析和长期记忆生成引人入胜的开篇
- **核心观点总结**: 提炼3-5个关键技术洞察
- **深度思考**: 使用最强LLM模型进行批判性分析

### 3. 微信公众号适配
- 专业术语自动加粗处理
- 避免不必要分点，确保内容流畅
- 精确的图片集成和引用
- 符合微信阅读习惯的排版优化

### 4. 智能交付
- 异步文件保存到drafts_folder
- 规范的文件命名格式
- 关联图片文件夹管理
- 完整的元数据记录

## 主要类和数据结构

### 核心类

```python
class SynthesisEngine:
    """内容合成引擎主类"""
    
    async def synthesize_article(
        self,
        analysis_pairs: List[AnalysisPair],
        global_analysis: GlobalAnalysisMap,
        paper_metadata: PaperMetadata,
        figures: Optional[List[FigureReference]] = None
    ) -> SynthesizedArticle:
        """合成完整文章"""
```

### 数据结构

```python
@dataclass
class AnalysisPair:
    """分析对数据结构"""
    original: str              # 原文片段
    analysis: str              # 深度解读
    section_id: str           # 章节标识
    confidence_score: float   # 置信度评分

@dataclass
class GlobalAnalysisMap:
    """全局分析映射"""
    summary: str                        # 论文总结
    key_insights: List[str]             # 关键洞察
    technical_terms: Dict[str, str]     # 技术术语
    methodology_analysis: str           # 方法论分析
    significance_assessment: str        # 意义评估

@dataclass
class SynthesizedArticle:
    """合成文章结果"""
    title: str                          # 文章标题
    content: str                        # 完整内容
    metadata: PaperMetadata            # 论文元数据
    figures: List[FigureReference]     # 图片引用
    word_count: int                    # 字数统计
    generation_timestamp: datetime     # 生成时间
    article_id: str                   # 文章唯一标识
```

## 使用方法

### 基本使用

```python
from modules.synthesis_engine import SynthesisEngine
from modules.config_manager import ConfigManager
from modules.llm_provider import LLMProviderService
from modules.memory_system import HybridMemorySystem

# 初始化依赖
config_manager = ConfigManager()
await config_manager.initialize('config.yaml')

llm_provider = LLMProviderService(config_manager)

memory_system = HybridMemorySystem(config_manager)
await memory_system.initialize()

# 创建合成引擎
engine = SynthesisEngine(
    config_manager=config_manager,
    llm_provider=llm_provider,
    memory_system=memory_system
)

await engine.initialize()

# 合成文章
article = await engine.synthesize_article(
    analysis_pairs=analysis_pairs,
    global_analysis=global_analysis,
    paper_metadata=paper_metadata,
    figures=figures
)

print(f"生成文章: {article.title}")
print(f"字数: {article.word_count}")
print(f"保存位置: {config_manager.paths.drafts_folder}")
```

### 高级配置

#### 自定义Jinja2模板

在 `prompts/` 目录下可以自定义以下模板：
- `synthesis_title_generation.jinja2` - 标题生成模板
- `synthesis_introduction.jinja2` - 引言生成模板
- `synthesis_key_insights.jinja2` - 核心洞察模板
- `synthesis_deep_thinking.jinja2` - 深度思考模板

#### 微信公众号格式化

```python
from modules.synthesis_engine import WeChatMarkdownFormatter

formatter = WeChatMarkdownFormatter()
formatted_content = formatter.format_content(raw_content, figures)
```

## 技术特性

### 依赖注入架构
- 严格遵循依赖注入原则
- 无副作用构造函数
- 显式异步初始化

### 模板化设计
- 外部Jinja2模板管理
- 支持模板热更新
- 灵活的内容生成策略

### 智能图片集成
- 基于语义相关性的图片插入
- 自动图片引用匹配
- 智能插入位置计算

### 生产级特性
- 完整的错误处理机制
- 详细的日志记录
- 性能指标监控
- 异步文件操作

## 配置要求

### 必需配置项

在 `config.yaml` 中需要配置：

```yaml
llm_settings:
  provider: "openai"
  primary_model: "gpt-4o"
  insight_model: "gpt-4o"  # 用于深度思考的最强模型
  temperature: 0.7
  max_tokens: 4000

paths:
  drafts_folder: "./output/drafts"
  prompt_templates_folder: "./prompts"

output:
  max_article_word_count: 5000
  min_article_word_count: 2000
  markdown_format_version: "wechat_md"
```

### 依赖包

```bash
pip install jinja2 aiofiles
```

## 输出格式

### 文章结构
1. **标题** - 学术深度与传播力并重
2. **引言** - 引人入胜的背景介绍
3. **核心洞察** - 3-5个关键技术洞察
4. **技术解读** - 按原文顺序的详细分析
5. **Deep Scholar AI深度思考** - 最高水平的批判性分析

### 文件命名规范
`YYYYMMDD_HHMMSS_[安全标题]_[文章ID前8位].md`

### 元数据头部
```yaml
---
title: 文章标题
article_id: 唯一标识
generated_at: 生成时间ISO格式
word_count: 字数统计
source: Deep Scholar AI
paper_title: 原论文标题
paper_authors: 作者列表
arxiv_id: arXiv标识
---
```

## 质量保证

### AI痕迹消除
- 严格避免"突破性"、"革命性"等夸大词汇
- 使用专业学术语言风格
- 体现顶级学者的思考深度

### 内容质量控制
- 多层次的语义理解
- 批判性思维体现
- 跨学科视角融合
- 前瞻性判断展现

## 扩展接口

模块提供了多个扩展点：
- 自定义格式化器
- 自定义模板引擎
- 自定义图片处理逻辑
- 自定义保存策略

## 注意事项

1. **模板文件必须存在**：确保所有Jinja2模板文件在prompts目录中
2. **LLM API配额**：深度思考功能使用最强模型，注意API使用量
3. **文件权限**：确保drafts_folder有写入权限
4. **依赖初始化顺序**：必须先初始化所有依赖项再创建SynthesisEngine
5. **异步上下文**：所有主要方法都是异步的，需要在async context中调用

## 故障排除

### 常见问题

1. **模板文件找不到**
   - 检查prompts文件夹路径
   - 确认所有.jinja2文件存在

2. **LLM调用失败**
   - 检查API密钥配置
   - 确认模型名称正确

3. **文件保存失败**
   - 检查目录权限
   - 确认磁盘空间充足

4. **内存系统错误**
   - 检查向量数据库连接
   - 确认嵌入服务可用
