# Deep Scholar AI - Agent 配置和说明

这个文件包含了 Deep Scholar AI 系统的配置信息和常用命令，帮助 AI Assistant 更好地理解和操作这个项目。

## 🏗️ 项目结构

```
公众号agent/
├── main.py                    # 主应用入口
├── run_revision.py           # 交互式修订工具
├── config.yaml              # 配置文件模板
├── requirements.txt          # Python依赖
├── modules/                  # 核心模块
│   ├── config_manager.py     # 配置管理
│   ├── logger.py            # 日志服务
│   ├── llm_provider.py      # LLM服务抽象
│   ├── embedding_provider.py # 嵌入服务抽象
│   ├── memory_system.py     # 记忆系统
│   ├── discovery_engine.py  # 发现引擎
│   ├── ingestion_pipeline.py # PDF处理管道
│   ├── analysis_core.py     # 深度分析核心
│   └── synthesis_engine.py  # 文章合成引擎
├── prompts/                  # Jinja2提示模板
├── tests/                    # 测试套件
├── data/                     # 数据存储
├── logs/                     # 日志文件
├── input/                    # 输入文件
└── output/                   # 输出文件
```

## 🚀 常用命令

### 应用启动
```bash
# 交互模式
python main.py --config config.yaml --mode interactive

# 每日工作流
python main.py --config config.yaml --mode daily

# 处理单篇论文
python main.py --config config.yaml --mode process --paper-url "https://arxiv.org/abs/1706.03762"
```

### 测试命令
```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_config_manager.py -v

# 运行集成测试
pytest tests/test_integration.py -v -m integration

# 测试覆盖率
pytest tests/ --cov=modules --cov-report=html
```

### 代码质量检查
```bash
# 代码格式化
black modules/ tests/ main.py

# 代码检查
flake8 modules/ tests/ main.py

# 类型检查
mypy modules/ main.py
```

### 交互式修订
```bash
# 修订单个文件
python run_revision.py --config config.yaml --file "output/drafts/article.md"

# 批量修订目录
python run_revision.py --config config.yaml --directory "output/drafts/"
```

## ⚙️ 配置说明

### 必需配置项
```yaml
api_keys:
  openai: "sk-..."              # OpenAI API密钥
  anthropic: "..."              # Anthropic API密钥（可选）
  twitter_bearer_token: "..."   # Twitter API令牌（可选）

llm_settings:
  provider: "openai"            # LLM提供商
  primary_model: "gpt-4o"       # 主要模型
  
embedding_settings:
  provider: "openai"            # 嵌入提供商
  model: "text-embedding-3-large"  # 嵌入模型
```

### 环境变量（推荐）
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
export TWITTER_BEARER_TOKEN="..."
```

## 🔧 开发和调试

### 开发模式启动
```bash
# 使用调试级别日志
python main.py --config config.yaml --mode interactive
# 配置文件中设置 logging.level: "DEBUG"
```

### 常见问题排查
1. **API密钥问题**：检查环境变量或配置文件
2. **依赖问题**：运行 `pip install -r requirements.txt`
3. **权限问题**：确保对数据目录有写权限
4. **内存不足**：减少并发配置参数

### 性能优化
```yaml
performance:
  max_concurrent_downloads: 5      # 根据网络调整
  max_concurrent_llm_calls: 3      # 根据API限制调整
  api_retry_attempts: 5            # 重试次数
```

## 📊 监控和日志

### 日志文件位置
- 主日志：`./logs/deep_scholar.log`
- 错误日志：包含在主日志中，可通过级别过滤

### 重要日志类别
- `function_calls`：函数调用跟踪
- `performance`：性能指标
- `api_calls`：API调用统计
- `paper_processing`：论文处理进度
- `memory_system`：记忆系统操作

### 监控指标
- API调用次数和成本
- 处理论文数量
- 记忆系统大小
- 错误率和重试统计

## 🧪 测试策略

### 测试分类
- **单元测试**：测试单个模块功能
- **集成测试**：测试模块间协作
- **端到端测试**：测试完整工作流

### 测试覆盖率目标
- 核心模块：>90%
- 工具类：>80%
- 总体覆盖率：>85%

## 🔄 持续集成

### 代码质量门禁
1. 所有测试必须通过
2. 代码覆盖率不低于80%
3. 类型检查无错误
4. 代码格式符合规范

### 部署流程
1. 更新版本号
2. 运行完整测试套件
3. 构建Docker镜像（如需要）
4. 部署到目标环境

## 🎯 使用建议

### 最佳实践
1. **配置管理**：使用环境变量存储敏感信息
2. **错误处理**：始终检查日志了解错误详情
3. **资源管理**：合理设置并发参数避免API限流
4. **版本控制**：定期备份配置和数据

### 性能建议
1. 使用本地嵌入模型降低API成本
2. 配置合适的缓存策略
3. 定期清理临时文件和日志
4. 监控内存和磁盘使用情况
5. 文章质量优先策略：默认“深度思考”启用洞察模型（更强），解释模板严格约束为中文散文体、禁列表/编号/小标题与机械衔接，避免AI腔。若需进一步控费，可调低 `llm_settings.insight_model` 或仅在关键段落启用最强模型。

## 📝 代码风格指南

### 命名约定
- 类名：PascalCase（如 `ConfigManager`）
- 函数名：snake_case（如 `initialize_system`）
- 常量：UPPER_SNAKE_CASE（如 `MAX_RETRIES`）
- 私有方法：前缀下划线（如 `_internal_method`）

### 文档字符串格式
使用Google风格的文档字符串：
```python
async def process_paper(self, paper_url: str) -> Optional[str]:
    """
    处理单篇论文
    
    Args:
        paper_url: 论文URL
        
    Returns:
        生成的文章文件路径，失败时返回None
        
    Raises:
        ValueError: URL格式无效
        NetworkError: 网络连接失败
    """
```

### 异常处理模式
```python
try:
    result = await risky_operation()
    logger.info("操作成功")
    return result
except SpecificError as e:
    logger.error(f"特定错误: {e}")
    raise
except Exception as e:
    logger.error(f"未知错误: {e}", exc_info=True)
    raise SystemError(f"操作失败: {e}") from e
```

这个文档会帮助 AI Assistant 更好地理解项目结构和操作方式。
