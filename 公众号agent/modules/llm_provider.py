"""
LLM服务抽象层

提供统一的、可配置的LLM接口，支持多种后端：
- OpenAI (GPT-4, GPT-3.5等)
- Google Gemini
- Anthropic Claude

实现健壮的重试机制和错误处理，绝不返回随机或无意义的值。
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, AsyncGenerator
from enum import Enum
import aiohttp
import json
import random
from contextlib import asynccontextmanager

from .config_manager import ConfigManager
from .logger import get_logger, log_api_call, log_performance


class LLMProvider(Enum):
    """LLM提供商枚举"""
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"


@dataclass
class LLMRequest:
    """LLM请求数据类"""
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    stream: bool = False
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """LLM响应数据类"""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class LLMError(Exception):
    """LLM相关错误基类"""
    pass


class LLMAPIError(LLMError):
    """API调用错误"""
    def __init__(self, message: str, status_code: Optional[int] = None, provider: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider


class LLMRateLimitError(LLMAPIError):
    """API限流错误"""
    def __init__(self, message: str, retry_after: Optional[int] = None, provider: Optional[str] = None):
        super().__init__(message, 429, provider)
        self.retry_after = retry_after


class LLMAuthenticationError(LLMAPIError):
    """认证错误"""
    def __init__(self, message: str, provider: Optional[str] = None):
        super().__init__(message, 401, provider)


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.logger = get_logger(f"llm_client_{self.__class__.__name__.lower()}")
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """生成回复"""
        pass
    
    @abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """流式生成回复"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """获取提供商名称"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI客户端（支持自定义 base_url 以对接 OpenAI 兼容中转服务）"""

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None):
        super().__init__(api_key, model)
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=aiohttp.ClientTimeout(total=300)  # 5分钟超时
            )
        return self.session

    def __del__(self):
        try:
            if getattr(self, "session", None) and not self.session.closed:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.session.close())
                    else:
                        loop.run_until_complete(self.session.close())
                except Exception:
                    try:
                        asyncio.run(self.session.close())
                    except Exception:
                        pass
        except Exception:
            pass
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """生成回复"""
        start_time = time.time()
        
        # 构建请求数据
        messages = request.messages.copy()
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})
        
        payload = {
            "model": request.model or self.model,
            "messages": messages,
            "temperature": request.temperature or 0.7,
            "max_tokens": request.max_tokens or 4000,
            "stream": False
        }
        
        # 添加额外参数
        if request.additional_params:
            payload.update(request.additional_params)
        
        session = await self._get_session()
        
        try:
            async with session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    await self._handle_error_response(response.status, response_data)
                
                # 解析响应
                choice = response_data["choices"][0]
                content = choice["message"]["content"]
                
                # 计算token使用量和成本
                usage = response_data.get("usage", {})
                tokens_used = usage.get("total_tokens")
                cost_usd = self._calculate_cost(payload["model"], usage) if usage else None
                
                # 记录API调用
                duration = time.time() - start_time
                log_api_call(
                    provider="openai",
                    model=payload["model"],
                    tokens_used=tokens_used,
                    cost=cost_usd,
                    success=True
                )
                log_performance("openai_generation", duration, tokens=tokens_used)
                
                return LLMResponse(
                    content=content,
                    model=payload["model"],
                    provider="openai",
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    finish_reason=choice.get("finish_reason"),
                    raw_response=response_data
                )
                
        except Exception as e:
            log_api_call(
                provider="openai",
                model=payload["model"],
                success=False,
                error_message=str(e)
            )
            raise
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """流式生成回复"""
        # 构建请求数据
        messages = request.messages.copy()
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})

        payload = {
            "model": request.model or self.model,
            "messages": messages,
            "temperature": request.temperature or 0.7,
            "max_tokens": request.max_tokens or 4000,
            "stream": True
        }

        session = await self._get_session()

        try:
            async with session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                if response.status != 200:
                    error_data = await response.json()
                    await self._handle_error_response(response.status, error_data)

                # 稳健 SSE：逐行解析 event/data，兼容 CRLF，严格累积 data: 片段
                buffer = ""
                data_accumulator: list[str] = []
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue
                    try:
                        buffer += chunk.decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                    # 逐行消费（兼容 \r\n 与 \n）
                    while True:
                        newline_index = buffer.find("\n")
                        crlf_index = buffer.find("\r\n")
                        # 选择最早出现的换行
                        if crlf_index != -1 and (newline_index == -1 or crlf_index < newline_index):
                            line = buffer[:crlf_index]
                            buffer = buffer[crlf_index + 2:]
                        elif newline_index != -1:
                            line = buffer[:newline_index]
                            buffer = buffer[newline_index + 1:]
                        else:
                            break

                        line = line.strip()
                        if not line:
                            # 遇到空行：尝试提交累积的 data
                            if data_accumulator:
                                payload_str = "".join(data_accumulator).strip()
                                data_accumulator.clear()
                                if payload_str == "[DONE]":
                                    return
                                try:
                                    data_obj = json.loads(payload_str)
                                    choices = data_obj.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content_piece = delta.get("content")
                                        if content_piece:
                                            yield content_piece
                                except Exception:
                                    # 严格不抛，继续累积后续块
                                    pass
                            continue

                        if line.startswith("data:"):
                            piece = line[5:].strip()
                            if piece:
                                data_accumulator.append(piece)

        except Exception as e:
            self.logger.error(f"Stream generation failed: {e}")
            raise
    
    async def _handle_error_response(self, status_code: int, response_data: Dict[str, Any]) -> None:
        """处理错误响应"""
        error_info = response_data.get("error", {})
        error_message = error_info.get("message", "Unknown error")
        error_code = error_info.get("code")
        
        if status_code == 401:
            raise LLMAuthenticationError(f"OpenAI authentication failed: {error_message}", "openai")
        elif status_code == 429:
            retry_after = int(response_data.get("retry_after", 60))
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {error_message}", retry_after, "openai")
        else:
            raise LLMAPIError(f"OpenAI API error: {error_message} (code: {error_code})", status_code, "openai")
    
    def _calculate_cost(self, model: str, usage: Dict[str, Any]) -> float:
        """计算调用成本（美元）"""
        # OpenAI定价表（2024年价格，每1000 tokens）
        pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }
        
        if model not in pricing:
            return 0.0
        
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        input_cost = (input_tokens / 1000) * pricing[model]["input"]
        output_cost = (output_tokens / 1000) * pricing[model]["output"]
        
        return input_cost + output_cost
    
    def get_provider_name(self) -> str:
        return "openai"
    
    async def close(self):
        """关闭会话"""
        if self.session and not self.session.closed:
            await self.session.close()


class GoogleClient(BaseLLMClient):
    """Google Gemini 客户端（google-generativeai）"""

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        super().__init__(api_key, model)
        self._configured = False
        self._model_instance = None

    def _ensure_client(self, system_instruction: Optional[str] = None):
        if not self._configured:
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise LLMAPIError("google-generativeai library not installed. Please install: pip install google-generativeai", provider="google") from e
            genai.configure(api_key=self.api_key)
            self._genai = genai
            self._configured = True
        # 每次根据 system_instruction 创建模型实例，保证系统提示生效
        try:
            self._model_instance = self._genai.GenerativeModel(model_name=self.model, system_instruction=system_instruction) if system_instruction else self._genai.GenerativeModel(model_name=self.model)
        except Exception as e:
            raise LLMAPIError(f"Failed to create Gemini model: {e}", provider="google")

    def _to_gemini_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, object]]:
        """将通用 messages 转为 Gemini 历史格式。"""
        history = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                # system 不放入 history，作为系统指令处理
                continue
            history.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": content}]
            })
        return history

    async def generate(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        self._ensure_client(system_instruction=request.system_prompt)

        history = self._to_gemini_history(request.messages)
        generation_config = {
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "max_output_tokens": request.max_tokens if request.max_tokens is not None else 4000,
        }

        try:
            # 使用内容历史直接生成（单轮）
            response = await asyncio.to_thread(self._model_instance.generate_content, history, generation_config=generation_config)
            # 提取文本
            content_text = ""
            try:
                # 0.3.0 版本返回对象含有 .text
                content_text = getattr(response, "text", None) or ""
                if not content_text and hasattr(response, "candidates"):
                    # 兼容性提取
                    for c in response.candidates:
                        if hasattr(c, "content") and getattr(c.content, "parts", None):
                            for p in c.content.parts:
                                t = getattr(p, "text", "")
                                if t:
                                    content_text += t
            except Exception:
                content_text = str(response)

            duration = time.time() - start_time
            log_api_call(provider="google", model=self.model, tokens_used=None, cost=None, success=True)
            log_performance("google_generation", duration)

            return LLMResponse(
                content=content_text,
                model=self.model,
                provider="google",
                tokens_used=None,
                cost_usd=None,
                finish_reason=None,
                raw_response=None,
            )
        except Exception as e:
            log_api_call(provider="google", model=self.model, success=False, error_message=str(e))
            raise

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        self._ensure_client(system_instruction=request.system_prompt)
        history = self._to_gemini_history(request.messages)
        generation_config = {
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "max_output_tokens": request.max_tokens if request.max_tokens is not None else 4000,
        }
        try:
            # 使用线程池避免阻塞事件循环
            def _stream():
                return self._model_instance.generate_content(history, generation_config=generation_config, stream=True)

            stream = await asyncio.to_thread(_stream)
            for chunk in stream:
                try:
                    txt = getattr(chunk, "text", None)
                    if txt:
                        yield txt
                except Exception:
                    continue
        except Exception as e:
            self.logger.error(f"Google streaming failed: {e}")
            raise

    def get_provider_name(self) -> str:
        return "google"

    async def close(self):
        # google-generativeai 无持久连接，无需关闭
        return


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude客户端"""
    
    BASE_URL = "https://api.anthropic.com/v1"
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        super().__init__(api_key, model)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                timeout=aiohttp.ClientTimeout(total=300)
            )
        return self.session

    def __del__(self):
        try:
            if getattr(self, "session", None) and not self.session.closed:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.session.close())
                    else:
                        loop.run_until_complete(self.session.close())
                except Exception:
                    try:
                        asyncio.run(self.session.close())
                    except Exception:
                        pass
        except Exception:
            pass
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """生成回复"""
        start_time = time.time()
        
        # 转换消息格式
        messages = []
        system_message = None
        
        for msg in request.messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                messages.append(msg)
        
        if request.system_prompt:
            system_message = request.system_prompt
        
        payload = {
            "model": request.model or self.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4000,
            "temperature": request.temperature or 0.7
        }
        
        if system_message:
            payload["system"] = system_message
        
        session = await self._get_session()
        
        try:
            async with session.post(f"{self.BASE_URL}/messages", json=payload) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    await self._handle_error_response(response.status, response_data)
                
                content = response_data["content"][0]["text"]
                
                # 计算使用量
                usage = response_data.get("usage", {})
                tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                cost_usd = self._calculate_cost(payload["model"], usage) if usage else None
                
                duration = time.time() - start_time
                log_api_call(
                    provider="anthropic",
                    model=payload["model"],
                    tokens_used=tokens_used,
                    cost=cost_usd,
                    success=True
                )
                log_performance("anthropic_generation", duration, tokens=tokens_used)
                
                return LLMResponse(
                    content=content,
                    model=payload["model"],
                    provider="anthropic",
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    finish_reason=response_data.get("stop_reason"),
                    raw_response=response_data
                )
                
        except Exception as e:
            log_api_call(
                provider="anthropic",
                model=payload["model"],
                success=False,
                error_message=str(e)
            )
            raise
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """流式生成回复（基于 Anthropic Messages SSE）。"""
        session = await self._get_session()

        # 转换消息与系统指令
        messages = []
        system_message = None
        for msg in request.messages:
            if msg.get("role") == "system":
                system_message = msg.get("content")
            else:
                messages.append(msg)
        if request.system_prompt:
            system_message = request.system_prompt

        payload: Dict[str, Any] = {
            "model": request.model or self.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4000,
            "temperature": request.temperature or 0.7,
            "stream": True,
        }
        if system_message:
            payload["system"] = system_message

        try:
            async with session.post(f"{self.BASE_URL}/messages", json=payload) as response:
                if response.status != 200:
                    error_data = await response.json()
                    await self._handle_error_response(response.status, error_data)

                # SSE 解析（兼容 CRLF）：逐行累积 data: 片段并按事件边界提交
                buffer = ""
                current_event = None
                data_accumulator: list[str] = []
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue
                    try:
                        buffer += chunk.decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                    while True:
                        # 支持 CRLF 与 LF
                        crlf_idx = buffer.find("\r\n")
                        lf_idx = buffer.find("\n")
                        if crlf_idx != -1 and (lf_idx == -1 or crlf_idx < lf_idx):
                            line = buffer[:crlf_idx]
                            buffer = buffer[crlf_idx + 2:]
                        elif lf_idx != -1:
                            line = buffer[:lf_idx]
                            buffer = buffer[lf_idx + 1:]
                        else:
                            break

                        line = line.strip()
                        if not line:
                            # 事件块结束，尝试提交 data 累积
                            if data_accumulator:
                                payload_str = "".join(data_accumulator).strip()
                                data_accumulator.clear()
                                if payload_str == "[DONE]":
                                    return
                                try:
                                    data_obj = json.loads(payload_str)
                                    event_type = current_event or data_obj.get("type")
                                    if event_type in ("content_block_delta", "message_delta"):
                                        delta = data_obj.get("delta") or {}
                                        text_piece = delta.get("text")
                                        if text_piece:
                                            yield text_piece
                                    elif event_type == "message_stop":
                                        return
                                except Exception:
                                    pass
                            continue

                        if line.startswith("event:"):
                            current_event = line.split(":", 1)[1].strip()
                            continue
                        if line.startswith("data:"):
                            piece = line[5:].strip()
                            if piece:
                                data_accumulator.append(piece)
                            continue
        except Exception as e:
            self.logger.error(f"Anthropic streaming failed: {e}")
            raise
    
    async def _handle_error_response(self, status_code: int, response_data: Dict[str, Any]) -> None:
        """处理错误响应"""
        error_info = response_data.get("error", {})
        error_message = error_info.get("message", "Unknown error")
        error_type = error_info.get("type")
        
        if status_code == 401:
            raise LLMAuthenticationError(f"Anthropic authentication failed: {error_message}", "anthropic")
        elif status_code == 429:
            raise LLMRateLimitError(f"Anthropic rate limit exceeded: {error_message}", None, "anthropic")
        else:
            raise LLMAPIError(f"Anthropic API error: {error_message} (type: {error_type})", status_code, "anthropic")
    
    def _calculate_cost(self, model: str, usage: Dict[str, Any]) -> float:
        """计算调用成本（美元）"""
        # Anthropic定价表（2024年价格，每1000 tokens）
        pricing = {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        }
        
        if model not in pricing:
            return 0.0
        
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        input_cost = (input_tokens / 1000) * pricing[model]["input"]
        output_cost = (output_tokens / 1000) * pricing[model]["output"]
        
        return input_cost + output_cost
    
    def get_provider_name(self) -> str:
        return "anthropic"
    
    async def close(self):
        """关闭会话"""
        if self.session and not self.session.closed:
            await self.session.close()


class LLMProviderService:
    """
    LLM提供商服务
    
    统一的LLM接口，支持多种后端，实现健壮的重试机制和错误处理。
    作为依赖注入到需要LLM服务的模块中。
    """
    
    def __init__(self, config: ConfigManager):
        """
        初始化LLM提供商服务
        
        Args:
            config: 配置管理器
        """
        if not config.is_initialized():
            raise ValueError("ConfigManager must be initialized")
        
        self.config = config
        self.logger = get_logger("llm_provider_service")
        self.clients: Dict[str, BaseLLMClient] = {}
        self.primary_client: Optional[BaseLLMClient] = None
        self.insight_client: Optional[BaseLLMClient] = None
        
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """初始化LLM客户端"""
        llm_config = self.config.llm_settings
        
        # 初始化主要客户端
        primary_provider = llm_config.provider.lower()
        primary_model = llm_config.primary_model
        
        if primary_provider == "openai":
            api_key = self.config.get_api_key("openai") or self.config.get_api_key("gemini") or self.config.get_api_key("google_gemini")
            if api_key:
                # 可选自定义 base_url（OpenAI 兼容中转）
                base_url = None
                try:
                    base_url = (
                        getattr(getattr(self.config, "llm_settings", None), "base_url", None)
                        or getattr(getattr(self.config, "embedding_settings", None), "base_url", None)
                    )
                except Exception:
                    base_url = None
                self.primary_client = OpenAIClient(api_key, primary_model, base_url=base_url)
                self.clients["openai"] = self.primary_client
        elif primary_provider == "google":
            api_key = self.config.get_api_key("google") or self.config.get_api_key("google_gemini")
            if not api_key:
                raise ValueError("Google Gemini API key is required for Google provider")
            self.primary_client = GoogleClient(api_key, primary_model)
            self.clients["google"] = self.primary_client
        elif primary_provider == "anthropic":
            api_key = self.config.get_api_key("anthropic")
            if api_key:
                self.primary_client = AnthropicClient(api_key, primary_model)
                self.clients["anthropic"] = self.primary_client
        
        # 初始化深度洞察客户端（可能使用不同的模型）
        insight_model = llm_config.insight_model
        if insight_model != primary_model:
            # 如果洞察模型不同，需要单独初始化
            if insight_model.startswith("gpt") or insight_model.startswith("o1") or insight_model.startswith("claude"):
                api_key = self.config.get_api_key("openai") or self.config.get_api_key("gemini") or self.config.get_api_key("google_gemini")
                if api_key:
                    base_url = None
                    try:
                        base_url = (
                            getattr(getattr(self.config, "llm_settings", None), "base_url", None)
                            or getattr(getattr(self.config, "embedding_settings", None), "base_url", None)
                        )
                    except Exception:
                        base_url = None
                    self.insight_client = OpenAIClient(api_key, insight_model, base_url=base_url)
            elif insight_model.startswith("claude"):
                api_key = self.config.get_api_key("anthropic")
                if api_key:
                    self.insight_client = AnthropicClient(api_key, insight_model)
            elif insight_model.startswith("gemini"):
                api_key = self.config.get_api_key("google") or self.config.get_api_key("google_gemini")
                if api_key:
                    self.insight_client = GoogleClient(api_key, insight_model)
        else:
            self.insight_client = self.primary_client
        
        if not self.primary_client:
            raise ValueError(f"Failed to initialize primary LLM client for provider: {primary_provider}")
        
        self.logger.info(f"Initialized LLM service with provider: {primary_provider}, model: {primary_model}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        use_insight_model: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        生成回复
        
        Args:
            messages: 对话消息
            model: 指定模型
            temperature: 温度参数
            max_tokens: 最大token数
            system_prompt: 系统提示
            use_insight_model: 是否使用洞察模型
            **kwargs: 其他参数
            
        Returns:
            LLM响应
            
        Raises:
            LLMError: LLM调用失败
        """
        client = self.insight_client if use_insight_model else self.primary_client
        if not client:
            raise LLMError("No available LLM client")
        
        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            additional_params=kwargs
        )
        
        return await self._generate_with_retry(client, request)
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成回复
        
        Args:
            messages: 对话消息
            model: 指定模型
            temperature: 温度参数
            max_tokens: 最大token数
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段
        """
        if not self.primary_client:
            raise LLMError("No available LLM client")
        
        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            stream=True,
            additional_params=kwargs
        )
        
        async for chunk in self.primary_client.generate_stream(request):
            yield chunk
    
    async def _generate_with_retry(self, client: BaseLLMClient, request: LLMRequest) -> LLMResponse:
        """
        带重试机制的生成
        
        Args:
            client: LLM客户端
            request: LLM请求
            
        Returns:
            LLM响应
        """
        max_retries = self.config.performance.api_retry_attempts
        base_delay = self.config.performance.api_retry_delay_seconds
        
        for attempt in range(max_retries + 1):
            try:
                return await client.generate(request)
                
            except LLMRateLimitError as e:
                if attempt == max_retries:
                    raise
                
                # 使用API返回的重试时间或指数退避
                delay = e.retry_after or (base_delay * (2 ** attempt))
                self.logger.warning(f"Rate limit hit, retrying in {delay} seconds (attempt {attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(delay)
                
            except LLMAuthenticationError:
                # 认证错误不重试
                raise
                
            except (LLMAPIError, Exception) as e:
                if attempt == max_retries:
                    raise
                
                # 指数退避
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                self.logger.warning(f"API error, retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries + 1}): {e}")
                await asyncio.sleep(delay)
        
        raise LLMError("Max retries exceeded")
    
    async def close(self) -> None:
        """关闭所有客户端"""
        for client in self.clients.values():
            await client.close()
        
        if self.insight_client and self.insight_client not in self.clients.values():
            await self.insight_client.close()
        
        self.logger.info("LLM service closed")
