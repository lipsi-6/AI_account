"""
交互式编辑审稿功能

支持通过特定注释格式进行文章修订：
<!-- REVISE: "修改指令" --> 原文内容 <!-- END_REVISE -->

工作流程：
1. 解析用户修订请求
2. 调用LLM进行精确修改
3. 替换原文并保存版本化文件
"""

import asyncio
import argparse
import sys
import re
from pathlib import Path
from typing import List, Tuple, Optional
import aiofiles

from modules.config_manager import ConfigManager
from modules.logger import setup_logging, get_logger
from modules.llm_provider import LLMProviderService


class RevisionRequest:
    """修订请求数据类"""
    def __init__(self, instruction: str, original_text: str, start_pos: int, end_pos: int):
        self.instruction = instruction
        self.original_text = original_text
        self.start_pos = start_pos
        self.end_pos = end_pos


class RevisionProcessor:
    """修订处理器"""
    
    REVISION_PATTERN = re.compile(
        r'<!-- REVISE:\s*"([^"]+)"\s*-->(.*?)<!-- END_REVISE -->',
        re.DOTALL | re.IGNORECASE
    )
    
    def __init__(self, llm_provider: LLMProviderService):
        self.llm_provider = llm_provider
        self.logger = get_logger("revision_processor")
    
    async def parse_revision_requests(self, content: str) -> List[RevisionRequest]:
        """解析修订请求"""
        requests = []
        
        for match in self.REVISION_PATTERN.finditer(content):
            instruction = match.group(1).strip()
            original_text = match.group(2).strip()
            start_pos = match.start()
            end_pos = match.end()
            
            requests.append(RevisionRequest(
                instruction=instruction,
                original_text=original_text,
                start_pos=start_pos,
                end_pos=end_pos
            ))
            
            self.logger.info(f"发现修订请求: {instruction[:50]}...")
        
        return requests
    
    async def process_revision(self, request: RevisionRequest) -> str:
        """处理单个修订请求"""
        # 构建修订Prompt
        revision_prompt = f"""你是一位名叫Deep Scholar AI的顶尖AI研究员，正在审阅自己的草稿。你的目标是根据用户的精确反馈，以最严谨和专业的方式修改文本。

请根据以下用户反馈，**精确地、仅**修改提供的"原始文本"。你的输出必须是**修改后的文本，绝无任何额外解释或说明**。如果原始文本中包含数学公式，请确保修改后公式仍然完整无误。

---
用户反馈: "{request.instruction}"
---
你的原始文本:
{request.original_text}
---
修改后的文本:"""

        try:
            response = await self.llm_provider.generate(
                messages=[{"role": "user", "content": revision_prompt}],
                temperature=0.3,  # 较低的温度以确保准确性
                max_tokens=2000
            )
            
            revised_text = response.content.strip()
            
            self.logger.info(f"修订完成，原文长度: {len(request.original_text)}, 修改后长度: {len(revised_text)}")
            
            return revised_text
            
        except Exception as e:
            self.logger.error(f"修订处理失败: {e}")
            raise
    
    async def apply_revisions(self, content: str, requests: List[RevisionRequest]) -> str:
        """应用所有修订"""
        if not requests:
            return content
        
        # 按位置倒序排序，从后往前替换以避免位置偏移
        requests.sort(key=lambda x: x.start_pos, reverse=True)
        
        revised_content = content
        
        for request in requests:
            try:
                # 处理修订
                revised_text = await self.process_revision(request)
                
                # 替换内容
                revised_content = (
                    revised_content[:request.start_pos] + 
                    revised_text + 
                    revised_content[request.end_pos:]
                )
                
                self.logger.info(f"应用修订: {request.instruction[:30]}...")
                
            except Exception as e:
                self.logger.error(f"应用修订失败 '{request.instruction}': {e}")
                # 继续处理其他修订
                continue
        
        return revised_content


class InteractiveRevisionTool:
    """交互式修订工具"""
    
    def __init__(self):
        self.config: Optional[ConfigManager] = None
        self.llm_provider: Optional[LLMProviderService] = None
        self.processor: Optional[RevisionProcessor] = None
        self.logger = None
    
    async def initialize(self, config_path: str):
        """初始化修订工具"""
        try:
            # 初始化配置管理器
            self.config = ConfigManager()
            await self.config.initialize(config_path)
            
            # 设置日志系统
            await setup_logging(self.config)
            self.logger = get_logger("interactive_revision")
            
            # 初始化LLM服务
            self.llm_provider = LLMProviderService(self.config)
            
            # 初始化修订处理器
            self.processor = RevisionProcessor(self.llm_provider)
            
            self.logger.info("修订工具初始化完成")
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise
    
    async def run_revision_cycle(self, file_path: str) -> None:
        """运行修订周期"""
        if not self.processor:
            raise RuntimeError("修订工具未初始化")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        self.logger.info(f"开始处理修订: {file_path}")
        
        try:
            # 读取文件内容
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # 解析修订请求
            requests = await self.processor.parse_revision_requests(content)
            
            if not requests:
                print("📝 未发现修订请求")
                return
            
            print(f"📝 发现 {len(requests)} 个修订请求")
            
            # 显示修订预览
            for i, request in enumerate(requests, 1):
                print(f"\n修订 {i}: {request.instruction}")
                print(f"原文预览: {request.original_text[:100]}...")
            
            # 确认执行（非阻塞输入，避免阻塞事件循环）
            confirm = (await asyncio.to_thread(input, f"\n是否执行这 {len(requests)} 个修订？(y/n): ")).strip().lower()
            if confirm not in ['y', 'yes', '是']:
                print("❌ 用户取消修订")
                return
            
            # 应用修订
            print("\n🔄 正在处理修订...")
            revised_content = await self.processor.apply_revisions(content, requests)
            
            # 生成版本化文件名
            version_file_path = self._get_next_version_path(file_path)
            
            # 保存修改后的文件
            async with aiofiles.open(version_file_path, 'w', encoding='utf-8') as f:
                await f.write(revised_content)
            
            print(f"✅ 修订完成，已保存为: {version_file_path}")
            self.logger.info(f"修订完成: {version_file_path}")
            
        except Exception as e:
            self.logger.error(f"修订周期失败: {e}", exc_info=True)
            print(f"❌ 修订失败: {e}")
            raise
    
    def _get_next_version_path(self, original_path: Path) -> Path:
        """获取下一个版本的文件路径"""
        base_name = original_path.stem
        suffix = original_path.suffix
        parent = original_path.parent
        
        # 查找已存在的版本
        version = 2
        while True:
            version_path = parent / f"{base_name}_v{version}{suffix}"
            if not version_path.exists():
                return version_path
            version += 1
    
    async def batch_revision(self, directory: str) -> None:
        """批量修订目录下的所有markdown文件"""
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"目录不存在或不是目录: {directory}")
        
        # 查找所有markdown文件
        md_files = list(directory.glob("*.md"))
        
        if not md_files:
            print("📝 目录中没有找到markdown文件")
            return
        
        print(f"📁 发现 {len(md_files)} 个markdown文件")
        
        for md_file in md_files:
            print(f"\n📄 处理: {md_file.name}")
            try:
                await self.run_revision_cycle(str(md_file))
            except Exception as e:
                print(f"❌ 处理 {md_file.name} 失败: {e}")
                continue
        
        print("\n✅ 批量修订完成")
    
    async def close(self):
        """关闭修订工具"""
        if self.llm_provider:
            await self.llm_provider.close()
        
        if self.logger:
            self.logger.info("修订工具已关闭")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Deep Scholar AI - 交互式修订工具")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )
    parser.add_argument(
        "--file",
        help="要修订的文件路径"
    )
    parser.add_argument(
        "--directory",
        help="要批量修订的目录路径"
    )
    
    args = parser.parse_args()
    
    if not args.file and not args.directory:
        print("❌ 请提供 --file 或 --directory 参数")
        sys.exit(1)
    
    # 检查配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 创建并初始化修订工具
    tool = InteractiveRevisionTool()
    
    try:
        print("🚀 初始化修订工具...")
        await tool.initialize(str(config_path))
        
        if args.file:
            await tool.run_revision_cycle(args.file)
        elif args.directory:
            await tool.batch_revision(args.directory)
            
    except KeyboardInterrupt:
        print("\n接收到中断信号")
    except Exception as e:
        print(f"❌ 修订工具运行失败: {e}")
        sys.exit(1)
    finally:
        await tool.close()


if __name__ == "__main__":
    asyncio.run(main())
