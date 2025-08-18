#!/usr/bin/env python3
"""
简单测试脚本
用于测试PDF处理功能
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目路径到sys.path
sys.path.insert(0, str(Path(__file__).parent))

from main import DeepScholarAI

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
        # 创建应用实例
        app = DeepScholarAI()
        
        # 初始化
        await app.initialize("config.yaml")
        
        # 处理PDF
        print("📄 开始处理PDF...")
        result = await app.process_paper(str(pdf_path.absolute()))
        
        if result:
            print(f"✅ PDF处理成功！输出文件: {result}")
            return True
        else:
            print("❌ PDF处理失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            await app.shutdown()
        except:
            pass

if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️ 警告: 未设置API密钥环境变量 (OPENAI_API_KEY 或 DEEPSEEK_API_KEY)")
        print("请设置你的DeepSeek API密钥:")
        print("export OPENAI_API_KEY='你的deepseek_api_key'")
        print("或者在config.yaml中配置api_keys.openai")
        
    # 运行测试
    success = asyncio.run(test_pdf_processing())
    
    if success:
        print("\n🎉 测试完成！")
        sys.exit(0)
    else:
        print("\n💥 测试失败！")
        sys.exit(1)
