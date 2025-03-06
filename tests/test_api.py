import requests
import json
import time
from typing import Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.conversation_id = None

    def test_chat_basic(self):
        """测试基础对话功能"""
        logger.info("测试基础对话功能...")
        
        # 1. 发起新对话
        response = self.chat("你好，请介绍一下你自己", use_rag=False)
        assert response.status_code == 200, "新对话请求失败"
        data = response.json()
        self.conversation_id = data["conversation_id"]
        logger.info(f"对话ID: {self.conversation_id}")
        logger.info(f"AI回复: {data['response']}")

        # 2. 继续对话
        response = self.chat("你能做些什么？", use_rag=False)
        assert response.status_code == 200, "继续对话请求失败"
        logger.info(f"AI回复: {response.json()['response']}")

    def test_chat_rag(self):
        """测试 RAG 对话功能"""
        logger.info("测试 RAG 对话功能...")
        
        # 1. 添加文档
        doc_content = """
        人工智能(AI)是计算机科学的一个分支，它企图了解智能的实质，
        并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        人工智能的研究包括机器学习、计算机视觉、自然语言处理等领域。
        """
        response = self.add_document(doc_content)
        assert response.status_code == 200, "添加文档失败"
        logger.info("文档添加成功")

        # 2. 使用 RAG 进行对话
        response = self.chat("解释一下什么是人工智能？", use_rag=True)
        assert response.status_code == 200, "RAG 对话请求失败"
        logger.info(f"AI回复: {response.json()['response']}")

    def test_document_operations(self):
        """测试文档操作功能"""
        logger.info("测试文档操作功能...")
        
        # 1. 添加文档
        doc_content = "这是一个测试文档。包含一些测试内容。"
        response = self.add_document(doc_content)
        assert response.status_code == 200, "添加文档失败"
        logger.info("文档添加成功")

        # 2. 搜索文档
        response = self.search_documents("测试", top_k=2)
        assert response.status_code == 200, "搜索文档失败"
        results = response.json()["results"]
        logger.info(f"搜索结果: {json.dumps(results, ensure_ascii=False, indent=2)}")

    def test_conversation_management(self):
        """测试对话管理功能"""
        logger.info("测试对话管理功能...")
        
        # 1. 获取对话列表
        response = self.list_conversations()
        assert response.status_code == 200, "获取对话列表失败"
        data = response.json()
        logger.info(f"对话列表: {json.dumps(data, ensure_ascii=False, indent=2)}")

        # 2. 获取特定对话历史
        if self.conversation_id:
            response = self.get_conversation(self.conversation_id)
            assert response.status_code == 200, "获取对话历史失败"
            history = response.json()
            logger.info(f"对话历史: {json.dumps(history, ensure_ascii=False, indent=2)}")

    def chat(self, content: str, use_rag: bool = False) -> requests.Response:
        """发送聊天请求"""
        url = f"{self.base_url}/api/chat"
        data = {
            "content": content,
            "use_rag": use_rag
        }
        if self.conversation_id:
            data["conversation_id"] = self.conversation_id
        return requests.post(url, json=data)

    def add_document(self, source: str, type: str = "text") -> requests.Response:
        """添加文档"""
        url = f"{self.base_url}/api/documents/add"
        return requests.post(url, json={"source": source, "type": type})

    def search_documents(self, query: str, top_k: int = 3) -> requests.Response:
        """搜索文档"""
        url = f"{self.base_url}/api/documents/search"
        return requests.post(url, json={"query": query, "top_k": top_k})

    def list_conversations(self) -> requests.Response:
        """获取对话列表"""
        url = f"{self.base_url}/api/chat/conversations"
        return requests.get(url)

    def get_conversation(self, conversation_id: str) -> requests.Response:
        """获取特定对话历史"""
        url = f"{self.base_url}/api/chat/conversations"
        params = {"conversation_id": conversation_id}
        return requests.get(url, params=params)

def main():
    tester = APITester()
    
    try:
        # 运行所有测试
        tester.test_chat_basic()
        time.sleep(1)  # 添加延迟避免请求过快
        
        tester.test_chat_rag()
        time.sleep(1)
        
        tester.test_document_operations()
        time.sleep(1)
        
        tester.test_conversation_management()
        
        logger.info("所有测试完成!")
        
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 