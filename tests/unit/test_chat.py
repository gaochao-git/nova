import pytest
from app.services.chat.memory import ConversationBufferMemory
from app.services.chat.llm import DeepSeekChat

@pytest.fixture
def memory():
    return ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

@pytest.fixture
def llm():
    return DeepSeekChat(
        api_key="test-key",
        temperature=0.7
    )

def test_memory_add_message(memory):
    memory.chat_memory.add_user_message("Hello")
    memory.chat_memory.add_ai_message("Hi there!")
    
    messages = memory.chat_memory.messages
    assert len(messages) == 2
    assert messages[0].content == "Hello"
    assert messages[1].content == "Hi there!" 