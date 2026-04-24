
# 阿里百炼 LLM 模块
"""
阿里百炼 LLM 模块
==============

使用 OpenAI 兼容模式调用阿里百炼大模型。

支持的模型：
- qwen3.6-35b-a3b: 千问3.6 35B 模型
- 其他 qwen 系列模型

API 文档参考：
- https://help.aliyun.com/zh/model-studio/developer-reference/openai-compatible-api
"""

import os
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI
from src.config import load_config

# 加载配置（支持 settings.yaml 和环境变量）
load_config()


class SourceType(Enum):
    """
    回答来源类型枚举
    """
    REFERENCE = "参考资料"
    BUILTIN = "大模型内置知识"
    UNKNOWN = "未知"


@dataclass
class LLMConfig:
    """
    LLM 配置类

    封装阿里百炼 API 的配置参数。
    """

    api_key: str
    model: str = "qwen3.6-35b-a3b"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """
        从环境变量创建配置实例
        """
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 环境变量未设置。"
                "请在 .env.local 文件中添加: DASHSCOPE_API_KEY=your-api-key"
            )

        return cls(
            api_key=api_key,
            model=os.getenv("DASHSCOPE_MODEL", "qwen3.6-35b-a3b"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            top_p=float(os.getenv("LLM_TOP_P", "0.9")),
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )


def build_rag_prompt(
    retrieved_context: str,
    user_query: str,
) -> str:
    """
    构建 RAG 提示词模板

    使用用户指定的提示词模板，包含角色设定、行为准则和约束条件。

    Args:
        retrieved_context: 从向量数据库检索到的上下文内容
        user_query: 用户的问题

    Returns:
        str: 完整的提示词字符串
    """
    prompt = f"""### Role (角色设定)
你是一个严谨的知识库分析专家。你的任务是根据提供的【参考资料】回答用户问题。

### Instructions (行为准则)
1. **优先权**：必须优先检索【参考资料】中的内容。
2. **强制标注**：
   - 如果答案完全来自【参考资料】，请在回答末尾标注：`[来源：参考资料]`。
   - 如果【参考资料】中不包含答案，但你通过自身知识储备进行了回答，请在回答末尾标注：`[来源：大模型内置知识]`，并明确说明资料中未提及相关信息。
   - 如果你无法确定答案，且【参考资料】中没有提及，请直接回答："在提供的资料中找不到相关信息，我无法回答该问题。"
3. **禁止幻觉**：严禁编造【参考资料】中不存在的细节（如日期、数据、专有名词）。

### Constraints (约束条件)
- 回答结构清晰，使用 Markdown 格式。
- 如果参考资料里有页码或章节信息，请在引用处标注。

---
### Context (参考资料)
{retrieved_context}

---
### User Question (用户问题)
{user_query}"""
    return prompt


class DashScopeLLM:
    """
    阿里百炼 LLM 客户端

    使用 OpenAI 兼容模式调用大模型进行对话和生成。
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化 LLM 客户端

        Args:
            config: LLM 配置，如果为 None 则从环境变量读取
        """
        if config is None:
            config = LLMConfig.from_env()

        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        生成文本回复

        Args:
            prompt: 用户输入的提示词
            system_prompt: 系统提示词（可选）
            **kwargs: 额外参数，覆盖默认配置

        Returns:
            str: 模型生成的回复文本
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"API 调用失败: {str(e)}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        多轮对话

        Args:
            messages: 对话历史列表，每个元素包含 role 和 content
            **kwargs: 额外参数

        Returns:
            str: 模型生成的回复文本
        """
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"API 调用失败: {str(e)}")

    def rag_query(
        self,
        user_query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        执行 RAG 查询

        这是核心方法，将检索到的文档和用户问题组合成提示词，
        调用大模型生成回答。

        Args:
            user_query: 用户的问题
            retrieved_docs: 从向量数据库检索到的文档列表，
                           每个文档包含 content 和 metadata

        Returns:
            Dict[str, Any]: 包含以下字段的字典：
                - answer: 模型生成的回答
                - source_type: 回答来源类型（参考资料/内置知识/未知）
                - retrieved_contexts: 使用的检索上下文
        """
        if not retrieved_docs:
            prompt = build_rag_prompt(
                retrieved_context="当前没有可用的参考资料，请告知用户需要先上传文档。",
                user_query=user_query
            )
            answer = self.generate(prompt)
            return {
                "answer": answer,
                "source_type": SourceType.UNKNOWN.value,
                "retrieved_contexts": [],
            }

        contexts = []
        for i, doc in enumerate(retrieved_docs):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            file_name = metadata.get("file_name", f"文档{i+1}")
            chunk_index = metadata.get("chunk_index", i)

            context_entry = f"【文档{i+1}: {file_name} - 第{chunk_index + 1}段】\n{content}"
            contexts.append(context_entry)

        retrieved_context = "\n\n---\n\n".join(contexts)

        prompt = build_rag_prompt(
            retrieved_context=retrieved_context,
            user_query=user_query
        )

        answer = self.generate(prompt)

        source_type = self._extract_source_type(answer)

        return {
            "answer": answer,
            "source_type": source_type.value,
            "retrieved_contexts": retrieved_docs,
        }

    def _extract_source_type(self, answer: str) -> SourceType:
        """
        从回答中提取来源类型

        根据回答末尾的标注判断来源类型。

        Args:
            answer: 模型生成的回答

        Returns:
            SourceType: 来源类型枚举
        """
        if "[来源：参考资料]" in answer:
            return SourceType.REFERENCE
        elif "[来源：大模型内置知识]" in answer:
            return SourceType.BUILTIN
        else:
            return SourceType.UNKNOWN


_llm_instance: Optional[DashScopeLLM] = None


def get_llm(config: Optional[LLMConfig] = None) -> DashScopeLLM:
    """
    获取 LLM 实例（单例模式）

    Args:
        config: LLM 配置，如果为 None 则从环境变量读取

    Returns:
        DashScopeLLM: LLM 实例
    """
    global _llm_instance

    if _llm_instance is None:
        _llm_instance = DashScopeLLM(config)

    return _llm_instance
