# 文档处理模块
"""
文档处理模块
============

实现文档的加载、解析和切片功能，
是 RAG 系统中数据预处理的核心模块。

设计思路：
1. 文档加载：根据文件扩展名选择合适的加载器
2. 文本提取：从文档中提取纯文本内容
3. 文本切片：将长文本切分成合适大小的片段
4. 元数据增强：添加文档来源、创建时间等元数据

为什么需要文档切片？
- 向量模型有最大输入长度限制
- 过长的文本会导致语义模糊
- 合适的片段大小可以提高检索精度
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
from datetime import datetime

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from dotenv import load_dotenv

# 加载环境变量
# 优先加载 .env.local（包含敏感信息，不会提交到 git）
# 然后加载 .env（作为默认值）
_env_local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env.local')
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')

if os.path.exists(_env_local_path):
    load_dotenv(_env_local_path, override=True)

if os.path.exists(_env_path):
    load_dotenv(_env_path, override=False)


class SupportedFormats(Enum):
    """
    支持的文档格式枚举

    使用枚举定义支持的文件类型，
    便于类型检查和扩展新格式。

    属性:
        PDF: PDF 文档格式
        MARKDOWN: Markdown 文档格式
        TEXT: 纯文本格式
    """

    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"

    @classmethod
    def from_extension(cls, extension: str) -> Optional["SupportedFormats"]:
        """
        根据文件扩展名判断文档格式

        Args:
            extension: 文件扩展名（如 ".pdf" 或 "pdf"）

        Returns:
            Optional[SupportedFormats]: 对应的格式枚举，不支持的格式返回 None

        示例:
            >>> SupportedFormats.from_extension(".pdf")
            <SupportedFormats.PDF: 'pdf'>
            >>> SupportedFormats.from_extension("md")
            <SupportedFormats.MARKDOWN: 'markdown'>
        """
        # 统一处理为小写并移除前导点
        ext = extension.lower().lstrip(".")

        if ext in ["pdf"]:
            return cls.PDF
        elif ext in ["md", "markdown"]:
            return cls.MARKDOWN
        elif ext in ["txt", "text"]:
            return cls.TEXT
        return None

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        获取所有支持的文件扩展名列表

        Returns:
            List[str]: 支持的扩展名列表

        示例:
            >>> SupportedFormats.get_supported_extensions()
            ['pdf', 'md', 'markdown', 'txt', 'text']
        """
        return ["pdf", "md", "markdown", "txt", "text"]


class DocumentLoader:
    """
    文档加载器

    负责从文件系统加载各种格式的文档，
    并将其转换为 LangChain 的 Document 对象。

    设计模式：
    - 使用工厂模式根据文件类型选择加载器
    - 封装不同加载器的差异，提供统一接口

    属性:
        loaders: 格式到加载器类的映射字典
    """

    def __init__(self):
        """
        初始化文档加载器

        预注册支持的格式和对应的加载器类。
        """
        # 格式到加载器类的映射
        self.loaders = {
            SupportedFormats.PDF: PyPDFLoader,
            SupportedFormats.MARKDOWN: UnstructuredMarkdownLoader,
            SupportedFormats.TEXT: TextLoader,
        }

    def load_file(self, file_path: str) -> List[Document]:
        """
        加载单个文档文件

        根据文件扩展名自动选择合适的加载器，
        读取文件内容并返回 Document 对象列表。

        Args:
            file_path: 文档文件的完整路径

        Returns:
            List[Document]: Document 对象列表

        Raises:
            ValueError: 文件格式不支持
            FileNotFoundError: 文件不存在
            Exception: 加载过程中的其他错误

        示例:
            >>> loader = DocumentLoader()
            >>> docs = loader.load_file("example.pdf")
            >>> print(len(docs))
            5  # PDF 通常每页生成一个 Document
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 获取文件扩展名
        _, extension = os.path.splitext(file_path)

        # 判断文档格式
        doc_format = SupportedFormats.from_extension(extension)
        if doc_format is None:
            raise ValueError(
                f"不支持的文件格式: {extension}。"
                f"支持的格式: {SupportedFormats.get_supported_extensions()}"
            )

        # 获取对应的加载器类
        loader_class = self.loaders.get(doc_format)
        if loader_class is None:
            raise ValueError(f"未找到格式 {doc_format} 的加载器")

        try:
            # 创建加载器实例并加载文档
            loader = loader_class(file_path)
            documents = loader.load()

            # 增强元数据
            self._enhance_metadata(documents, file_path, doc_format)

            print(f"成功加载文档: {file_path}，共 {len(documents)} 页/段")
            return documents

        except Exception as e:
            raise Exception(f"加载文档 {file_path} 时出错: {e}")

    def load_directory(
        self,
        directory_path: str,
        recursive: bool = False,
    ) -> List[Document]:
        """
        加载目录中的所有支持的文档

        Args:
            directory_path: 目录路径
            recursive: 是否递归加载子目录

        Returns:
            List[Document]: 所有加载的 Document 对象列表

        示例:
            >>> loader = DocumentLoader()
            >>> all_docs = loader.load_directory("./documents", recursive=True)
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"目录不存在: {directory_path}")

        all_documents = []

        # 获取目录中的所有文件
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)

                # 检查是否是支持的格式
                _, ext = os.path.splitext(filename)
                if SupportedFormats.from_extension(ext) is not None:
                    try:
                        documents = self.load_file(file_path)
                        all_documents.extend(documents)
                    except Exception as e:
                        print(f"跳过文件 {file_path}: {e}")

            # 如果不递归，跳过子目录
            if not recursive:
                break

        print(f"从目录 {directory_path} 加载了 {len(all_documents)} 个文档片段")
        return all_documents

    def _enhance_metadata(
        self,
        documents: List[Document],
        file_path: str,
        doc_format: SupportedFormats,
    ) -> None:
        """
        增强文档元数据

        为每个 Document 对象添加额外的元数据，
        包括文件信息、创建时间、格式等。

        Args:
            documents: Document 对象列表
            file_path: 源文件路径
            doc_format: 文档格式

        修改:
            直接修改传入的 documents 列表中的对象的 metadata 属性
        """
        # 获取文件的基本信息
        path_obj = Path(file_path)
        file_stat = path_obj.stat()

        # 基础元数据
        base_metadata = {
            "source": file_path,
            "file_name": path_obj.name,
            "file_size": file_stat.st_size,
            "format": doc_format.value,
            "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            "loaded_at": datetime.now().isoformat(),
        }

        # 为每个文档添加元数据
        for i, doc in enumerate(documents):
            # 合并原有元数据和增强元数据
            enhanced_metadata = {
                **base_metadata,
                **doc.metadata,
                "chunk_index": i,  # 当前片段在文件中的索引
                "total_chunks": len(documents),  # 文件总片段数
            }

            # 更新文档的元数据
            doc.metadata = enhanced_metadata


class TextSplitter:
    """
    文本切片器

    将长文本切分成合适大小的片段，
    是 RAG 系统中非常关键的一步。

    为什么要切片？
    1. 向量模型有最大输入长度限制
    2. 过长的文本会导致语义稀释
    3. 合适的片段可以提高检索精度
    4. 较小的片段更容易匹配用户查询

    切片策略：
    - 递归字符分割：通用文本，按字符切分
    - Markdown 标题分割：保留文档结构
    - 语义分割：按语义单元切分（需要额外模型）

    属性:
        chunk_size: 每个片段的最大字符数
        chunk_overlap: 片段之间的重叠字符数
        separators: 分割时使用的分隔符列表
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ):
        """
        初始化文本切片器

        Args:
            chunk_size: 每个片段的最大字符数，默认从环境变量读取或 1000
            chunk_overlap: 片段之间的重叠字符数，默认从环境变量读取或 200
            separators: 分割时使用的分隔符列表，按优先级排列

        重叠的作用：
            - 保持上下文的连贯性
            - 避免重要信息被切断
            - 一般设置为 chunk_size 的 10%-20%
        """
        # 从环境变量读取默认值
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "200"))

        # 默认分隔符，按优先级排列
        # 尝试按语义边界分割，保持段落和句子的完整性
        self.separators = separators or [
            "\n\n",  # 段落分隔
            "\n",    # 换行
            ". ",    # 句子结束（英文）
            "。",    # 句子结束（中文）
            " ",     # 空格
            "",      # 字符
        ]

    def split_documents(
        self,
        documents: List[Document],
        doc_format: Optional[SupportedFormats] = None,
    ) -> List[Document]:
        """
        分割文档列表

        根据文档格式选择合适的分割策略，
        对所有 Document 对象进行切片。

        Args:
            documents: 要分割的 Document 列表
            doc_format: 文档格式，用于选择分割策略

        Returns:
            List[Document]: 分割后的 Document 列表

        示例:
            >>> splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
            >>> chunks = splitter.split_documents(documents)
            >>> print(f"分割为 {len(chunks)} 个片段")
        """
        # 如果有明确的格式，选择专用分割器
        if doc_format == SupportedFormats.MARKDOWN:
            return self._split_markdown(documents)
        else:
            return self._split_generic(documents)

    def _split_generic(self, documents: List[Document]) -> List[Document]:
        """
        通用文本分割

        使用 RecursiveCharacterTextSplitter，
        这是 LangChain 最常用的文本分割器。

        工作原理：
        1. 尝试按第一个分隔符分割
        2. 如果片段仍然太大，尝试下一个分隔符
        3. 递归此过程直到片段大小合适

        Args:
            documents: 要分割的 Document 列表

        Returns:
            List[Document]: 分割后的 Document 列表
        """
        # 创建递归字符分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,  # 使用字符数计算长度
            is_separator_regex=False,
        )

        # 执行分割
        split_docs = text_splitter.split_documents(documents)

        # 为每个片段添加切片元数据
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_size"] = len(doc.page_content)
            doc.metadata["split_index"] = i
            doc.metadata["split_method"] = "recursive_character"

        print(f"文本分割完成: {len(documents)} -> {len(split_docs)} 个片段")
        return split_docs

    def _split_markdown(self, documents: List[Document]) -> List[Document]:
        """
        Markdown 文档专用分割

        使用 MarkdownHeaderTextSplitter，
        保留 Markdown 的标题结构，
        使每个片段都有明确的语义边界。

        优势：
        - 保持文档的层级结构
        - 每个片段都有完整的标题上下文
        - 提高检索的相关性

        Args:
            documents: 要分割的 Document 列表

        Returns:
            List[Document]: 分割后的 Document 列表
        """
        # 首先合并所有 Markdown 文档的内容
        # 因为 Markdown 分割器需要完整的文档结构
        combined_text = "\n".join([doc.page_content for doc in documents])
        combined_metadata = documents[0].metadata if documents else {}

        # 定义要识别的标题级别
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

        # 创建 Markdown 标题分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,  # 保留标题在内容中
        )

        # 按标题分割
        md_header_splits = markdown_splitter.split_text(combined_text)

        # 如果分割后的片段仍然太大，进行二次分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

        # 对每个标题片段进行二次分割
        final_chunks = []
        for doc in md_header_splits:
            # 合并元数据
            doc.metadata = {
                **combined_metadata,
                **doc.metadata,
                "split_method": "markdown_header",
            }

            # 如果片段太大，进一步分割
            if len(doc.page_content) > self.chunk_size:
                sub_chunks = text_splitter.split_documents([doc])
                for i, chunk in enumerate(sub_chunks):
                    chunk.metadata["sub_chunk_index"] = i
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(doc)

        # 添加索引元数据
        for i, doc in enumerate(final_chunks):
            doc.metadata["split_index"] = i
            doc.metadata["chunk_size"] = len(doc.page_content)

        print(f"Markdown 分割完成: {len(documents)} -> {len(final_chunks)} 个片段")
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """
        直接分割纯文本字符串

        Args:
            text: 要分割的文本字符串

        Returns:
            List[str]: 分割后的文本片段列表

        示例:
            >>> splitter = TextSplitter(chunk_size=100)
            >>> chunks = splitter.split_text("非常长的文本...")
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )
        return text_splitter.split_text(text)


class DocumentProcessor:
    """
    文档处理器

    整合文档加载和文本切片功能，
    提供一站式的文档处理接口。

    这是对外的主要接口，封装了加载和切片的细节。

    处理流程：
    1. 加载文档文件
    2. (可选) 预处理文本
    3. 切片文本
    4. 返回处理好的 Document 列表

    属性:
        loader: DocumentLoader 实例
        splitter: TextSplitter 实例
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        初始化文档处理器

        Args:
            chunk_size: 切片大小，默认从环境变量读取
            chunk_overlap: 切片重叠，默认从环境变量读取
        """
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process_file(
        self,
        file_path: str,
        split: bool = True,
    ) -> List[Document]:
        """
        处理单个文件

        完整的处理流程：加载 -> (可选)切片 -> 返回

        Args:
            file_path: 文件路径
            split: 是否进行切片，默认为 True

        Returns:
            List[Document]: 处理后的 Document 列表

        示例:
            >>> processor = DocumentProcessor(chunk_size=1000)
            >>> docs = processor.process_file("document.pdf")
            >>> print(f"处理完成，共 {len(docs)} 个片段")
        """
        # 加载文档
        documents = self.loader.load_file(file_path)

        if not documents:
            print(f"警告: 文件 {file_path} 没有加载到任何内容")
            return []

        # 获取文档格式
        _, ext = os.path.splitext(file_path)
        doc_format = SupportedFormats.from_extension(ext)

        # 切片（如果需要）
        if split:
            documents = self.splitter.split_documents(documents, doc_format)

        return documents

    def process_directory(
        self,
        directory_path: str,
        recursive: bool = False,
        split: bool = True,
    ) -> List[Document]:
        """
        处理目录中的所有文件

        Args:
            directory_path: 目录路径
            recursive: 是否递归处理子目录
            split: 是否进行切片

        Returns:
            List[Document]: 所有处理后的 Document 列表
        """
        # 加载目录中的所有文档
        documents = self.loader.load_directory(directory_path, recursive)

        if not documents:
            print(f"警告: 目录 {directory_path} 中没有加载到任何文档")
            return []

        # 切片（如果需要）
        if split:
            # 注意：这里假设所有文档都是相同格式
            # 如果需要混合格式处理，应该按格式分组后分别处理
            documents = self.splitter.split_documents(documents)

        return documents

    @staticmethod
    def get_document_stats(documents: List[Document]) -> Dict[str, Any]:
        """
        获取文档统计信息

        分析 Document 列表，生成统计报告。

        Args:
            documents: Document 列表

        Returns:
            Dict[str, Any]: 统计信息字典，包括：
                - total_count: 文档总数
                - total_chars: 总字符数
                - avg_chars: 平均字符数
                - min_chars: 最小字符数
                - max_chars: 最大字符数
                - sources: 来源文件列表
                - formats: 格式分布

        示例:
            >>> stats = DocumentProcessor.get_document_stats(docs)
            >>> print(f"总字符数: {stats['total_chars']}")
        """
        if not documents:
            return {"total_count": 0, "error": "没有文档"}

        # 计算字符数统计
        char_counts = [len(doc.page_content) for doc in documents]

        # 收集来源文件
        sources = list(set([doc.metadata.get("source", "unknown") for doc in documents]))

        # 收集格式分布
        formats = {}
        for doc in documents:
            fmt = doc.metadata.get("format", "unknown")
            formats[fmt] = formats.get(fmt, 0) + 1

        return {
            "total_count": len(documents),
            "total_chars": sum(char_counts),
            "avg_chars": sum(char_counts) / len(char_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "sources": sources,
            "formats": formats,
        }
