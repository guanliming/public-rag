# 配置加载器
"""
配置加载器
========

统一加载项目配置，支持 YAML 文件和环境变量。

配置加载顺序（优先级从低到高）：
1. settings.yaml（非敏感配置，可提交到 git）
2. .env（敏感配置，不应提交到 git）
3. .env.local（本地覆盖，最高优先级）

所有配置最终都会设置为环境变量，保持与现有代码的兼容性。
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# YAML 配置到环境变量的映射
# 注意：数据库配置全部视为敏感信息，不在 settings.yaml 中配置
# 数据库配置请放在 .env 文件中
YAML_TO_ENV_MAPPING = {
    "embedding": {
        "model_type": "EMBEDDING_MODEL_TYPE",
        "model_name": "EMBEDDING_MODEL_NAME",
        "device": "EMBEDDING_DEVICE",
    },
    "document": {
        "chunk_size": "CHUNK_SIZE",
        "chunk_overlap": "CHUNK_OVERLAP",
    },
    "llm": {
        "model": "DASHSCOPE_MODEL",
        "temperature": "LLM_TEMPERATURE",
        "max_tokens": "LLM_MAX_TOKENS",
        "top_p": "LLM_TOP_P",
        "base_url": "DASHSCOPE_BASE_URL",
    },
    "retrieval": {
        "k": "RETRIEVAL_K",
    },
    "flask": {
        "host": "FLASK_HOST",
        "port": "FLASK_PORT",
        "debug": "FLASK_DEBUG",
    },
    "upload": {
        "folder": "UPLOAD_FOLDER",
        "max_content_length": "MAX_CONTENT_LENGTH",
    },
    "database": {
        "auto_recreate_table": "AUTO_RECREATE_TABLE",
    },
}


def get_project_root() -> str:
    """获取项目根目录路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))


def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    """
    从 YAML 文件加载配置

    Args:
        yaml_path: YAML 文件路径

    Returns:
        配置字典，如果文件不存在或加载失败则返回空字典
    """
    if not HAS_YAML:
        print("警告: PyYAML 未安装，无法加载 YAML 配置文件")
        return {}

    if not os.path.exists(yaml_path):
        return {}

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config if config else {}
    except Exception as e:
        print(f"警告: 加载 YAML 配置文件失败: {e}")
        return {}


def yaml_to_env(config: Dict[str, Any], override: bool = False) -> None:
    """
    将 YAML 配置映射到环境变量

    Args:
        config: YAML 配置字典
        override: 是否覆盖已存在的环境变量
    """
    for section, mapping in YAML_TO_ENV_MAPPING.items():
        section_config = config.get(section, {})
        if not isinstance(section_config, dict):
            continue

        for yaml_key, env_key in mapping.items():
            value = section_config.get(yaml_key)
            if value is None:
                continue

            env_value = _format_env_value(value)
            
            if override or env_key not in os.environ:
                os.environ[env_key] = env_value


def _format_env_value(value: Any) -> str:
    """
    将值格式化为环境变量字符串

    Args:
        value: 任意类型的值

    Returns:
        格式化后的字符串
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def load_config(
    project_root: Optional[str] = None,
    verbose: bool = False
) -> None:
    """
    加载项目配置

    配置加载顺序（优先级从低到高）：
    1. settings.yaml（非敏感配置）
    2. .env（敏感配置）
    3. .env.local（本地覆盖，最高优先级）

    Args:
        project_root: 项目根目录路径，如果为 None 则自动检测
        verbose: 是否显示详细加载信息
    """
    if project_root is None:
        project_root = get_project_root()

    # 1. 加载 settings.yaml（非敏感配置，优先级最低）
    yaml_path = os.path.join(project_root, "settings.yaml")
    if os.path.exists(yaml_path):
        if verbose:
            print(f"加载配置文件: settings.yaml")
        yaml_config = load_yaml_config(yaml_path)
        yaml_to_env(yaml_config, override=False)
    elif verbose:
        print("未找到 settings.yaml，跳过")

    # 2. 加载 .env（敏感配置，优先级高于 YAML）
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        if verbose:
            print(f"加载配置文件: .env")
        load_dotenv(env_path, override=True)
    elif verbose:
        print("未找到 .env，跳过")

    # 3. 加载 .env.local（本地覆盖，优先级最高）
    env_local_path = os.path.join(project_root, ".env.local")
    if os.path.exists(env_local_path):
        if verbose:
            print(f"加载配置文件: .env.local")
        load_dotenv(env_local_path, override=True)
    elif verbose:
        print("未找到 .env.local，跳过")


def get_config_value(key: str, default: Any = None) -> Any:
    """
    获取配置值

    Args:
        key: 配置键（环境变量名）
        default: 默认值

    Returns:
        配置值
    """
    return os.getenv(key, default)
