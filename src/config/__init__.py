# 配置模块
"""
配置模块
======

统一管理项目配置，支持从 YAML 文件和环境变量加载配置。

配置加载顺序（优先级从低到高）：
1. settings.yaml（非敏感配置，可提交到 git）
2. .env（敏感配置，不应提交到 git）
3. .env.local（本地覆盖，最高优先级）

所有配置最终都会设置为环境变量，保持与现有代码的兼容性。
"""

from .loader import load_config, get_config_value

__all__ = ["load_config", "get_config_value"]
