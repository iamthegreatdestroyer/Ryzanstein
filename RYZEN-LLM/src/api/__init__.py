"""
API Layer Package
[REF:API-008] - API Layer

This package implements the OpenAI-compatible API server and
MCP protocol integration for external tool use.

Modules:
    server: FastAPI-based API server
    mcp_bridge: MCP protocol handler
    streaming: SSE streaming support
"""

__version__ = "0.1.0"
__author__ = "RYZEN-LLM Project"

# TODO: Export main classes
# from .server import app
# from .mcp_bridge import MCPBridge
# from .streaming import StreamManager

__all__ = [
    # "app",
    # "MCPBridge",
    # "StreamManager",
]
