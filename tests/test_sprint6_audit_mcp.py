"""Sprint 6: ZK audit trail + MCP endpoint tests."""

import asyncio
import sys
from pathlib import Path

API_PATH = str(Path(__file__).parent.parent / "RYZEN-LLM" / "src" / "api")
if API_PATH not in sys.path:
    sys.path.insert(0, API_PATH)


# ---------------------------------------------------------------------------
# audit_trail.py tests
# ---------------------------------------------------------------------------

def test_audit_chain_append_and_integrity():
    from audit_trail import AuditChain
    chain = AuditChain()
    chain.append("Inference", "prompt=hello", "user1")
    chain.append("Inference", "prompt=world", "user1")
    assert len(chain) == 2
    assert chain.verify_integrity()


def test_audit_chain_tamper_detection():
    from audit_trail import AuditChain
    chain = AuditChain()
    chain.append("Inference", "query", "user")
    chain.append("Inference", "query2", "user")
    chain._entries[0].hash = "tampered"
    assert not chain.verify_integrity()


def test_merkle_tree_root_exists():
    from audit_trail import AuditChain, MerkleTree
    chain = AuditChain()
    for i in range(4):
        chain.append("Inference", f"q{i}", "user")
    tree = MerkleTree.from_entries(chain.entries)
    root = tree.root()
    assert root is not None
    assert len(root) == 64  # sha256 hex


def test_merkle_tree_empty():
    from audit_trail import MerkleTree
    tree = MerkleTree([])
    assert tree.root() is None


def test_zk_proof_generate_and_verify():
    from audit_trail import ZkProof
    proof = ZkProof.generate(b"my-secret", b"audit-message")
    assert ZkProof.verify(proof, b"audit-message")


def test_zk_proof_wrong_message():
    from audit_trail import ZkProof
    proof = ZkProof.generate(b"secret", b"correct")
    assert not ZkProof.verify(proof, b"wrong")


def test_zk_proof_deterministic_commitment():
    from audit_trail import ZkProof
    p1 = ZkProof.generate(b"fixed-secret", b"msg1")
    p2 = ZkProof.generate(b"fixed-secret", b"msg2")
    assert p1.commitment == p2.commitment


def test_audit_manager_seal():
    from audit_trail import AuditManager
    mgr = AuditManager(request_id="req-001", api_key_prefix="testkey")
    mgr.record_inference(model="ryzanstein-7b", prompt_tokens=50, output_tokens=100)
    result = mgr.seal()
    assert result["request_id"] == "req-001"
    assert result["entry_count"] == 1
    assert result["chain_integrity"] is True
    assert result["zk_verified"] is True
    assert len(result["merkle_root"]) == 64


# ---------------------------------------------------------------------------
# mcp_bridge.py tests
# ---------------------------------------------------------------------------

def test_mcp_list_tools():
    from mcp_bridge import MCPBridge
    bridge = MCPBridge()
    bridge.register_tool("echo", "Echo input", {}, lambda args: args)
    tools = bridge.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "echo"


def test_mcp_handle_ping():
    from mcp_bridge import MCPBridge
    bridge = MCPBridge()

    async def run():
        return await bridge.handle_request({"method": "ping", "id": 1})

    result = asyncio.run(run())
    assert result["result"]["status"] == "pong"


def test_mcp_handle_tools_list():
    from mcp_bridge import MCPBridge
    bridge = MCPBridge()
    bridge.register_tool("noop", "No-op", {}, lambda args: {})

    async def run():
        return await bridge.handle_request({"method": "tools/list", "id": 2})

    result = asyncio.run(run())
    assert "tools" in result["result"]
    assert len(result["result"]["tools"]) == 1


def test_mcp_handle_tools_call():
    from mcp_bridge import MCPBridge
    bridge = MCPBridge()
    bridge.register_tool("add", "Add two numbers", {}, lambda args: {"sum": args["a"] + args["b"]})

    async def run():
        return await bridge.handle_request({
            "method": "tools/call",
            "id": 3,
            "params": {"name": "add", "arguments": {"a": 3, "b": 4}},
        })

    result = asyncio.run(run())
    import json
    content = json.loads(result["result"]["content"][0]["text"])
    assert content["sum"] == 7


def test_mcp_handle_unknown_tool():
    from mcp_bridge import MCPBridge
    bridge = MCPBridge()

    async def run():
        return await bridge.handle_request({
            "method": "tools/call",
            "id": 4,
            "params": {"name": "nonexistent", "arguments": {}},
        })

    result = asyncio.run(run())
    assert result["error"]["code"] == -32601


def test_mcp_unknown_method():
    from mcp_bridge import MCPBridge
    bridge = MCPBridge()

    async def run():
        return await bridge.handle_request({"method": "bad/method", "id": 5})

    result = asyncio.run(run())
    assert result["error"]["code"] == -32601


def test_mcp_session_lifecycle():
    from mcp_bridge import MCPBridge
    bridge = MCPBridge()
    bridge.create_session("s1")
    assert "s1" in bridge.sessions
    bridge.cleanup_session("s1")
    assert "s1" not in bridge.sessions
