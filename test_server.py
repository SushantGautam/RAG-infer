"""
Test script for the RAG FastAPI server.
This script tests the server endpoints without requiring Milvus or OpenAI API.
"""

import json
import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import fastapi
        import uvicorn
        import langchain
        import pydantic
        print("✓ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_server_structure():
    """Test that the server file has the correct structure."""
    print("\nTesting server structure...")
    try:
        # Import without running main
        import rag_server
        
        # Check for required components
        assert hasattr(rag_server, 'app'), "Missing FastAPI app"
        assert hasattr(rag_server, 'ChatCompletionRequest'), "Missing ChatCompletionRequest model"
        assert hasattr(rag_server, 'ChatCompletionResponse'), "Missing ChatCompletionResponse model"
        assert hasattr(rag_server, 'Message'), "Missing Message model"
        assert hasattr(rag_server, 'parse_args'), "Missing parse_args function"
        assert hasattr(rag_server, 'initialize_rag_system'), "Missing initialize_rag_system function"
        
        # Check endpoints
        routes = [route.path for route in rag_server.app.routes]
        assert '/' in routes, "Missing root endpoint"
        assert '/health' in routes, "Missing health endpoint"
        assert '/v1/chat/completions' in routes, "Missing chat completions endpoint"
        
        print("✓ Server structure is correct")
        print(f"  Found endpoints: {', '.join(routes)}")
        return True
    except Exception as e:
        print(f"✗ Structure test failed: {e}")
        return False


def test_request_models():
    """Test that request/response models work correctly."""
    print("\nTesting Pydantic models...")
    try:
        from rag_server import ChatCompletionRequest, ChatCompletionResponse, Message, ChatCompletionChoice, Usage
        
        # Test Message
        msg = Message(role="user", content="What is FastAPI?")
        assert msg.role == "user"
        assert msg.content == "What is FastAPI?"
        
        # Test ChatCompletionRequest
        req = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[Message(role="user", content="What is FastAPI?")]
        )
        assert req.model == "gpt-3.5-turbo"
        assert len(req.messages) == 1
        
        # Test ChatCompletionResponse
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content="FastAPI is a web framework"),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )
        assert resp.id == "chatcmpl-123"
        assert resp.choices[0].message.content == "FastAPI is a web framework"
        
        print("✓ Pydantic models work correctly")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_documents_exist():
    """Test that sample documents exist."""
    print("\nTesting sample documents...")
    import os
    
    docs_path = "./documents"
    expected_files = ["fastapi.txt", "langchain.txt", "milvus.txt"]
    
    if not os.path.exists(docs_path):
        print(f"✗ Documents directory not found: {docs_path}")
        return False
    
    found_files = []
    for file in expected_files:
        file_path = os.path.join(docs_path, file)
        if os.path.exists(file_path):
            found_files.append(file)
        else:
            print(f"  Warning: {file} not found")
    
    if found_files:
        print(f"✓ Found {len(found_files)}/{len(expected_files)} sample documents")
        for file in found_files:
            print(f"  - {file}")
        return True
    else:
        print("✗ No sample documents found")
        return False


def test_chat_completions_parsing():
    """Integration-style tests that assert assistant content is cleanly parsed.

    We patch the initialization to avoid real external calls and provide
    fake qa_chain/llm behaviors to validate parsing of both direct results
    and fallback behavior when the pipeline fails.
    """
    import os
    from fastapi.testclient import TestClient
    import rag_server as server

    # Fake QA that returns a dict-like response
    class FakeQA:
        def invoke(self, payload):
            # Return nested structure similar to LCEL runs
            return {"choices": [{"message": {"content": "FastAPI is a high-performance Python web framework"}}]}

    # Fake QA that returns a string with appended metadata
    class FakeQAMetaStr:
        def invoke(self, payload):
            return "content='FastAPI is a modern web framework' additional_kwargs={'refusal': None} response_metadata={'token_usage':...}"

    # Fake LLM used by fallback
    class FakeLLM:
        def invoke(self, prompt):
            return "content='Fallback: FastAPI is a modern web framework' response_metadata={'foo': 'bar'}"

    # Patch initialization to set qa_chain/llm without performing real initialization
    def _init_stub(args):
        server.qa_chain = FakeQA()
        server.llm = FakeLLM()

    server.initialize_rag_system = _init_stub

    # Start TestClient which will call the patched initialize function during startup
    client = TestClient(server.app)

    # Ensure qa_chain and llm are set (some TestClient lifecycles don't call our stub reliably)
    server.qa_chain = FakeQA()
    server.llm = FakeLLM()

    # Case 1: qa_chain returns nested dict -> parsed content should be clean
    resp = client.post("/v1/chat/completions", json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "What is FastAPI?"}]})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("choices"), list)
    content = data["choices"][0]["message"]["content"]
    assert "FastAPI" in content
    for bad in ["additional_kwargs", "response_metadata", "usage_metadata", "tool_calls"]:
        assert bad not in content

    # Case 1b: qa_chain returns multiple choices - preserve multiple OpenAI-compatible choices
    class FakeMultiQA:
        def invoke(self, payload):
            return {"choices": [
                {"index": 0, "message": {"role": "assistant", "content": "First answer."}, "finish_reason": "stop"},
                {"index": 1, "message": {"role": "assistant", "content": "Second answer."}, "finish_reason": "stop"}
            ], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    server.qa_chain = FakeMultiQA()
    resp_multi = client.post("/v1/chat/completions", json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "What is FastAPI?"}]})
    assert resp_multi.status_code == 200
    data_multi = resp_multi.json()
    assert len(data_multi["choices"]) == 2
    assert data_multi["usage"]["total_tokens"] == 15

    # Case 2: qa_chain returns a metadata string (simulate pipeline issue) -> still parsed
    server.qa_chain = FakeQAMetaStr()
    resp2 = client.post("/v1/chat/completions", json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "What is FastAPI?"}]})
    assert resp2.status_code == 200
    content2 = resp2.json()["choices"][0]["message"]["content"]
    assert content2.startswith("FastAPI") or "FastAPI" in content2
    for bad in ["additional_kwargs", "response_metadata", "usage_metadata", "tool_calls"]:
        assert bad not in content2

    # Case 3: qa_chain pipeline fails and LLM invocation raises (e.g., invalid API key) -> use retriever context
    class FakeQAError:
        def invoke(self, payload):
            raise Exception("pipeline error: dict cannot be converted to PyString")

    class FakeLLMRaise:
        def invoke(self, prompt):
            raise Exception("Error code: 401 - {'error': {'message': 'Incorrect API key', 'code': 'invalid_api_key'}}")

    server.qa_chain = FakeQAError()
    server.llm = FakeLLMRaise()

    resp3 = client.post("/v1/chat/completions", json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "What is FastAPI?"}]})
    assert resp3.status_code == 200
    content3 = resp3.json()["choices"][0]["message"]["content"]
    # No retriever fallback: when the pipeline and external LLM both fail, the server echoes the user's question
    assert content3.strip() == "What is FastAPI?" or content3.strip().startswith("What is FastAPI")

    print("✓ Chat completion parsing tests passed")
    return True


def test_live_integration_with_openai_key():
    """Live integration test that starts the server and calls the real endpoint.

    This test only runs when the environment variable `OPENAI_API_KEY` is present.
    It starts the server on port 8001, waits for `/health` to report `rag_initialized`,
    then POSTs to `/v1/chat/completions` and validates the response is OpenAI-compatible
    and that the `message.content` does not include appended metadata.
    """
    import os
    import subprocess
    import time
    import requests
    import signal
    import pytest

    # Load .env from the project root if present so the test can pick up stored keys
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        from dotenv import load_dotenv
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
    except Exception:
        # dotenv not available or failed; proceed with current env
        pass

    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("No OPENAI_API_KEY set in environment or .env; skipping live integration test")

    port = 8001
    env = os.environ.copy()
    # Ensure server uses provided env key (including values loaded from .env)
    proc = subprocess.Popen(
        ["python", "rag_server.py", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    try:
        # Wait for server to be healthy
        timeout = 60
        for _ in range(timeout):
            try:
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if r.status_code == 200 and r.json().get("rag_initialized") is True:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            proc.kill()
            out, _ = proc.communicate(timeout=1)
            raise AssertionError("Server did not become healthy in time. Logs:\n" + (out.decode(errors='ignore') if out else ''))

        # Send a real request
        r2 = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "What is FastAPI?"}]},
            timeout=60,
        )
        assert r2.status_code == 200
        data = r2.json()
        assert isinstance(data.get("choices"), list)
        content = data["choices"][0]["message"]["content"]
        # Ensure metadata isn't embedded in the content
        for bad in ["additional_kwargs", "response_metadata", "usage_metadata", "tool_calls"]:
            assert bad not in content

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass



def main():
    """Run all tests."""
    print("=" * 60)
    print("RAG FastAPI Server Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_server_structure,
        test_request_models,
        test_documents_exist,
        test_chat_completions_parsing,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
