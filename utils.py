import time
def ai_message_to_chat_completion(m): #is langchain.messages.AIMessage
    g = lambda o, k, d=None: o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)
    d = lambda o: o if isinstance(o, dict) else (getattr(o, "dict", lambda: {})() if o else {})

    try:
        rm, um = d(g(m, "response_metadata")), d(g(m, "usage_metadata"))
        tk = d(rm.get("token_usage"))
        p = tk.get("prompt_tokens", um.get("input_tokens", 0))
        c = tk.get("completion_tokens", um.get("output_tokens", 0))
        t = tk.get("total_tokens", p + c)

        return {
            "id": rm.get("id") or g(m, "id") or f"chatcmpl-{int(time.time()*1e3)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": rm.get("model_name", "unknown"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": g(m, "content")},
                "finish_reason": rm.get("finish_reason", "stop"),
                "logprobs": rm.get("logprobs")
            }],
            "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t},
        }

    except Exception as e:
        return {
            "id": f"chatcmpl-error-{int(time.time()*1e3)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "unknown",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": str(g(m, "content", e))},
                "finish_reason": "stop",
                "logprobs": None
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "error": {"message": str(e)}
        }
