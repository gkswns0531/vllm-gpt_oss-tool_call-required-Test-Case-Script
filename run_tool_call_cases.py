#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from difflib import SequenceMatcher
from typing import Any

import requests


def _post(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout_s,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    obj = resp.json()
    if not isinstance(obj, dict):
        raise RuntimeError("Response JSON is not an object")
    return obj


def _get_message(obj: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("Missing choices[0]")
    ch0 = choices[0]
    msg = ch0.get("message")
    if not isinstance(msg, dict):
        raise RuntimeError("Missing choices[0].message")
    return ch0, msg


def _tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    tc = msg.get("tool_calls") or []
    if not isinstance(tc, list):
        return []
    out: list[dict[str, Any]] = []
    for x in tc:
        if isinstance(x, dict):
            out.append(x)
    return out


def _fn_name(tc: dict[str, Any]) -> str | None:
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return None
    name = fn.get("name")
    return name if isinstance(name, str) else None


def _fn_args(tc: dict[str, Any]) -> str:
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return ""
    args = fn.get("arguments")
    return args if isinstance(args, str) else ""


TOOLS_WEATHER_LOCATION: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_location",
            "description": "Get the user's current location.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


TOOLS_COMPLEX: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local time for a given city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_stock_price",
            "description": "Look up the latest stock price for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "currency": {"type": "string", "enum": ["USD", "KRW", "EUR"]},
                },
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_planner",
            "description": "Plan a route between two places.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "mode": {"type": "string", "enum": ["driving", "walking", "transit"]},
                    "avoid_tolls": {"type": "boolean"},
                },
                "required": ["origin", "destination"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a calendar event and invite attendees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start_time": {
                        "type": "string",
                        "description": "ISO-8601 timestamp, e.g. 2026-01-29T09:00:00-05:00",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "ISO-8601 timestamp, e.g. 2026-01-29T10:00:00-05:00",
                    },
                    "timezone": {"type": "string"},
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Email addresses",
                    },
                    "location": {"type": "string"},
                },
                "required": ["title", "start_time", "end_time"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate a short piece of text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_language": {
                        "type": "string",
                        "enum": ["English", "Korean", "Japanese", "Spanish"],
                    },
                },
                "required": ["text", "target_language"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": "Summarize a document into a small number of bullet points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document": {"type": "string"},
                    "max_bullets": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["document"],
                "additionalProperties": False,
            },
        },
    },
]


def _allowed_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
    return names


def _validate_tool_calls(
    tool_calls: list[dict[str, Any]],
    *,
    allowed_names: set[str],
) -> tuple[bool, str | None]:
    if not tool_calls:
        return False, "missing_tool_calls"

    for tc in tool_calls:
        name = _fn_name(tc)
        if not name or name not in allowed_names:
            return False, f"unexpected_tool_name:{name}"
        raw_args = _fn_args(tc)
        # OpenAI-style tool_calls expects arguments to be a JSON string.
        try:
            args_obj = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            return False, "invalid_json_arguments"
        if not isinstance(args_obj, dict):
            return False, "arguments_not_object"

    return True, None


def _stream_chunks(
    url: str, payload: dict[str, Any], timeout_s: float
) -> list[dict[str, Any]]:
    """
    Minimal SSE client for OpenAI-compatible Chat Completions streaming.

    Collects all JSON "data:" chunks until [DONE].
    """
    payload = dict(payload)
    payload["stream"] = True

    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        stream=True,
        timeout=timeout_s,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    chunks: list[dict[str, Any]] = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        s = line.strip()
        if not s.startswith("data:"):
            continue
        data = s[len("data:") :].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if isinstance(obj, dict):
            chunks.append(obj)

    return chunks


def _extract_reasoning_and_calls_from_chunks(
    chunks: list[dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    """
    Extract accumulated reasoning text and tool call arguments from streaming chunks.

    This follows the OpenAI streaming shape:
      chunk.choices[0].delta.reasoning_content
      chunk.choices[0].delta.tool_calls[*].function.{name,arguments}
    """
    reasoning: str = ""
    tool_calls: dict[int, dict[str, str]] = {}

    for chunk in chunks:
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        ch0 = choices[0]
        if not isinstance(ch0, dict):
            continue
        delta = ch0.get("delta")
        if not isinstance(delta, dict):
            continue

        rc = delta.get("reasoning_content")
        if isinstance(rc, str) and rc:
            reasoning += rc
        r = delta.get("reasoning")
        if isinstance(r, str) and r and not rc:
            reasoning += r

        tcs = delta.get("tool_calls") or []
        if not isinstance(tcs, list):
            continue

        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            idx = tc.get("index", 0)
            if not isinstance(idx, int):
                idx = 0
            entry = tool_calls.setdefault(idx, {"name": "", "arguments": ""})
            fn = tc.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
                if isinstance(name, str) and name:
                    entry["name"] = name
                args = fn.get("arguments")
                if isinstance(args, str) and args:
                    entry["arguments"] += args

    function_names = [v["name"] for _, v in sorted(tool_calls.items())]
    arguments = [v["arguments"] for _, v in sorted(tool_calls.items())]
    return reasoning, arguments, function_names


def _tool_schemas_by_name(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        schema = fn.get("parameters")
        if isinstance(name, str) and isinstance(schema, dict):
            out[name] = schema
    return out


def _schema_validate_object(args_obj: dict[str, Any], schema: dict[str, Any]) -> bool:
    # This is intentionally lightweight (no external dependencies).
    if schema.get("type") != "object":
        return True
    props = schema.get("properties")
    if not isinstance(props, dict):
        props = {}
    required = schema.get("required")
    if not isinstance(required, list):
        required = []
    for k in required:
        if isinstance(k, str) and k not in args_obj:
            return False
    if schema.get("additionalProperties") is False:
        allowed = {k for k in props.keys() if isinstance(k, str)}
        for k in args_obj.keys():
            if k not in allowed:
                return False

    for k, v in args_obj.items():
        if k not in props:
            continue
        prop = props.get(k)
        if not isinstance(prop, dict):
            continue
        t = prop.get("type")
        if t == "string" and not isinstance(v, str):
            return False
        if t == "integer" and not isinstance(v, int):
            return False
        if t == "boolean" and not isinstance(v, bool):
            return False
        if t == "array":
            if not isinstance(v, list):
                return False
            items = prop.get("items")
            if isinstance(items, dict) and items.get("type") == "string":
                if not all(isinstance(x, str) for x in v):
                    return False
        enum = prop.get("enum")
        if isinstance(enum, list) and enum and v not in enum:
            return False
        if isinstance(t, str) and t in ("integer",) and isinstance(v, int):
            mn = prop.get("minimum")
            mx = prop.get("maximum")
            if isinstance(mn, int) and v < mn:
                return False
            if isinstance(mx, int) and v > mx:
                return False

    return True


TOOLS_ENTRYPOINT_EXAMPLE: list[dict[str, Any]] = [
    t
    for t in TOOLS_COMPLEX
    if isinstance(t, dict)
    and isinstance(t.get("function"), dict)
    and t["function"].get("name") in {"calculator", "get_time"}
]


def case_no_tools(*, url: str, model: str, timeout_s: float) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say: This is a test"}],
        "temperature": 0.0,
    }
    obj = _post(url, payload, timeout_s)
    _ch0, msg = _get_message(obj)
    if _tool_calls(msg):
        raise RuntimeError("Unexpected tool_calls in no-tools case")
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Expected non-empty content in no-tools case")


def case_single_tool_tokyo(*, url: str, model: str, timeout_s: float) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
        "tools": TOOLS_WEATHER_LOCATION,
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    obj = _post(url, payload, timeout_s)
    ch0, msg = _get_message(obj)
    tool_calls = _tool_calls(msg)
    if not tool_calls:
        raise RuntimeError("No tool_calls detected")
    if ch0.get("finish_reason") not in (None, "tool_calls"):
        raise RuntimeError(f"Unexpected finish_reason: {ch0.get('finish_reason')}")

    names = [_fn_name(tc) for tc in tool_calls]
    if "get_current_weather" not in names:
        raise RuntimeError(f"Expected get_current_weather; got {names}")

    idx = names.index("get_current_weather")
    raw_args = _fn_args(tool_calls[idx])
    if not raw_args:
        raise RuntimeError("Missing get_current_weather arguments")
    parsed = json.loads(raw_args)
    loc = parsed.get("location")
    if not isinstance(loc, str) or "tokyo" not in loc.lower():
        raise RuntimeError(f"Expected location=Tokyo; got {parsed}")


def case_multiple_tools_weather_and_location(*, url: str, model: str, timeout_s: float) -> None:
    user_prompt = "First call get_user_location. Then call get_current_weather for Tokyo."

    payload1 = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "tools": TOOLS_WEATHER_LOCATION,
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    obj1 = _post(url, payload1, timeout_s)
    _ch01, msg1 = _get_message(obj1)
    tool_calls1 = _tool_calls(msg1)
    if not tool_calls1:
        raise RuntimeError("No tool_calls detected in step 1")

    # Best-effort: accept either single-call (common) or multi-call (rarer).
    names1 = [_fn_name(tc) for tc in tool_calls1]
    if "get_user_location" in names1 and "get_current_weather" in names1:
        return

    first = tool_calls1[0]
    tool_call_id1 = first.get("id")
    if not isinstance(tool_call_id1, str) or not tool_call_id1:
        raise RuntimeError("Missing tool_call id in step 1")

    first_name = _fn_name(first)
    if first_name not in ("get_user_location", "get_current_weather"):
        raise RuntimeError(f"Unexpected first tool: {first_name}")

    # Provide a stub tool result, then ask the model to continue.
    if first_name == "get_user_location":
        tool_result = {"location": "Tokyo"}
    else:
        tool_result = {"location": "Tokyo", "temp_c": 25, "condition": "Sunny"}

    payload2 = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "tool_calls": tool_calls1, "content": None},
            {
                "role": "tool",
                "tool_call_id": tool_call_id1,
                "content": json.dumps(tool_result),
            },
            {"role": "user", "content": "Continue."},
        ],
        "tools": TOOLS_WEATHER_LOCATION,
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    obj2 = _post(url, payload2, timeout_s)
    _ch02, msg2 = _get_message(obj2)
    tool_calls2 = _tool_calls(msg2)
    if not tool_calls2:
        raise RuntimeError("No tool_calls detected in step 2")

    names2 = [_fn_name(tc) for tc in tool_calls2]
    if first_name == "get_user_location":
        expected = "get_current_weather"
    else:
        expected = "get_user_location"
    if expected not in names2:
        raise RuntimeError(f"Expected {expected} in step 2; got {names2}")


def case_tool_then_final_text(*, url: str, model: str, timeout_s: float) -> None:
    payload1 = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
        "tools": TOOLS_WEATHER_LOCATION,
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    obj1 = _post(url, payload1, timeout_s)
    _ch0, msg1 = _get_message(obj1)
    tool_calls1 = _tool_calls(msg1)
    if not tool_calls1:
        raise RuntimeError("No tool_calls detected in step 1")
    first = tool_calls1[0]
    tool_call_id = first.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise RuntimeError("Missing tool_call id in step 1")
    name = _fn_name(first)
    if name != "get_current_weather":
        raise RuntimeError(f"Expected get_current_weather as first call; got {name}")

    payload2 = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is the weather in Tokyo?"},
            {"role": "assistant", "tool_calls": tool_calls1, "content": None},
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(
                    {"location": "Tokyo", "temp_c": 25, "condition": "Sunny"}
                ),
            },
        ],
        "temperature": 0.0,
    }
    obj2 = _post(url, payload2, timeout_s)
    _ch02, msg2 = _get_message(obj2)
    if _tool_calls(msg2):
        raise RuntimeError("Unexpected tool_calls in step 2 (expected final text)")
    content = msg2.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Expected non-empty content in step 2")


def case_tool_choice_required_batch(
    *,
    url: str,
    model: str,
    timeout_s: float,
    n: int,
    seed: int,
) -> None:
    """
    Send N requests with tool_choice="required" and report success rate.

    "Success" means:
    - HTTP 200
    - message.tool_calls is present and non-empty
    - each tool call has a known tool name and JSON object arguments
    """
    if n <= 0:
        raise RuntimeError("n must be > 0")

    rng = random.Random(seed)
    prompts: list[str] = [
        "Use the calculator to compute (123 + 456) * 7.",
        "Get the current time in New York.",
        "Search the web for the latest vLLM release notes and return the top 3 results.",
        "Look up the latest stock price for AAPL in USD.",
        "Plan a driving route from San Francisco to San Jose and avoid tolls.",
        (
            "Create a calendar event titled 'Design Review' from 2026-02-03T09:00:00-08:00 "
            "to 2026-02-03T10:00:00-08:00, invite alice@example.com and bob@example.com, "
            "and set the location to 'Conf Room A'."
        ),
        "Translate 'Server is running smoothly.' to Korean.",
        "Summarize the following document into at most 5 bullet points: 'vLLM is a fast LLM inference engine.'",
        (
            "I am coordinating a trip. First find the local time in Tokyo, then calculate 7*8, "
            "then plan a transit route from Shinjuku to Akihabara."
        ),
    ]

    allowed_names = _allowed_tool_names(TOOLS_COMPLEX)

    stats = {
        "ok": 0,
        "no_tool_calls": 0,
        "invalid_tool_calls": 0,
        "http_error": 0,
        "other_error": 0,
    }

    for i in range(1, n + 1):
        prompt = rng.choice(prompts)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS_COMPLEX,
            "tool_choice": "required",
            "temperature": 0.0,
        }
        try:
            obj = _post(url, payload, timeout_s)
            _ch0, msg = _get_message(obj)
            tool_calls = _tool_calls(msg)
            ok, reason = _validate_tool_calls(tool_calls, allowed_names=allowed_names)
            if ok:
                stats["ok"] += 1
            else:
                if reason == "missing_tool_calls":
                    stats["no_tool_calls"] += 1
                else:
                    stats["invalid_tool_calls"] += 1
        except RuntimeError:
            stats["http_error"] += 1
        except Exception:
            stats["other_error"] += 1

        if i % 10 == 0 or i == n:
            print(f"Progress: {i}/{n} requests completed", flush=True)

    total = n
    ok = stats["ok"]
    success_rate = (ok / total) * 100.0 if total else 0.0

    print("\n=== tool_choice=\"required\" batch summary ===")
    print(f"Model: {model}")
    print(f"URL:   {url}")
    print(f"Runs:  {total}")
    print(f"OK:    {ok}")
    print(f"Success rate: {success_rate:.1f}%")
    print("--- Breakdown ---")
    print(f"no_tool_calls:      {stats['no_tool_calls']}")
    print(f"invalid_tool_calls: {stats['invalid_tool_calls']}")
    print(f"http_error:         {stats['http_error']}")
    print(f"other_error:        {stats['other_error']}")

    # Fail the case if not 100% to make CI-style usage possible.
    if ok != total:
        raise RuntimeError(f"Success rate was {success_rate:.1f}% ({ok}/{total})")


def case_stream_get_time_with_reasoning(*, url: str, model: str, timeout_s: float) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the current time in New York?"}],
        "tools": TOOLS_ENTRYPOINT_EXAMPLE,
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    chunks = _stream_chunks(url, payload, timeout_s)
    reasoning, arguments, function_names = _extract_reasoning_and_calls_from_chunks(chunks)

    if "get_time" not in function_names:
        raise RuntimeError(f"get_time function not called (found: {function_names})")
    if not any("New York" in a for a in arguments):
        raise RuntimeError(f"Expected get_time arguments for New York (found: {arguments})")
    if not reasoning.strip():
        raise RuntimeError("Expected non-empty reasoning_content in streaming response")
    if not any(k in reasoning for k in ("New York", "time", "current")):
        raise RuntimeError("Reasoning content is not relevant to the request")


def case_stream_multiple_tools(*, url: str, model: str, timeout_s: float) -> None:
    system_msg = (
        "You can call multiple tools. If more than one tool is required, you may call them "
        "either in a single assistant message or across multiple turns (tool loop)."
    )
    user_prompt = (
        "First, calculate 7 * 8 using the calculator. "
        "Then, use get_time to tell me the current time in New York."
    )

    payload1 = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        "tools": TOOLS_ENTRYPOINT_EXAMPLE,
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    chunks1 = _stream_chunks(url, payload1, timeout_s)
    reasoning1, _arguments1, function_names1 = _extract_reasoning_and_calls_from_chunks(
        chunks1
    )

    if "calculator" not in function_names1:
        raise RuntimeError(f"Calculator tool missing (found: {function_names1})")
    if "get_time" in function_names1:
        if not reasoning1.strip():
            raise RuntimeError("Expected non-empty reasoning_content in streamed response")
        return

    # Tool-loop continuation: provide a stub calculator result and ask the model to proceed.
    payload2 = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_calculator_1",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"expression":"7 * 8"}'},
                    }
                ],
                "content": None,
            },
            {"role": "tool", "tool_call_id": "call_calculator_1", "content": "56"},
            {"role": "user", "content": "Continue."},
        ],
        "tools": TOOLS_ENTRYPOINT_EXAMPLE,
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    chunks2 = _stream_chunks(url, payload2, timeout_s)
    reasoning2, _arguments2, function_names2 = _extract_reasoning_and_calls_from_chunks(
        chunks2
    )
    if "get_time" not in function_names2:
        raise RuntimeError(f"Time tool missing (found: {function_names2})")
    if not (reasoning1 + reasoning2).strip():
        raise RuntimeError("Expected non-empty reasoning_content in streamed response")


def case_invalid_tool_call(*, url: str, model: str, timeout_s: float) -> None:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Can you help with something, but don’t actually perform any calculation?"
                ),
            }
        ],
        "tools": TOOLS_ENTRYPOINT_EXAMPLE,
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    obj = _post(url, payload, timeout_s)
    _ch0, msg = _get_message(obj)
    if _tool_calls(msg):
        raise RuntimeError("Model unexpectedly produced tool_calls on invalid input")


def case_tool_call_with_temperature(*, url: str, model: str, timeout_s: float) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Calculate 123 + 456 using the calculator."}],
        "tools": TOOLS_ENTRYPOINT_EXAMPLE,
        "tool_choice": "auto",
        "temperature": 0.7,
    }
    obj = _post(url, payload, timeout_s)
    _ch0, msg = _get_message(obj)
    if not (_tool_calls(msg) or (isinstance(msg.get("content"), str) and msg["content"].strip())):
        raise RuntimeError("Response missing both text content and tool calls")


def case_tool_response_schema_accuracy(*, url: str, model: str, timeout_s: float) -> None:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "First, calculate 7 * 8 using the calculator. "
                    "Then, use get_time to tell me the current time in New York."
                ),
            }
        ],
        "tools": TOOLS_ENTRYPOINT_EXAMPLE,
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    obj = _post(url, payload, timeout_s)
    _ch0, msg = _get_message(obj)
    calls = _tool_calls(msg)
    if not calls:
        raise RuntimeError("No tool calls produced")

    schemas = _tool_schemas_by_name(TOOLS_ENTRYPOINT_EXAMPLE)
    for call in calls:
        name = _fn_name(call)
        if not name or name not in schemas:
            raise RuntimeError(f"No matching tool schema found for {name}")
        raw = _fn_args(call)
        args_obj = json.loads(raw) if raw.strip() else {}
        if not isinstance(args_obj, dict):
            raise RuntimeError("Tool arguments must be a JSON object")
        if not _schema_validate_object(args_obj, schemas[name]):
            raise RuntimeError(f"Tool call arguments do not match schema for {name}: {args_obj}")


def case_semantic_consistency_with_temperature(*, url: str, model: str, timeout_s: float) -> None:
    """
    Check that higher temperature does not cause extreme semantic drift in plain text.

    This is intentionally tool-free to ensure message.content is present.
    """
    prompt = "Compute 123 + 456. Reply with just the number."
    expected = "579"

    for temp in (0.0, 0.5, 1.0):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
        }
        obj = _post(url, payload, timeout_s)
        _ch0, msg = _get_message(obj)
        text = msg.get("content")
        if not isinstance(text, str):
            raise RuntimeError("Expected text content for semantic consistency test")
        got = "".join(ch for ch in text.strip() if ch.isdigit())
        if expected not in got:
            raise RuntimeError(
                f"Expected answer {expected} across temperatures; got '{text.strip()}' at T={temp}"
            )


CASES = {
    "no_tools": case_no_tools,
    "single_tool_tokyo": case_single_tool_tokyo,
    "multiple_tools": case_multiple_tools_weather_and_location,
    "tool_then_final_text": case_tool_then_final_text,
    "tool_choice_required_batch": case_tool_choice_required_batch,
    "stream_get_time_with_reasoning": case_stream_get_time_with_reasoning,
    "stream_multiple_tools": case_stream_multiple_tools,
    "invalid_tool_call": case_invalid_tool_call,
    "tool_call_with_temperature": case_tool_call_with_temperature,
    "tool_response_schema_accuracy": case_tool_response_schema_accuracy,
    "semantic_consistency_with_temperature": case_semantic_consistency_with_temperature,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--case", action="append", default=[], help="Case name to run")
    ap.add_argument(
        "--n", type=int, default=100, help="Runs for batch cases (default: 100)"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for batch prompt selection (default: 1337)",
    )
    args = ap.parse_args()

    # By default, run the lightweight, single-request smoke tests.
    # Batch tests are opt-in because they are intentionally expensive.
    selected = args.case or [
        k for k in CASES.keys() if k != "tool_choice_required_batch"
    ]
    unknown = [c for c in selected if c not in CASES]
    if unknown:
        raise SystemExit(f"Unknown cases: {unknown}. Available: {sorted(CASES)}")

    failures: list[tuple[str, str]] = []
    for name in selected:
        try:
            if name == "tool_choice_required_batch":
                CASES[name](
                    url=args.url,
                    model=args.model,
                    timeout_s=args.timeout_s,
                    n=args.n,
                    seed=args.seed,
                )
            else:
                CASES[name](url=args.url, model=args.model, timeout_s=args.timeout_s)
            print(f"[PASS] {name}")
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {name} - {type(e).__name__}: {e}")

    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"- {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
