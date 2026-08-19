# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-compatible chat template content-format detection."""

from __future__ import annotations

from collections import deque

import jinja2
import jinja2.nodes


def _is_var(node: jinja2.nodes.Node, name: str) -> bool:
    return isinstance(node, jinja2.nodes.Name) and node.ctx == "load" and node.name == name


def _is_attr(node: jinja2.nodes.Node, name: str, key: str) -> bool:
    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var(node.node, name) and node.attr == key
    if isinstance(node, jinja2.nodes.Getitem):
        return (
            _is_var(node.node, name)
            and isinstance(node.arg, jinja2.nodes.Const)
            and node.arg.value == key
        )
    return False


def _related_names(root: jinja2.nodes.Template, initial: str) -> list[str]:
    names: list[str] = [initial]
    queue = deque([initial])
    while queue:
        current = queue.popleft()
        for assignment in root.find_all(jinja2.nodes.Assign):
            if _is_var(assignment.node, current) and isinstance(
                assignment.target, jinja2.nodes.Name
            ):
                name = assignment.target.name
                if name not in names:
                    names.append(name)
                    queue.append(name)
    return names


def detect_content_format(chat_template: str) -> str:
    try:
        import transformers.utils.chat_template_utils as hf_chat_utils

        compiled = hf_chat_utils._compile_jinja_template(chat_template)
        root = compiled.environment.parse(chat_template)
    except Exception:
        return "string"
    message_names: list[str] = []
    for loop in root.find_all(jinja2.nodes.For):
        if isinstance(loop.target, jinja2.nodes.Name) and any(
            _is_var(loop.iter, name) for name in _related_names(root, "messages")
        ):
            message_names.append(loop.target.name)
    for loop in root.find_all(jinja2.nodes.For):
        if any(_is_attr(loop.iter, name, "content") for name in message_names):
            return "openai"
        if isinstance(loop.iter, jinja2.nodes.Name) and loop.iter.name == "content":
            return "openai"
    return "string"
