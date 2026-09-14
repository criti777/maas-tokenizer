# Kimi Image Token Counting Implementation Plan

> Execute inline: user explicitly confirmed implementation, no subagents. The optional executing-plans skill is unavailable; execute these steps directly.

**Goal:** Add image geometry counts for Kimi K2.6/K3 without changing metadata-free requests.
**Architecture:** Validate top-level metadata in service, compute geometry in a pure module, pass K3 image prompts to its existing renderer; add visual counts minus one structural media pad per image.
**Tech Stack:** Python, existing Pydantic request, pytest, existing tokenizers. No new runtime dependency.

## Global Constraints

- No pixel decoding/network at runtime; image shape is [width, height].
- Missing/null metadata preserves existing behavior; malformed supplied metadata fails before loading assets.
- Only image/image_url parts for the two Kimi profiles. No video/audio counting.
- Do not modify model assets, expose ports, deploy or push.

## Task 1: Geometry and metadata tests

Files: tests/test_image_tokens.py, src/maas_tokenizer/image_tokens.py.
- [ ] Write service test asserting count(with_metadata) == count(without_metadata) + 637 for one 800x600 K2.6 image.
- [ ] Run `.venv/bin/python -m pytest tests/test_image_tokens.py --model all -q`; verify missing-feature failures.
- [ ] Implement image_geometry(width, height, profile_id) using official scale min(1, sqrt(budget / patch_product), 7168/width, 7168/height), truncate then pad to 28.
- [ ] Validate metadata list, positive integer dimensions, image count and types before rendering.
- [ ] Compare deterministic/random dimensions against pinned official navit_resize_image results.

## Task 2: Rendering integration

Files: src/maas_tokenizer/service.py, src/maas_tokenizer/renderers.py, tests/test_image_tokens.py.
- [ ] Supply K3 prompts containing original dimensions via a copied parsed request's chat_template_kwargs.
- [ ] Reject conflicting custom image prompts/templates and ambiguous literal K3 image placeholders in metadata mode; these must not consume structural image metadata.
- [ ] Count structural image occurrences rather than scanning user token IDs. Apply sum(visual_count - 1) after rendering.
- [ ] Test no mutation, repeated hashes, mixed text, multi-message/multi-image, URL omitted, null/absent metadata parity, invalid metadata, custom template protection and K3 literal control text.

## Task 3: Verification and documentation

Files: README.md, tests/test_image_tokens.py.
- [ ] Run API TestClient checks with real service and temporary log paths; no server port.
- [ ] Document full top-level JSON example, count scope and errors.
- [ ] Run `.venv/bin/python -m pytest --model all -q` and `git diff --check`.
- [ ] Review diff for unrelated changes, report actual results and unpushed branch.

## Execution completed

All three tasks above were executed inline. Initial tests produced 10 expected
failures because the old service ignored metadata. Image tests: 360 passed,
including 320 pinned official geometry cases. Full suite with all models:
566 passed; total coverage 92.64%, image module coverage 98.92%.
No renderer/vendor/model asset changes, no server port, no remote push.
# 后续简化修订

用户确认改为独立图片累加：先更新回归测试，确认旧代码失败；移除图片对应关系和文本修改；更新 README；执行 --model all 全量回归。下文为首次实现的历史计划，最新行为以 README 和修订设计为准。
