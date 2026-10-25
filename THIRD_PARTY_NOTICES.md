# Third-party notices

Parts of `vendor/vllm/extracted/` are derived from vLLM v0.26.0, copyright
vLLM contributors, under the Apache License 2.0. The upstream project and
license text are available at <https://github.com/vllm-project/vllm> and
<https://www.apache.org/licenses/LICENSE-2.0>.

The derived source files retain their SPDX copyright and license headers.

Kimi K3 tokenizer configuration and Python encoding files under
`model_assets/moonshotai--Kimi-K3/` are distributed by Moonshot AI under the
Kimi K3 License. The complete upstream license is retained beside those assets
as `LICENSE`; it includes additional terms for certain Model-as-a-Service
commercial use.

DeepSeek V4 input encoding in `vendor/vllm/extracted/deepseek_v4_encoding.py`
is derived from DeepSeek's MIT-licensed `encoding/encoding_dsv4.py` in
`deepseek-ai/DeepSeek-V4-Pro-0813` at revision
`72e1d3230f6c080a530b0a1d46f8eb4602340597`, byte-identical to
`deepseek-ai/DeepSeek-V4-Flash-0731` at
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
Upstream SHA-256: `abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6`.
Only input encoding is retained; upstream assertions are replaced by runtime
validation, and the unused output parsing helpers are omitted. The full upstream
MIT notice is retained in `vendor/vllm/extracted/DEEPSEEK_V4_LICENSE`.

The corresponding tokenizer/config assets are pinned to those same revisions
under `model_assets/deepseek-ai--DeepSeek-V4-Flash-0731/` and
`model_assets/deepseek-ai--DeepSeek-V4-Pro-0813/`; each asset directory retains
the upstream `LICENSE`.

GLM-5.3 tokenizer/config assets and the unmodified `chat_template.jinja` are
distributed by Z.ai in `zai-org/GLM-5.3`, pinned at
`ba32f8c19c8738e2c57c71798c601d477a7bf844` (2026-10-14 tool-result ID
validation and reordering optimization). They are stored under
`model_assets/zai-org--GLM-5.3/`, together with the complete upstream `LICENSE`.
Template SHA-256: `0e2b6268ac54f04d2bc1157e8cfb896cb66f25c0b5e319ea12e56014c3914f24`.
All asset checksums and sizes are recorded in `models/manifests/glm-5.3.json`.
