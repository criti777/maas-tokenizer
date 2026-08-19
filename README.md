# MaaS Tokenizer

一个只做提示词 token 计数的离线 HTTP 服务：接收 OpenAI Chat Completions 风格请求，按照指定模型完成请求规范化、聊天模板渲染和 tokenizer encode，最终只返回 token 数量。它不加载模型权重，也不执行推理。

## 支持的模型

- `deepseek-v3`
- `deepseek-v3.2`
- `deepseek-v4`
- `kimi-k2.6`
- `glm-5.1`
- `glm-5.2`
- `minimax-m2.7`

模型资产固定在 `model_assets/<model>/`，包括 tokenizer、chat template 和必要配置，不包括权重。`models/profiles.json` 负责把请求中的模型名统一路由到对应资产和 renderer。DeepSeek V3.2/V4 使用从 vLLM 提取的专用路径，其余模型使用固定的 Hugging Face chat-template 路径。

## 安装与启动

要求 Python 3.11+：

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/uvicorn maas_tokenizer.api:app --host 0.0.0.0 --port 8080
```

服务启动时只读取模型注册表。某个模型第一次收到请求时才校验并加载其 tokenizer/template，之后在进程内缓存；不同模型分别加载和缓存。

生产多 worker 部署时，每个 worker 都有自己的缓存，因此每个 worker 会各自加载一份用到的 tokenizer。

## API

### `POST /v1/token-count`

请求体沿用 OpenAI Chat Completions 风格，至少需要 `model` 和 `messages`：

```bash
curl http://127.0.0.1:8080/v1/token-count \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5.2",
    "messages": [
      {"role": "system", "content": "你是一个助手。"},
      {"role": "user", "content": "你好"}
    ]
  }'
```

成功响应只有最终数量：

```json
{"token_count": 18}
```

工具定义、tool call、thinking/reasoning 等字段会按对应模型的固定规则进入规范化和渲染。Kimi K2.6 可在纯文本阶段渲染媒体占位符；其他模型遇到图片、音频或视频 content part 时返回 `501 processor_required`，不会下载媒体，也不会伪造视觉 token 数量。

错误响应使用 FastAPI 的 `detail`，并标明失败阶段：

```json
{
  "detail": {
    "stage": "profile_resolution",
    "type": "unknown_model",
    "message": "..."
  }
}
```

主要状态码：

- `400`：请求字段或消息结构不能处理；
- `404`：模型未登记；
- `422`：请求体不是合法 JSON/对象；
- `501`：该多模态请求需要实际 processor；
- `500`：本地模型资产完整性异常或内部错误。

## 代码链路

```text
src/maas_tokenizer/api.py
  -> service.py
  -> registry.py + models/profiles.json
  -> assets.py + models/manifests/ + model_assets/
  -> renderers.py
       -> vendor/vllm/extracted/（vLLM 规范化/专用 renderer）
       -> Transformers / tiktoken（encode）
  -> len(token_ids)
```

- `api.py`：HTTP 入口与错误状态映射；
- `service.py`：统一处理流程、模型级懒加载和线程安全缓存；
- `registry.py`：严格解析七个固定模型，不做未知模型回退；
- `assets.py`：加载前检查所需本地资产；
- `protocol.py`：OpenAI 风格请求结构；
- `renderers.py`：按 profile 分派 HF、DeepSeek V3.2 或 DeepSeek V4 路径；
- `vendor/vllm/extracted/`：从固定 vLLM 路径提取的必要文本处理代码；
- `model_assets/`：每个模型各自的 tokenizer/template/config；
- `tests/`：API、缓存、模型渲染、特殊路径和打包边界测试。

## 测试与覆盖率

安装测试依赖：

```bash
.venv/bin/pip install -e '.[test]'
```

运行不加载全部模型的快速测试：

```bash
.venv/bin/pytest
```

只验证一个模型：

```bash
.venv/bin/pytest --model glm-5.2
```

验证全部七个模型并检查流水线覆盖率门禁：

```bash
.venv/bin/pytest --model all \
  --cov=maas_tokenizer \
  --cov=vendor.vllm.extracted \
  --cov-report=term-missing
```

覆盖率低于 85% 时测试失败。测试不需要模型权重或外部模型服务。

## 范围边界

本项目只计算多模态 processor 之前的文本 token 数。它不包含数据集生成、JSONL 批处理、结果哈希、模型权重、图片下载/解码、pixel values、视觉 embedding 或模型推理。

第三方代码与资产来源见 `THIRD_PARTY_NOTICES.md`；vendored 源文件保留原有 SPDX 声明。
