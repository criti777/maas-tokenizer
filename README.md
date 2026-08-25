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
.venv/bin/uvicorn maas_tokenizer.api:app --host 0.0.0.0 --port 8080 --workers 1
```

服务启动阶段会通过正式计数链路预热 `glm-5.2`，预热成功后才开始接收流量；其他模型第一次收到请求时才校验并加载其 tokenizer/template。所有已加载模型之后都在进程内按 profile 缓存。

生产部署必须保持 `--workers 1`。每个 Pod 内只有一个计算线程串行执行 tokenizer 请求；多 worker 会各自创建队列和缓存，无法保证 Pod 级串行。

## Pod 内排队与限流

`/tokenizer` 使用有界 FIFO 队列。默认最多有 1 个请求执行、100 个请求等待：队列已满时立即返回 `429 queue_full`；请求排队 2 秒仍未开始执行时返回 `429 queue_timeout`。两种响应都包含 `Retry-After: 1`，超时任务不会在后台继续计算。

配置项：

```text
TOKENIZER_QUEUE_SIZE=100
TOKENIZER_QUEUE_TIMEOUT_SECONDS=2
TOKENIZER_LOG_PATH=/opt/cloud/logs/access.log
TOKENIZER_LOG_MAX_BYTES=104857600
TOKENIZER_LOG_BACKUP_COUNT=5
TOKENIZER_LOG_REQUEST_BODY=false
TOKENIZER_LOG_REQUEST_BODY_MAX_BYTES=65536
```

`GET /health` 不进入计算队列，可直接用于 Kubernetes 存活探针。

## 访问日志

每个 `/tokenizer` 请求输出一条单行日志，同时写入 stdout 和 `TOKENIZER_LOG_PATH`。日志包含 `X-Span-Id`（缺失时自动生成）、model、成功/失败/拒绝状态、HTTP 状态、排队耗时、计算耗时和总耗时。请求正文默认不记录。

联调时可设置 `TOKENIZER_LOG_REQUEST_BODY=true`，把 FastAPI 已解析的请求对象以紧凑 JSON 追加到同一行。JSON 的 UTF-8 大小不超过 `TOKENIZER_LOG_REQUEST_BODY_MAX_BYTES` 时记录完整正文；超过限制时只记录实际字节数和 `request_body=<omitted_too_large>`，不会截断正文。开启后日志可能包含完整消息内容，请只在允许保留请求内容的环境使用。

访问日志使用 `Asia/Shanghai` 时区，时间格式为 `yyyy-MM-dd HH:mm:ss.SSS`，字段之间用 `|` 分隔。请求头 `X-Span-Id` 在日志中的字段名为 `x_span_id`；浮点数保留两位小数，整数按原值输出，空值输出为空字符串。字段值中的 `|` 编码为 `%7C`，CR、LF 和 Tab 分别编码为 `\r`、`\n` 和 `\t`，确保每个请求只占一行。

默认格式：

```text
timestamp=<时间>|x_span_id=<请求标识>|model=<模型>|status=<状态>|reason=<原因>|http_status=<状态码>|queue_wait_ms=<排队毫秒>|process_ms=<处理毫秒>|total_ms=<总毫秒>
```

示例：

```text
timestamp=2026-08-25 15:30:12.083|x_span_id=abc-123|model=glm-5.2|status=success|reason=|http_status=200|queue_wait_ms=0.03|process_ms=13.65|total_ms=14.73
```

默认日志文件达到 100 MB 后轮转并保留 5 份。容器运行用户必须能够创建和写入 `/opt/cloud/logs`；在 CCE 中应把日志卷挂载到该目录并赋予 UID/GID 1000 写权限。

## API

### `POST /tokenizer`

请求体沿用 OpenAI Chat Completions 风格，至少需要 `model` 和 `messages`：

```bash
curl http://127.0.0.1:8080/tokenizer \
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
18
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
- `429`：Pod 等待队列已满或排队超过 2 秒；
- `500`：本地模型资产完整性异常或内部错误。

## 代码链路

```text
src/maas_tokenizer/api.py
  -> service.py
  -> registry.py + models/profiles.json
  -> assets.py + models/manifests/ + model_assets/
  -> renderers.py
       -> vendor/vllm/extracted/（vLLM 规范化/专用 renderer）
       -> AutoTokenizer（模板渲染）+ Gigatoken（encode）
  -> len(token_ids)
```

- `api.py`：HTTP 入口与错误状态映射；
- `scheduler.py`：有界 FIFO、排队超时与单线程串行执行；
- `access_logging.py`：stdout 和轮转文件访问日志；
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
