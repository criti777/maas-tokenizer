# MaaS Tokenizer

一个只做提示词 token 计数的离线 HTTP 服务：接收 OpenAI Chat Completions 风格请求，按照指定模型完成请求规范化、聊天模板渲染和 tokenizer encode，最终只返回 token 数量。它不加载模型权重，也不执行推理。

## 支持的模型

- `deepseek-v3`
- `deepseek-v3.2`
- `deepseek-v4`
  - aliases: `DeepSeek-V4-Flash-0731`, `DeepSeek-V4-Pro-0813`
- `kimi-k2.6`
- `kimi-k3`
- `glm-5.1`
- `glm-5.2`
- `glm-5.3-flash`
- `minimax-m2.7`
- `minimax-m3`

模型资产固定在 `model_assets/<model>/`，包括 tokenizer、chat template 和必要配置，不包括权重。`models/profiles.json` 负责把请求中的模型名统一路由到对应资产和 renderer。DeepSeek V3.2/V4 使用从 vLLM 提取的专用路径；Kimi K3 使用官方 Python XTML 分段渲染；其他模型使用固定的 Hugging Face chat-template 路径。

## 安装与启动

要求 Python 3.11+：

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m maas_tokenizer.main
```

服务启动阶段会通过正式计数链路预热 `glm-5.2`，预热成功后才开始接收流量；其他模型第一次收到请求时才校验并加载其 tokenizer/template。所有已加载模型之后都在进程内按 profile 缓存。

生产部署必须保持 `--workers 1`。每个 Pod 内只有一个计算线程串行执行 tokenizer 请求；多 worker 会各自创建队列和缓存，无法保证 Pod 级串行。

## Pod 内排队与限流

`/tokenizer` 使用有界 FIFO 队列。默认最多有 1 个请求执行、100 个请求等待：队列已满时立即返回 `429 queue_full`；请求排队 200 毫秒仍未开始执行时返回 `429 queue_timeout`。两种响应都包含 `Retry-After: 1`，超时任务不会在后台继续计算。该超时只限制开始执行前的排队时间，不限制已经开始的模板渲染和 tokenizer encode。

配置项：

```text
TOKENIZER_QUEUE_SIZE=100
TOKENIZER_QUEUE_TIMEOUT_MS=200
TOKENIZER_LOG_PATH=/opt/cloud/logs/maas-tokenizer/access.log
TOKENIZER_LOG_MAX_BYTES=104857600
TOKENIZER_LOG_BACKUP_COUNT=5
TOKENIZER_LOG_REQUEST_BODY=false
TOKENIZER_LOG_REQUEST_BODY_MAX_BYTES=65536
TOKENIZER_RUN_LOG_PATH=/opt/cloud/logs/maas-tokenizer/run.log
TOKENIZER_RUN_LOG_MAX_BYTES=104857600
TOKENIZER_RUN_LOG_BACKUP_COUNT=5
TOKENIZER_RUN_LOG_LEVEL=INFO
```

`GET /health` 不进入计算队列，可直接用于 Kubernetes 存活探针。

## 访问日志

每个 `/tokenizer` 请求向 `TOKENIZER_LOG_PATH` 写入一条单行日志，不输出到终端。日志包含请求头中的 `X-Span-Id`、`X-Request-Id` 和 `Content-Length`，以及 model、成功时计算出的 token 数量、错误信息、HTTP 状态、排队耗时、计算耗时和总耗时。请求 ID 缺失时记录空字符串，不在服务内自动生成。请求正文默认不记录。

联调时可设置 `TOKENIZER_LOG_REQUEST_BODY=true`，把 FastAPI 已解析的请求对象以紧凑 JSON 写入最后一列。JSON 的 UTF-8 大小不超过 `TOKENIZER_LOG_REQUEST_BODY_MAX_BYTES` 时记录完整正文；超过限制时在最后一列写入 `<omitted_too_large>`，不会截断正文。开启后日志可能包含完整消息内容，请只在允许保留请求内容的环境使用。

访问日志使用 `Asia/Shanghai` 时区，时间格式为 `yyyy-MM-dd HH:mm:ss.SSS`，字段之间用 `|` 分隔。浮点数保留两位小数，整数按原值输出，空值输出为空字符串。字段值中的 `|` 编码为 `%7C`，CR、LF 和 Tab 分别编码为 `\r`、`\n` 和 `\t`，确保每个请求只占一行。

请求体日志关闭时采用固定 12 列纯 value 格式，不输出字段名：

```text
timestamp|x_span_id|x_request_id|model|content_length|prompt_tokens|error_code|error_message|http_status|queue_wait_ms|process_ms|total_ms
```

请求体日志开启时才在末尾追加第 13 列 `request_body`：

```text
2026-08-25 15:30:12.083|span-123|request-456|glm-5.2|82|18|||200|0.03|13.65|14.73|{"model":"glm-5.2","messages":[{"role":"user","content":"你好"}]}
```

失败时 `prompt_tokens` 为空，并写入错误码和错误信息：

```text
2026-08-25 15:30:15.126|span-123|request-456|glm-5.2|82||queue_full|tokenizer queue is full|429|0.00|0.00|1.28
```

默认日志文件达到 100 MB 后轮转并保留 5 份。容器运行用户必须能够创建和写入 `/opt/cloud/logs`；在 CCE 中应把日志卷挂载到该目录并赋予 UID/GID 1000 写权限。

## 运行日志

服务生命周期、GLM-5.2 预热、模型首次加载、内部 500 异常栈和调度器消费协程异常写入 `TOKENIZER_RUN_LOG_PATH`。run.log 使用 `Asia/Shanghai` 时间、日志级别和单行事件格式；异常 traceback 中的换行会编码为 `\\n`。已知的请求校验、未知模型和限流等预期错误只记录在 access.log，避免重复日志。

生产入口会把 Uvicorn、Python 根日志和 warnings 一并导入 run.log，并关闭 Uvicorn 逐请求日志。access.log 和 run.log 都只写轮转文件，不输出到 stdout/stderr。run.log 默认单文件 100 MB，保留 5 份。

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

成功响应为 JSON 对象：

```json
{"prompt_tokens": 18}
```

工具定义、tool call、thinking/reasoning 等字段会按对应模型的固定规则进入规范化和渲染。Kimi K2.6 和 Kimi K3 可在纯文本阶段渲染媒体占位符；其他模型遇到图片、音频或视频 content part 时返回 `501 processor_required`，不会下载媒体，也不会伪造视觉 token 数量。

Thinking 开关兼容布尔形式、旧字段和对象形式：

```json
{"thinking": true}
{"enable_thinking": false}
{"thinking": {"type": "enabled", "clear_thinking": false}}
```

对象中的 `type` 支持 `enabled`、`disabled` 和 `auto`；空对象按 `enabled` 处理，`auto` 表示不覆盖模型默认值。`clear_thinking` 和顶层 `preserve_thinking` 会继续传入聊天模板。服务会同时补齐模板常用的 `thinking` 与 `enable_thinking` 布尔参数；如果多个显式开关一真一假，则返回 `400 invalid_request`。GLM-5.2 的 `reasoning_effort` 为 `none` 或 `minimal` 时会强制关闭 Thinking。生成采样参数不影响输入 token 计数，因此服务不会改写 `temperature`、`top_p` 或 penalty 字段。

错误响应统一为仅含错误码和错误消息的 JSON 对象：

```json
{
  "error_code": "unknown_model",
  "error_msg": "..."
}
```

主要状态码：

- `400`：请求字段或消息结构不能处理；
- `404`：模型未登记；
- `422`：请求体不是合法 JSON/对象；
- `501`：该多模态请求需要实际 processor；
- `429`：Pod 等待队列已满或排队超过 200 毫秒（默认值）；
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
       -> Kimi K3 官方 XTML segments + 分段 encode
  -> len(token_ids)
```

- `api.py`：HTTP 入口与错误状态映射；
- `scheduler.py`：有界 FIFO、排队超时与单线程串行执行；
- `access_logging.py`：轮转文件访问日志；
- `run_logging.py`：服务运行日志、异常栈和 Uvicorn 文件日志；
- `service.py`：统一处理流程、模型级懒加载和线程安全缓存；
- `registry.py`：严格解析十个固定 profile，不做未知模型回退；
- `assets.py`：加载前检查所需本地资产；
- `protocol.py`：OpenAI 风格请求结构；
- `renderers.py`：按 profile 分派 HF、DeepSeek V3.2、DeepSeek V4 或 Kimi K3 路径；
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

验证全部十个 profile 并检查流水线覆盖率门禁：

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
