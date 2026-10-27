# MaaS Tokenizer

一个只做提示词 token 计数的离线 HTTP 服务：接收 OpenAI Chat Completions 风格请求，按照指定模型完成请求规范化、聊天模板渲染和 tokenizer encode，最终只返回 token 数量。它不加载模型权重，也不执行推理。

## 支持的模型

- `deepseek-v3`
- `deepseek-v3.2`
- `deepseek-v4-flash`（资产：`deepseek-ai/DeepSeek-V4-Flash-0731`）
- `deepseek-v4-pro`（资产：`deepseek-ai/DeepSeek-V4-Pro-0813`）
- `deepseek-v4.1-flash`（资产：`deepseek-ai/DeepSeek-V4.1-Flash-1017`；同时接受 `deepseek-ai/DeepSeek-V4.1-Flash` 请求名）
- `kimi-k2.6`
- `kimi-k3`
- `glm-5.1`
- `glm-5.2`
- `glm-5.3`
- `glm-5.3-flash`
- `minimax-m2.7`
- `minimax-m3`

模型资产固定在 `model_assets/<model>/`，包括 tokenizer、chat template 和必要配置，不包括权重。`models/profiles.json` 负责把请求中的模型名统一路由到对应资产和 renderer。DeepSeek V3.2 使用从 vLLM 提取的专用路径；V4 Flash/Pro 使用固定版本的官方 Python 消息编码器；V4.1 Flash 使用其固定版本的官方文本编码器；Kimi K3 使用官方 Python XTML 分段渲染；其他模型使用固定的 Hugging Face chat-template 路径。

所有模型统一只统计文本 content。多模态 content parts（如图片、音频、视频及其他非文本类型）会在渲染前移除；即使模型 renderer 能识别多模态标记，也不计入这些 payload。

DS V4 请求仅接受 `deepseek-v4-flash`、`deepseek-v4-pro`（忽略大小写），不接受旧 `deepseek-v4` 或 Hugging Face 仓库名。两者独立加载对应资产，共用新版官方渲染器：思考模式下 `low` 不加前缀，`high`、`max` 使用各自官方前缀；`xhigh` 兼容映射为 `max`，`minimal`/`medium` 映射为 `low`，`none` 关闭思考。损坏的工具参数 JSON 按官方规则保留原文并回退渲染。

`glm-5.3`（也接受 `zai-org/GLM-5.3`，忽略大小写）固定使用官方 `ba32f8c19c8738e2c57c71798c601d477a7bf844` 的分词资产和原版聊天模板，包含 2026-10-14 的工具结果 ID 校验及排序优化。工具结果中的原始文本分段、`id` 和批量 `output` 结构会交给官方模板处理。GLM-5.3 模板默认保留历史思考，`clear_thinking=true` 可清除历史思考；`reasoning_effort=low/high` 使用对应前缀，其他值按官方模板回退为 `max`。现有 `glm-5.3-flash` 继续使用原有独立配置。

## 安装与启动

要求 Python 3.11+：

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m maas_tokenizer.main
```

服务启动阶段通过正式计数链路依次预热 GLM 四个、DeepSeek 五个、Kimi 两个模型（每个模型发送一条简单 user 消息）；全部成功后才开始接收流量。两个 MiniMax 模型仍在首次请求时加载。各模型保留自己的聊天模板和 renderer，最终 encode 共用四个 Gigatoken 对象：GLM 四个使用 GLM-5.2 资产（512 MiB 缓存预算），DeepSeek 五个使用 DeepSeek-V4.1-Flash 资产（512 MiB），Kimi K2.6 和 K3 各用自己的资产（各 256 MiB）。跨 DeepSeek 版本共享编码器可能改变原来的 token 计数，这是本方案接受的取舍。缓存预算作用于新建的单个编码器，并非进程总内存上限。

走 Hugging Face chat-template 路径的模型各自加载模板 tokenizer，用它渲染提示词文本（`tokenize=False`），再由 Gigatoken 编码。DeepSeek V3.2、V4 Flash/Pro 和 V4.1 Flash 使用专用文本渲染器，不加载未使用的 Hugging Face tokenizer；DeepSeek V3 仍走 Hugging Face 模板路径。自动判断模板内容格式时，判断结果按实际模板文本在进程内缓存，最多保留 128 份；这与 Gigatoken 的编码缓存相互独立。

运行服务时保持 `--workers 1`。每个进程只有一个计算线程串行执行 tokenizer 请求；多 worker 会各自创建队列和缓存，无法保证进程间串行。

## 请求队列与限流

`/tokenizer` 使用有界 FIFO 队列，1 个请求执行。当前运行配置允许 2 个请求等待；队列已满时返回 `429 queue_full`，入队后等待超过 50 毫秒仍未开始执行时返回 `429 queue_timeout`。两种响应都包含 `Retry-After: 1`，超时任务不会在后台继续计算。等待时间从入队开始计算，不限制已开始的模板渲染和 tokenizer encode。FastAPI 在请求入队前已经读取并解析 JSON 正文；队列配置不限制请求体大小或入队前的内存占用。

以下展示当前使用的队列值；其余配置项为代码默认值。队列值由环境变量传入，未设置时仍默认 100 个等待位和 200 毫秒。

```text
TOKENIZER_QUEUE_SIZE=2
TOKENIZER_QUEUE_TIMEOUT_MS=50
TOKENIZER_MAX_EXECUTION_SECONDS=10
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

### `GET /health`

`GET /health` 不进入计算队列，也不加载模型。调度器记录当前正在执行任务的单调时钟开始时间，空闲时为 `None`，任务完成或报错后清除；等待队列里的请求不计入执行时长。

接口只读取调度器状态并返回：

- 调度器正常且没有超时任务：HTTP 200，`{"status":"ok"}`。
- 调度器未运行或已关闭：HTTP 503，`{"status":"unhealthy","reason":"scheduler_not_running"}`；消费协程意外停止则返回 `scheduler_consumer_stopped`。
- 当前执行任务超过 `TOKENIZER_MAX_EXECUTION_SECONDS`（默认 10 秒）：HTTP 504，`{"status":"unhealthy","reason":"execution_timeout"}`。

执行阈值覆盖已经开始的请求校验、首次模型加载、渲染和编码，不包含排队等待。`/health` 只报告状态，不取消当前计算，也不改变队列超时；空闲或仅有等待请求时不会因执行时长返回 504。

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

默认日志文件达到 100 MB 后轮转并保留 5 份。运行用户需要能够创建和写入配置的日志目录。

## 运行日志

服务生命周期、各模型启动预热、模型首次加载、内部 500 异常栈和调度器消费协程异常写入 `TOKENIZER_RUN_LOG_PATH`。run.log 使用 `Asia/Shanghai` 时间、日志级别和单行事件格式；异常 traceback 中的换行会编码为 `\\n`。已知的请求校验、未知模型和限流等预期错误只记录在 access.log，避免重复日志。

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
- `429`：等待队列已满或排队超过配置的等待时间（当前 50 毫秒）；
- `500`：本地模型资产完整性异常或内部错误。

## 代码链路

```text
src/maas_tokenizer/api.py
  -> service.py
  -> registry.py + models/profiles.json
  -> assets.py + models/manifests/ + model_assets/
  -> renderers.py
       -> HF 模板模型：AutoTokenizer 渲染文本（不在此步 tokenize）
       -> DeepSeek 专用模型：专用文本渲染路径（不加载 HF tokenizer）
       -> Kimi K3：官方 XTML 分段渲染与特殊片段处理
       -> Gigatoken 编码渲染结果（Kimi K3 特殊片段按其专用规则处理）
  -> len(token_ids)
```

- `api.py`：HTTP 入口与错误状态映射；
- `scheduler.py`：有界 FIFO、排队超时与单线程串行执行；
- `access_logging.py`：轮转文件访问日志；
- `run_logging.py`：服务运行日志、异常栈和 Uvicorn 文件日志；
- `service.py`：统一处理流程、模型级懒加载和线程安全缓存；
- `registry.py`：严格解析十三个固定 profile，不做未知模型回退；
- `assets.py`：加载前检查所需本地资产；
- `protocol.py`：OpenAI 风格请求结构；
- `renderers.py`：按 profile 分派 HF、DeepSeek V3.2/V4/V4.1 或 Kimi K3 路径；
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

验证全部十三个 profile 并检查流水线覆盖率门禁：

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
