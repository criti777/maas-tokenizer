# Pod 内请求队列与访问日志设计

## 目标

MaaS Tokenizer 部署为每 Pod 一个 Uvicorn worker、一个 CPU 核心。`POST /tokenizer` 在单 Pod 内有界排队并严格串行执行 token 计算，避免突发并发争抢 CPU。每个请求输出一条可由 `X-Span-Id` 关联的访问日志，同时写入标准输出和轮转文件。

ELB 和跨 Pod 负载均衡不在本次范围内。

## 请求执行模型

Pod 内维护一个容量为 100 的 FIFO 等待队列和一个后台消费者。消费者把任务提交给唯一的单线程执行器，调用现有 `TokenCountService.count()`。因此任何时刻最多只有一个请求执行模型渲染和 encode；FastAPI 事件循环仍可接收新连接、执行健康检查和快速拒绝过载请求。

队列容量只计算等待中的请求，不包含正在执行的一个请求：

- 队列未满：请求入队并等待；
- 队列已满：不入队，立即返回 HTTP 429；
- 入队后 2 秒仍未开始执行：取消该任务并返回 HTTP 429；
- 已经开始执行：继续执行到完成，不应用排队超时。

超时或客户端取消的任务必须标记为取消。消费者遇到尚未开始且已取消的任务时直接跳过，不再执行 token 计算。

两个 429 场景沿用当前错误结构，并设置 `Retry-After: 1`：

```json
{
  "detail": {
    "stage": "admission_control",
    "type": "queue_full",
    "message": "tokenizer queue is full"
  }
}
```

等待超时的 `type` 为 `queue_timeout`。现有请求校验、未知模型、多模态边界和内部错误状态映射保持不变。成功响应仍是直接 JSON 整数。

## 生命周期和配置

队列、后台消费者和单线程执行器由 FastAPI lifespan 创建并关闭。关闭时停止接收新任务，取消尚未执行的等待任务，并等待正在执行的任务结束，避免留下后台线程。

默认配置通过环境变量覆盖：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `TOKENIZER_QUEUE_SIZE` | `100` | 最大等待任务数 |
| `TOKENIZER_QUEUE_TIMEOUT_SECONDS` | `2` | 入队后最长等待时间 |
| `TOKENIZER_LOG_PATH` | `/opt/cloud/logs/access.log` | 访问日志文件 |
| `TOKENIZER_LOG_MAX_BYTES` | `104857600` | 单日志文件 100 MB |
| `TOKENIZER_LOG_BACKUP_COUNT` | `5` | 保留轮转文件数量 |

配置必须在启动时校验为有效正数。服务必须以 `--workers 1` 启动；多 worker 会各自创建队列和执行线程，无法保证 Pod 级严格串行。

## 访问日志

`/tokenizer` 的每次请求最终输出一条单行访问日志。使用同一个 logger 配置两个 handler：标准输出和 `RotatingFileHandler`。服务创建日志目录；目录或文件不可写时启动失败，避免运维误以为文件日志已经生效。

日志不记录请求正文、messages、tools 或其他用户内容。核心字段为：

- UTC ISO-8601 时间；
- `span_id`：请求头 `X-Span-Id`，缺失时生成 UUID；
- `model`：合法请求体中的模型，无法解析时为 `-`；
- `status`：`success`、`failed` 或 `rejected`；
- `reason`：拒绝或失败类型，成功时为 `-`；
- `http_status`；
- `queue_wait_ms`；
- `process_ms`；
- `total_ms`。

日志字段值需安全转义，防止请求头中的换行等内容伪造日志行。文件与 stdout 的内容一致。

## 模块边界

- `api.py`：HTTP 契约、span ID、错误映射，并把请求提交给调度器；
- 新的调度模块：有界 FIFO、排队超时、取消、单线程执行和生命周期；
- 新的日志模块：环境配置、双 handler、轮转和单行格式；
- `service.py`、renderer、Gigatoken encoder 和模型资产不改变业务语义。

`/health` 不经过计算队列，也不写 tokenizer 访问日志，确保负载高时 Kubernetes 仍能检查进程存活。

## 测试

测试必须覆盖：

1. 并发请求的 `service.count()` 最大同时执行数为 1，且按 FIFO 开始；
2. 100 个等待名额和 1 个执行名额的边界；
3. 队列满立即返回 429，并带 `Retry-After`；
4. 排队超过 2 秒返回 429，超时任务之后不会执行；
5. 成功、业务失败、队列满、排队超时各生成一条日志；
6. `X-Span-Id`、model、状态和三类耗时字段正确，恶意换行被转义；
7. stdout 与轮转文件均收到日志，文件达到阈值后轮转；
8. `/health` 在计算任务运行时仍可响应；
9. 现有 API、七模型渲染和 token 数回归不变。
