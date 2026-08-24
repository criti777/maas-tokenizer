# GLM-5.2 启动预热设计

## 目标

在 FastAPI 应用启动阶段完成一次 GLM-5.2 的完整 token 计数，使该模型的模板 tokenizer、Gigatoken encoder 以及首次 render/encode 开销在 Pod 接收业务流量前完成。避免第一个真实请求承担冷启动延迟，并避免后续请求在默认 2 秒队列等待期限内大量超时。

## 范围

- 只预热 `glm-5.2`。
- 不改变 `POST /tokenizer`、`GET /health` 或错误响应契约。
- 不修改 Dockerfile，也不在镜像构建阶段执行预热。
- 不预热其他六个模型；它们继续按首次请求懒加载。

## 启动流程

1. FastAPI `lifespan` 创建访问日志器和 `SerialScheduler`。
2. 启动串行调度器及其唯一 tokenizer 工作线程。
3. 通过调度器提交一个最小 GLM-5.2 请求：

   ```json
   {
     "model": "glm-5.2",
     "messages": [{"role": "user", "content": "warmup"}]
   }
   ```

4. 调用现有 `TokenCountService.count()` 完成资产校验、renderer 构建、模板渲染和 Gigatoken encode。
5. 忽略预热返回的 token 数量；renderer 保留在现有进程内模型缓存中。
6. 预热成功后 `lifespan` 才 `yield`，Uvicorn 随后报告应用启动完成并开始服务流量。

## 失败与关闭行为

- 预热出现任何异常时，应用启动失败，Pod 不进入可服务状态；不隐藏模型资产或初始化错误。
- 无论预热、正常运行还是关闭阶段发生什么，调度器和日志 handler 都必须在 `finally` 中释放。
- 预热是内部调用，不生成一条伪造的 `/tokenizer` HTTP 访问日志。
- 预热任务进入空队列后立即开始；现有 2 秒配置只限制排队等待，不限制预热计算耗时。

## 并发与缓存

预热必须通过 `SerialScheduler.submit()` 执行，而不是直接在 FastAPI 事件循环中同步计算。这样首次初始化和后续业务计算都由同一个专用 tokenizer 工作线程承担，并保持每个 Pod 同时最多一个 tokenizer 计算的约束。

`TokenCountService` 继续使用现有 `_renderers` 字典按 profile 缓存 renderer。预热不新增第二套缓存，也不改变锁或模型路由逻辑。

## 测试

- 应用启动时只提交一次 GLM-5.2 最小预热请求，并在启动完成前等待结果。
- 预热异常会传播并阻止应用启动，同时清理已启动的调度器和日志资源。
- 现有 API 测试使用替身服务完成启动，避免每个单元测试真实加载模型资产。
- 运行 API、调度器、缓存相关测试及默认完整测试集，确认接口、排队和缓存行为不变。

## 非目标

- 不提供运行时可配置的预热模型列表。
- 不增加后台预热、定时刷新、自动重试或预热状态接口。
- 不改变 Kubernetes Service、ELB、探针或副本策略。
