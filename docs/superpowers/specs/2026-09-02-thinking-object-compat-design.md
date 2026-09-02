# Thinking 对象兼容设计

## 目标

`POST /tokenizer` 在不改变现有模型缺省计数结果的前提下，接受上游网关支持的对象式 Thinking、顶层旧字段和当前已有字段。兼容层只归一化会影响聊天模板与输入 token 的参数，不复制推理生成参数。

兼容原则：对方接受的合法输入不得被本服务拒绝；对方拒绝的复杂模型限制可以降级处理；只有明确表达相反开关或无法可靠解释的字段类型才拒绝。

## 接受的输入

服务继续接受：

- 顶层 `thinking: bool`；
- `chat_template_kwargs.thinking: bool`；
- `chat_template_kwargs.enable_thinking: bool`；
- 顶层 `reasoning_effort`，允许 `none/minimal/low/medium/high/xhigh/max`。

新增接受：

- 顶层 `thinking: null`；
- 顶层 `thinking: {}`，按对方协议解释为开启；
- 顶层 `thinking: {"type":"enabled"}`；
- 顶层 `thinking: {"type":"disabled"}`；
- 顶层 `thinking: {"type":"auto"}`；
- `thinking.clear_thinking: bool`；
- 顶层旧字段 `enable_thinking: bool | null`；
- 顶层 `preserve_thinking: bool`，传给实际支持它的模板。

继续拒绝无法可靠解释的输入，例如字符串、数字、列表形式的 `thinking`，非字符串或未知的 `thinking.type`，以及非布尔的 `enable_thinking`、`clear_thinking`、`preserve_thinking`。

## 统一语义

兼容层把所有开关来源归一化为 `bool | None`：

| 输入 | 归一化结果 |
|---|---|
| `thinking: true/false` | 对应布尔值 |
| `thinking.type: enabled` | `True` |
| `thinking.type: disabled` | `False` |
| `thinking.type: auto` | `None` |
| `thinking: {}` | `True` |
| `thinking: null` 或字段缺失 | `None` |
| 任一合法 `enable_thinking` | 对应布尔值 |
| `reasoning_effort: none` | `False` |
| 其他合法 `reasoning_effort` | `True`，模型专属规则可进一步覆盖 |

`None` 表示请求没有明确覆盖模型模板的默认行为。与参考项目不同，本服务不会把字段缺失统一改成 `False`，避免已有模型的默认 prompt 与 token 数发生变化。

`auto` 对所有模型均可接受并归一化为 `None`；不复制参考项目针对 mixed-thinking 模型的拒绝规则。

## 冲突规则

兼容层从下列位置收集明确的 `True/False`：

- 顶层布尔 `thinking`；
- `thinking.type`；
- 顶层 `enable_thinking`；
- `chat_template_kwargs.thinking`；
- `chat_template_kwargs.enable_thinking`；
- `reasoning_effort` 派生值。

同时出现 `True` 和 `False` 时返回 `400 request_processing_error`，错误信息保持为 `conflicting thinking options`。`None` 不参与冲突；例如 `thinking.type=auto` 与 `enable_thinking=true` 最终为 `True`。

输入对象不原地修改。规范化只复制顶层对象以及实际需要修改的 `thinking`、`chat_template_kwargs` 等小型容器，继续共享大型 `messages` 内容。

## 传递给渲染层

规范化结果通过请求模型暴露为：

- `thinking` / `enable_thinking`: 明确开关为布尔值；未指定时不注入；
- `clear_thinking`: 默认 `False`；
- `preserve_thinking`: 请求值或缺省值；
- `reasoning_effort`: 保留原合法字符串。

通用 HF Renderer 同时传递 `thinking` 与 `enable_thinking`，兼容模板的两种命名；模板未读取的字段自然无效。DeepSeek V3.2/V4 将开关转换成 `thinking_mode="thinking"/"chat"`；Kimi K3 转换为官方分段编码器的布尔参数。内部结果为 `None` 时，各 Renderer 保持修改前的缺省行为。

`clear_thinking` 主要供 GLM 模板读取；其他模板可以忽略。DeepSeek V4 继续接收 `reasoning_effort`。Kimi K3 仍校验官方编码器实际支持的 effort 档位，因为不支持的值无法可靠生成官方 token 序列。

## 模型专属规则

为匹配参考网关，GLM-5.2 的顶层 `reasoning_effort=none/minimal` 强制得到 `False`；这个覆盖在模型已解析后执行，避免改变其他模型对 `minimal` 的现有解释。

不复制以下生成期改写，因为它们不会影响输入 token：

- `frequency_penalty`；
- `presence_penalty`；
- `temperature`；
- `top_p`。

LongCat 等未注册模型不增加专属分支。未来新增模型时，只在其模板确实需要时扩展 profile 或 Renderer 行为。

## 错误与兼容边界

合法的新格式与旧格式均进入现有计数链路，成功响应仍为 `{"prompt_tokens": int}`。明确冲突和格式错误继续返回现有两字段错误对象，不新增错误响应字段。

对象式 `thinking` 不会原样传入模板；渲染层只看到归一化后的布尔开关和独立辅助参数。因此模板无需同时理解枚举对象和布尔值。

## 测试

新增表驱动测试覆盖：

1. 对象式 `enabled/disabled/auto/{}/null`；
2. 顶层 `enable_thinking`；
3. 对象式 `clear_thinking` 与顶层 `preserve_thinking`；
4. 所有开关来源的一致组合与明确冲突；
5. 错误类型、未知 `type` 的稳定 400；
6. GLM-5.2 `reasoning_effort=none/minimal`；
7. HF、DeepSeek V3.2/V4、Kimi K3 的参数传递；
8. 未传 Thinking 时所有旧模型的固定 token 结果不变；
9. 全模型测试与覆盖率门禁继续通过。

