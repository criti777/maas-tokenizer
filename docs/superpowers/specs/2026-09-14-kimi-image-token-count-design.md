# Kimi 图片尺寸计数设计

## 范围与兼容性

仅实现 kimi-k2.6、kimi-k3 的图片视觉 token 计数，不处理视频或音频，不下载 URL、不解码 base64、不加载视觉模型。
用户已确认：未提供 multimodal_metadata 时完整保留旧计数行为；null 也视为未提供。
现有文本请求、其他模型请求和返回结构保持不变。

## 输入协议

顶层 multimodal_metadata 是数组。每项 media_type 必须为 image，shape 必须为两个正整数 [宽, 高]，不接受布尔值、浮点数或字符串作为尺寸。
size、hash 及额外字段不参与计数；同一 hash 重复出现仍逐次计数。
数组顺序与 messages 顺序及每条 content 数组中 image/image_url 出现顺序一致；数量必须相等。新模式仅接受 user 消息中的图片，避免工具消息重排或忽略内容造成错配。
显式空数组仅在没有图片时有效；缺失尺寸、无效类型或数量不匹配返回请求错误，不回退为部分计数。
提供非空 metadata 时不支持其他模型，也不允许混入视频/音频：返回明确错误，不宣称已计算完整多模态输入。

## 计算模块

新增独立纯尺寸函数，不引入 Torch/Pillow 等视觉依赖。
按官方 navit_resize_image 的缩放、整数截断和补齐次序实现，不将 patch 预算当作最终 token 硬上限。
patch_size=14、merge_kernel_size=2、patch_limit_on_one_side=512、fixed_output_tokens=null。
K2.6 的 in_patch_limit=16384；K3 为 65536。
实现时固定官方参考版本及来源，比较官方尺寸函数与本地函数的输出。

## 接入位置

service.count 在加载 renderer 前校验元数据及图片对应关系。
保留 renderer 原有文本流程；有元数据时，K3 使用官方带原始宽高的 image_prompts，K2.6 保留原有媒体包裹标记。官方 processor 使用 img.size，已核对。
总数为对应渲染 token 数减去实际被替换的图片占位 token 数，再加各图片视觉 token 数。
只扣除结构化图片对应的占位，不搜索并扣除用户普通文本中的相似标记。
K3 有 metadata 时若用户另外设置 image_prompts，拒绝冲突输入，避免两个来源决定图片提示。
metadata 模式拒绝自定义 chat_template；K3 同时拒绝文字内的 kimi_image_placeholder，因为官方会将它作为额外图片槽消费。这些限制均不影响无 metadata 的旧行为。
不修改调用方原始请求，不将图片配置放入所有模型共用的模板参数。

## 测试与验收

- 尺寸单测：1 像素、28 倍数及前后边界、普通图片、大图、极端长宽比、两种预算。
- 官方对照：固定尺寸案例及随机正整数尺寸，比较缩放后宽高、补齐及视觉 token 数。
- 集成：两模型单图、多图、文本混合、重复 hash、URL 空/缺失不影响尺寸计数。
- 校验：错误 shape、数量不匹配、非 image 元数据、不支持的模型、image_prompts 冲突。
- 兼容：字段缺省/null 完全等同旧计数；原有全部模型回归测试通过。
- K3 检查尺寸提示的 token 开销；验证占位替换无漏算和重复计数。
- 全程不访问生产 8080，不部署、不推送或合入 main。

## 官方参考

- https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/media_utils.py
- https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/preprocessor_config.json
- https://huggingface.co/moonshotai/Kimi-K3/blob/main/media_utils.py
- https://huggingface.co/moonshotai/Kimi-K3/blob/main/preprocessor_config.json
- https://huggingface.co/moonshotai/Kimi-K3/blob/main/kimi_k3_processor.py

## 方案取舍

选择纯尺寸模块配合原 renderer，避免引入像素处理和模型运行依赖。
不采用直接加载完整官方 Processor 的方案，因为计数无需处理像素且会增加依赖和资源开销。
不采用所有请求强制提供尺寸的方案，因为用户要求保留旧行为。
