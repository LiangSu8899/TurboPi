验证完成总结
Full Coverage TRT 验证结果 (debug-08)
方案	KV Cache 延迟	Hz
当前最佳 (torch_trt_fp8)	47.38 ms	21.1 Hz
Full Coverage TRT (QKV+O+MLP 全TRT)	45.75 ms	21.9 Hz
提升	1.63 ms	1.04x
✅ 精度: Key/Value cosine > 0.99, PASSED
⚠️ 加速: 仅 4%，QKV/O projection 本身计算量小，不是瓶颈
当前状态分析
已达成优化上限
TRT 编译优化已接近极限：

Vision: 17ms (TRT FP16) ✅
KV Cache: 47ms (TRT FP8 MLP) ✅
Denoise: 10ms/step (CUDA Graph) ✅
Total: 83.5ms (12.0 Hz)
20Hz 目标差距
要达到 50ms (20Hz)，需要削减 33.5ms：

KV Cache 需要从 47ms → 14ms (3.4x 加速)
单靠 TRT 优化已无法实现
后续优化方向建议
1️⃣ KV Cache Reuse (推荐优先尝试)
turbo_titan_pipeline.py 中有实现，策略：

帧间图像/状态相似时复用 KV Cache
预期收益: 1.5-2x (如果相似度高)

原理: 连续帧的 prefix embedding 变化小
→ KV Cache 可部分/完全复用
→ 跳过部分层计算
2️⃣ Action Chunking
每次推理输出多步 action：

例如: 每推理1次，执行3步 action
等效频率: 12Hz × 3 = 36Hz
需要验证对任务成功率影响
3️⃣ 异步流水线
Vision | KV Cache | Denoise 三阶段并行：

理论可隐藏 Vision 延迟 (17ms)
有效延迟: max(47, 17, 10) = 47ms (21Hz)
4️⃣ 减少层数
18层 → 12层 (删除中间层)：

预期收益: ~1.5x
风险: 精度损失，需要验证

接下来探索方向
KV Cache Reuse - 验证 turbo_titan_pipeline.py
Action Chunking - 测试不同 chunk size 对精度影响
异步流水线 - 实现三阶段并行
减少层数 - 验证精度影响