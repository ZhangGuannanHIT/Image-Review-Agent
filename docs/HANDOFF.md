# 项目交接摘要

## 当前路线

项目采用 FG-CLIP2 + LLM ReAct Agent。CLIP、OCR、RAG 和低层像素分析都是 Agent Tools：

- `analyze_image` 必须成功调用，调用守卫拒绝无 CLIP 证据的最终提交；
- GLM-OCR、低层特征是可选工具，RAG 当前硬关闭；
- LLM 不直接看图；
- API 只返回 `id` 和 `suggestion`。

详细架构见 `architecture.md`，接口见 `api.md`。

## 标签契约

训练、推理、评估不得偏离当前 Label Studio 四字段：

- risk：5 类单标签；
- violence type：4 路多标签；
- injury：4 类单标签；
- quality：5 类单标签。

`target` 暂不训练。缺失 type 使用 `type_valid` 掩码；其他缺失单标签使用 `IGNORE_INDEX`。

阶段性 JSON 仍在标注中。当前解析保留 1953 条、清理 3 条确定性类型冲突；其他跨字段张力只进入复核，不自动删除。

## 已完成

- 修复 ReviewEngine 的 `lora` 作用域错误；
- 修复 `WeightedCE/focal_gamma` 和 `nn.Module.type` 命名冲突；
- 接通 FG-CLIP2 processor、视觉塔和四任务头推理；
- 统一 head 与视觉塔的 device/dtype；
- 接通多标签缺失掩码；
- 实现 LoRA adapter + head 的最小 checkpoint；
- 实现受控 ReAct 图、CLIP 调用守卫和请求缓存；
- 实现 Mock Engine/LLM/OCR；
- 固定并完成智谱 GLM-OCR 的 JSON+Base64 接口适配；
- 固定并完成 DeepSeek-V4-Flash 独立配置与 LangChain 适配；
- 增加工具级 RAG 禁用开关；
- 建立项目内 Python 3.12 + CUDA 12.8 环境并验证 RTX 5060 Ti；
- 使用 `knife_man.jpg` 完成 FG-CLIP2 原始权重真实前向；
- 使用 DeepSeek-V4-Flash 完成真实 Agent 全流程，轨迹为 `analyze_image → analyze_lowlevel → finalize_review`；
- 修正 GLM-OCR Base64 data URI 并完成真实接口测试；
- 自动化测试 15 项全部通过；
- 冻结前端接口仍为正常/降级 `{id, suggestion}`、HTTP 错误 `{code, message, id}`；
- 更新 `.gitignore`，排除整个模型目录、权重、标注导出、数据、密钥和运行产物；
- 修复 API 图片校验和错误结构；
- 增加最小训练、评估入口与自动化测试；
- 更新 README、架构与 API 文档。

## 外部条件与后续工作

1. 完成全部标注并导出最终 JSON；
2. 提供完整本地图片目录；
3. 复用项目 `.venv` 进行训练和推理；
4. 运行最终数据体检，复核语义张力记录；
5. 生成并冻结 `data/split.json`；
6. 正式训练 LoRA 和四任务头；
7. 在验证集标定 violence type 阈值；
8. 在独立测试集生成正式指标；
9. 为前端联调补充跨域代理或明确部署为同源；
10. 如结题需要，再实现 RAG 判例库；
11. 正式上线前增加请求级总超时和明确的 504 语义。

## 运行

当前配置为真实基模联调：

```bash
.\.venv\Scripts\Activate.ps1
$env:LLM_API_KEY="你的 DeepSeek API Key"
python run_backend.py
```

当前 `require_checkpoint=false`，允许随机任务头验证链路，但结果不能作为模型效果。正式训练完成后必须改回 `lora=true`、`require_checkpoint=true` 并指定 checkpoint。

## API 联调口径

- 正常和业务降级均返回 HTTP 200 `{id, suggestion}`；当前没有 `degraded` 字段；
- 图片/字段错误返回 HTTP 400 `{code, message, id}`；服务未初始化返回 503；
- 当前没有请求级总超时和 504，前端超时暂建议 120 秒；
- `/health.mock=false` 仅代表真实视觉引擎，不代表分类头已经训练；
- RAG 当前未注册给 Agent；
- 完整细节以 `docs/api.md` 为准。

## Git 协作口径

仓库只提交代码、测试、文档和无密钥配置模板。`fg_clip/`、checkpoint、数据、阶段性标注 JSON、测试图片、真实密钥配置、虚拟环境和日志全部忽略。任何真实密钥一旦进入 Git 历史必须立即轮换，不能只靠后续删除文件解决。
