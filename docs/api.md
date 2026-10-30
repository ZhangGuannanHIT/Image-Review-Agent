# 图片审核 API

本文是当前前后端联调契约，以 `backend/api.py` 和 `backend/schemas.py` 的实际实现为准。

## 1. 接口概览

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/health` | 检查固定工作流与视觉模型是否已初始化 |
| `POST` | `/api/review` | 上传单张图片并取得审核建议 |
| `GET` | `/api/records` | 按工作区、域、动作和风险读取持久化队列 |
| `POST` | `/api/records/{id}/feedback` | 确认或多次调整人工结果 |
| `POST` | `/api/records/clear` | 清空指定工作区的持久化队列和审计原图 |
| `GET` | `/api/records/export.csv` | 导出指定工作区 CSV |

默认服务地址为 `http://localhost:8000`。若启动时设置了 `BACKEND_PORT`，端口以环境变量为准。
后端默认允许 `localhost` 与 `127.0.0.1` 的任意端口跨域访问；部署到其他域名时需要扩展
`backend/api.py` 中的 CORS 白名单，或由同源反向代理提供服务。

## 2. 图片审核

```http
POST /api/review
Content-Type: multipart/form-data
```

请求字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `image` | file | 是 | JPG、PNG 或 WEBP，默认不超过 10MB |
| `workspace_id` | string | 否 | 工作区稳定标识，默认 `default`；仅允许字母、数字、下划线和连字符 |

上传字段名必须是 `image`。服务按文件内容解码，不只依赖扩展名或浏览器提交的 MIME；使用 `FormData` 时不要手动设置 multipart boundary。

### 2.1 正常或业务降级响应

HTTP 200 保留原有 `id`、`suggestion`，并新增由程序生成的结构化审核结果：

```json
{
  "id": "7bb6329853124e2bb8ecb746d76d91e4",
  "suggestion": "审核结果：证据不确定，暂不认定违规；处置：人工复核。……",
  "schema_version": "2026-10-27-review-v1",
  "result": {
    "action": "review",
    "action_label": "人工复核",
    "tone": "review",
    "effective_risk": "L1_轻微违规",
    "risk_raw": "L2_中度违规",
    "domain": "virtual",
    "domain_label": "虚拟（二次元 / 插画 / 游戏场景等）",
    "domain_score": 0.98,
    "domain_source": "style_router",
    "confidence": 0.732,
    "reason": "程序裁决理由",
    "primary_reasons": ["binary 明确阴性", "文字语义证据不足"],
    "adjustments": ["动漫风格导致的风险假冲突已解除"],
    "conflict_type": "pseudo_binary_conflict",
    "uncertain": true,
    "policy_version": "2026-10-26-effective-v5",
    "suggestion_source": "llm_explanation",
    "degraded": false,
    "recommended_action": "limit",
    "recommended_action_label": "建议限流",
    "review_hint": true,
    "review_reason": "存在非轻度风险候选，建议人工确认前先采取限流措施"
  },
  "groups": [
    {
      "id": "basic_heads",
      "label": "CLIP 基础头",
      "description": "风险、类型、伤害与质量。",
      "tools": [
        {
          "id": "risk_head",
          "label": "风险等级头",
          "status": "ok",
          "status_label": "已完成",
          "outcome": "risk",
          "outcome_label": "发现风险",
          "summary": "原始候选 L2，置信度 0.732",
          "duration_ms": null,
          "details": {"scores": {"L2_中度违规": 0.732}}
        }
      ]
    }
  ],
  "timing": {
    "total_ms": 8420,
    "stages": [
      {
        "id": "analyze_image",
        "label": "FG-CLIP2 视觉推理",
        "category": "model",
        "duration_ms": 680
      }
    ]
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 本次审核编号，用于查询后端日志 |
| `suggestion` | string | 面向运营的正常建议或降级说明 |
| `schema_version` | string | 结构化响应版本 |
| `result` | object | 动作、有效风险、原始风险、域和核心理由 |
| `groups` | array | 四组工具证据：基础头、专项模型、文字规则、分支裁决 |
| `timing` | object | 后端总耗时和分阶段毫秒耗时 |

`action=review` 表示最终仍需人工确认，但 `recommended_action` 可给出
`limit`、`block` 等可执行建议，前端以“复核建议”单独展示。建议动作不会替代人工复核状态。

每个工具同时提供两个维度：

- `status`：`ok`、`detected`、`empty`、`skipped`、`failed`、`uncertain` 等执行状态；
- `outcome`：`safe`、`risk`、`mixed`、`neutral`、`unknown` 等证据结论。

详细分数、阈值、场景候选和门控原因位于工具 `details`。OCR 原文预览经过长度限制和手机号、
身份证号脱敏；绝对图片路径、内部消息、API 密钥和模型提示词不返回。

程序动作仍是唯一权威结论。默认 `selective` 模式下，放行和限流直接使用程序模板；
人工复核和下架才调用 LLM 填充固定的 `summary/evidence_note/limitation` 字段。
LLM 失败时自动回退程序模板，不修改 `result.action`。`suggestion_source` 取值为
`program_template`、`llm_explanation` 或 `deterministic_fallback`。

当前业务降级也返回同样结构。CLIP 失败时 `action=unavailable`，工具状态会明确标出哪些
步骤未执行。以下只展示关键字段：

```text
{
  "id": "7bb6329853124e2bb8ecb746d76d91e4",
  "suggestion": "审核流程暂不可用，无法完成本次审核，请稍后重试。",
  "schema_version": "2026-10-27-review-v1",
  "result": {"action": "unavailable", "action_label": "流程不可用", ...},
  "groups": [{"id": "basic_heads", "tools": [...]}, ...],
  "timing": {"total_ms": 120, "stages": []}
}
```

### 2.2 HTTP 错误响应

非 200 响应固定为顶层 `{code, message, id}`，没有 `detail` 包装：

```json
{
  "code": 40001,
  "message": "无法解析图片文件",
  "id": "b4a502a942f043ea9bf07a9e14d52aec"
}
```

| HTTP | code | 当前触发条件 |
|---:|---:|---|
| 400 | 40001 | 图片为空、不可解码、格式不支持或超过大小限制 |
| 400 | 40002 | 缺少 `image` 字段或文件名 |
| 500 | 50000 | 未被 Service 内部降级处理的 API 异常 |
| 503 | 50300 | 审核服务尚未初始化 |

前端错误分支读取 `message`，不是 `reason`。

## 3. 健康检查

```http
GET /health
```

HTTP 200 示例：

```json
{
  "status": "ok",
  "service_ready": true,
  "model_loaded": true,
  "mock": false
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | string | `ok` 或 `not_ready` |
| `service_ready` | boolean | 固定审核工作流是否已经构建 |
| `model_loaded` | boolean | 真实视觉模型或 Mock Engine 是否可用 |
| `mock` | boolean | `true` 表示模拟模式，`false` 表示真实视觉引擎 |

`/health` 不会调用 DeepSeek、OCR，也不产生模型 API 费用。当前接口尚不返回 GPU、
LLM/OCR 连通性；`mock=false` 只表示使用真实 FG-CLIP2 引擎。当前真实运行配置通过
`model.require_checkpoint=true` 强制加载训练后的 LoRA、四任务头、binary injury 和 quality fusion，
加载失败时服务不会进入 ready 状态。具体 checkpoint 是否完整仍以启动日志和单次审核日志为准。

## 4. 超时口径

当前配置：

- DeepSeek 单次请求超时：60 秒；
- 阿里云 OCR 单次请求超时：60 秒；
- 固定工作流没有 LLM 轮次上限，按条件顺序执行；
- 后端没有整个 `/api/review` 请求级总超时；
- 后端没有实现专用 HTTP 504 响应。

前端暂建议设置 120 秒超时。若前端主动在 30 秒终止请求，通常得到客户端网络/超时异常，不会收到后端 500；后端任务也不保证随浏览器断开立即取消。若 DeepSeek 超时被 Service 捕获，当前返回 HTTP 200 的降级建议。只有异常继续传播到 API 层时才返回 50000。

## 5. 浏览器调用

```javascript
async function reviewImage(file, timeoutMs = 120_000) {
  const form = new FormData();
  form.append('image', file);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch('/api/review', {
      method: 'POST',
      body: form,
      signal: controller.signal,
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.message || '审核失败');
    }
    return body; // id、suggestion、result、groups、timing
  } finally {
    clearTimeout(timer);
  }
}
```

axios：

```javascript
const form = new FormData();
form.append('image', file);
const { data } = await axios.post('/api/review', form, { timeout: 120_000 });
```

curl：

```bash
curl http://localhost:8000/health
curl -F "image=@path/to/test.jpg" http://localhost:8000/api/review
```

## 6. 当前内部保证

- `analyze_image`（FG-CLIP2）必须成功执行。
- `detect_text` 在 CLIP 后执行；高置信 L0 也不能跳过文字检测。
- `ocr_image` 仅在文字检测命中或检测失败时执行，quality 不再单独触发 OCR。
- OCR 只返回文字、坐标和置信度证据；新增 `content_profile` 仅用于区分普通文字与自由图片描述，不独立裁决动作。
- RAG 默认启用；OCR 成功取得文本后检索本地规则与案例。可通过 `features.rag_enabled=false` 关闭。
- 解释 LLM 不接收原始图片，只接收工具生成的文字证据。
- 例外：高置信动漫或游戏 UI 窄路由可调用结构化视觉复核。真实图片语义路由代码保留，但 `real_context_enabled=false`，默认不调用。
- 同一请求重复调用 CLIP 使用请求级缓存，不重复执行 GPU 推理。
- 真实模式配置为 `lora=true`、`require_checkpoint=true`，必须加载训练后的 LoRA 和分类头；Mock 模式只用于流程演示。

## 7. 已验证状态

截至当前版本：

- 真实 FG-CLIP2 原始权重可在 RTX 5060 Ti/CUDA 环境加载并推理；
- DeepSeek-V4-Flash 在程序锁定动作后生成最终解释；
- 一次真实链路为 `analyze_image → detect_text → OCR/RAG → 场景概念 → 策略 → LLM 解释`；
- 阿里云高精版 OCR 的真实请求已成功；
- RAG 已接入本地规则库，检索分数不是违规概率；最终处置沿用程序规则；
- 自动化测试 215 项通过，覆盖固定流程、grounded 文字语义、动漫路由、视觉复核、policy v2、OCR、RAG 和解释回退；
- 上述后端内部变化没有修改前端请求字段和响应结构。

## 8. 前端联调检查表

- 上传字段固定为 `image`；
- 正常和降级的 200 响应保留 `id`、`suggestion`，并读取 `result`、`groups`、`timing`；
- 工具表必须分别展示执行 `status` 与证据 `outcome`，不能把二者合并成单一状态；
- 缺失结构化字段时前端回退到 `suggestion`，不能误判为成功结构化审核；
- 非 200 响应读取 `code`、`message`、`id`；
- loading 和客户端超时分别处理；
- 请求超时暂设为 120 秒；
- `/health.mock=true` 时明确标识模拟结果；
- `/health.mock=false` 只表示真实视觉引擎已加载，不代表 OCR/LLM 外部服务可用；
- 不依赖任何未写入本契约的内部字段。
