# 图片审核 API

本文是当前前后端联调契约，以 `backend/api.py` 和 `backend/schemas.py` 的实际实现为准。

## 1. 接口概览

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/health` | 检查固定工作流与视觉模型是否已初始化 |
| `POST` | `/api/review` | 上传单张图片并取得审核建议 |

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

上传字段名必须是 `image`。服务按文件内容解码，不只依赖扩展名或浏览器提交的 MIME；使用 `FormData` 时不要手动设置 multipart boundary。

### 2.1 正常或业务降级响应

HTTP 200，固定只有两个字段：

```json
{
  "id": "7bb6329853124e2bb8ecb746d76d91e4",
  "suggestion": "该图片判定为中度违规，建议人工复核。"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 本次审核编号，用于查询后端日志 |
| `suggestion` | string | 面向运营的正常建议或降级说明 |

违规等级、置信度、OCR、低层特征、工具轨迹和 `trained` 状态不返回前端，只写入后端日志。
正常建议以程序生成的“审核结果；处置”开头，后面的解释由 LLM 基于完整
`decision_evidence` 生成。LLM 不能改变处置；解释失败或出现其他动作时，服务自动返回确定性模板。

当前业务降级也返回 HTTP 200 和 `{id, suggestion}`。例如 DeepSeek 或固定流程异常时：

```json
{
  "id": "7bb6329853124e2bb8ecb746d76d91e4",
  "suggestion": "审核流程暂不可用，无法完成本次审核，请稍后重试。"
}
```

目前没有 `degraded` 响应字段，降级说明包含在 `suggestion` 中。前端不得假设所有 HTTP 200 都代表模型成功完成正式裁决。

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
    return body; // { id, suggestion }
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
- OCR 只返回文字、坐标和置信度证据，不承担图片语义裁决。
- RAG 默认启用；OCR 成功取得文本后检索本地规则与案例。可通过 `features.rag_enabled=false` 关闭。
- LLM 不接收原始图片，只接收工具生成的文字证据。
- 例外：高置信动漫路由且初步动作非放行时，可调用结构化视觉复核；真实域不调用。
- 同一请求重复调用 CLIP 使用请求级缓存，不重复执行 GPU 推理。
- 真实模式配置为 `lora=true`、`require_checkpoint=true`，必须加载训练后的 LoRA 和分类头；Mock 模式只用于流程演示。

## 7. 已验证状态

截至当前版本：

- 真实 FG-CLIP2 原始权重可在 RTX 5060 Ti/CUDA 环境加载并推理；
- DeepSeek-V4-Flash 在程序锁定动作后生成最终解释；
- 一次真实链路为 `analyze_image → detect_text → OCR/RAG → 场景概念 → 策略 → LLM 解释`；
- 阿里云高精版 OCR 的真实请求已成功；
- RAG 已接入本地规则库，检索分数不是违规概率；最终处置沿用程序规则；
- 自动化测试 177 项通过，覆盖固定流程、grounded 文字语义、动漫路由、视觉复核、策略、OCR、RAG 和解释回退；
- 上述后端内部变化没有修改前端请求字段和响应结构。

## 8. 前端联调检查表

- 上传字段固定为 `image`；
- 正常和降级的 200 响应读取 `id`、`suggestion`；
- 非 200 响应读取 `code`、`message`、`id`；
- loading 和客户端超时分别处理；
- 请求超时暂设为 120 秒；
- `/health.mock=true` 时明确标识模拟结果；
- `/health.mock=false` 只表示真实视觉引擎已加载，不代表 OCR/LLM 外部服务可用；
- 不依赖任何未写入本契约的内部字段。
