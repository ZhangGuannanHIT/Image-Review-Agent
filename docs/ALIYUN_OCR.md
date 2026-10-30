# 阿里云高精版 OCR 接入说明

## 商品接口

- 商品：高精版 OCR 文字识别
- Endpoint：`https://gjbsb.market.alicloudapi.com/ocrservice/advanced`
- Method：`POST`
- 图片字段：`img`，纯 Base64，不加 Data URI 前缀
- 鉴权：`Authorization: APPCODE <AppCode>`

`AppKey/AppSecret` 不参与当前请求。不要把 `AppCode` 或 `AppSecret` 写到前端。

## 本地配置

`config/ocr.yaml`：

```yaml
aliyun_market_ocr:
  provider: "aliyun_market_ocr"
  model: "advanced"
  endpoint: "https://gjbsb.market.alicloudapi.com/ocrservice/advanced"
  appcode: "YOUR_APPCODE"
  timeout: 60
  max_image_mb: 25
  prob: true
  char_info: false
  rotate: true
  table: false
  sort_page: false
  no_stamp: false
  figure: false
  row: true
  paragraph: false
  oricoord: true
```

也可用环境变量 `ALIYUN_OCR_APPCODE` 覆盖文件中的 AppCode。

`config/ocr.yaml` 已被 Git 忽略，不能提交或写入日志。

## 已验证响应

项目客户端已成功解析：

- `texts`：按行文字
- `boxes`：归一化后的文字框
- `line_confidences`：逐行置信度
- `mean_confidence`：平均置信度
- `request_id`：接口请求 ID

## 后续策略

OCR 只提供文字和可读性证据，不直接决定违规。

- 能读出稳定文字：交给 RAG 检查规则、语境和反例
- 文字明确威胁或违法：可提高处置等级
- 文字置信度低、严重扭曲或残缺：进入人工复核
- 无文字：不能据此判断图片安全

## 兼容性

`backend/ocr.py` 仍保留旧的 `glm_ocr` Provider，但当前本地配置默认使用
`aliyun_market_ocr`。
