# 本地 GLM-OCR 部署与对比

## 部署结构

```text
fgclip conda       审核后端、FG-CLIP、策略
glmocr conda       GLM-OCR 0.9B、vLLM、可选文档布局
```

模型权重不进入 Git：

```text
/home/ygx/models/GLM-OCR
/home/ygx/models/PP-DocLayoutV3_safetensors
```

## 启动

审核后端仍使用 `fgclip`。OCR 模型服务由独立终端启动：

```bash
bash /mnt/d/Image-Review-Agent/deploy/glm_ocr/start_vllm.sh
```

服务地址：

```text
http://127.0.0.1:8080/v1/chat/completions
```

项目通过 `config/ocr.yaml` 的 `local_glm_ocr` 配置连接该地址，运行时不会回退云端。

## 显存

- GPU：RTX 4060 Laptop 8 GB
- GLM-OCR 权重：约 2.2 GiB
- vLLM 显存预算：`gpu_memory_utilization=0.46`
- 上下文：6144 tokens
- 与 FG-CLIP 同时运行实测总显存约 6.1 GiB，剩余约 1.8 GiB

文档布局模型使用 CPU，不占 OCR 模型的 GPU 显存。

## 三张同图对比

| 图片 | 本地 GLM-OCR | 云端 OCR | 本地字符召回 | 云端字符召回 |
|---|---:|---:|---:|---:|
| UI 文字图 | 1.49 / 1.53 s | 1.20 / 1.05 s | 待专项标签 | 待专项标签 |
| quality 文字图 | 0.14 / 0.14 s | 3.83 / 2.25 s | 0.796 | 0.185 |
| virtual 文字图 | 0.26 / 0.26 s | 1.14 / 2.85 s | 0.913 / 1.0 | 0.544 / 0.461 |

后两行字符召回来自现有 20 张 OCR 小样本的同类聚合结果。该样本不是独立泛化测试。

## 重要限制

GLM-OCR 的文档布局流水线会对每个文字区域各调用一次模型。复杂 UI 图曾产生 21 个区域，
首次总耗时约 42 秒。因此内容审核默认使用整图单次识别，文档布局仅作为可选文档模式。
