# 人工反馈闭环

## 与 Label Studio 的关系

当前实现是轻量级单图审核工作流，具备：

- SQLite 历史持久化；
- 按工作区隔离的审核队列；
- 自动审查区与已完成区；
- 已完成区下的通过队列和调整队列；
- 域、动作、风险筛选；
- 人工确认建议、多次降级/升级调整；
- 反馈原因；
- CSV 导出；
- 页面工作区与持久化队列分别清空；
- 文件夹按名称排序后串行审核。

不包含 Label Studio 的多项目、多标注员、任务分配、标注一致性等能力。

## 处理状态

- `queue=auto`：程序动作无需人工处理。
- `queue=auto` 或 `manual_status=pending`：仍在自动审查区，尚未完成人工确认。
- `manual_status=confirmed`：人工直接确认，进入已完成·通过。
- `manual_status=adjusted`：人工降级或升级，进入已完成·调整。
- `manual_status=rejected`：旧版兼容状态，界面归入已完成·调整；新流程不再提供驳回按钮。

调整弹窗提供明确的“降级”和“升级”二选一。首次调整以自动动作作为当前等级；
后续调整以当前人工最终动作作为等级，因此支持多次调整。降级只能选择严重度低于
当前等级的目标动作，升级只能选择严重度高于当前等级的目标动作；动作未变化或
方向与目标不一致时，后端拒绝保存。人工调整方向会写入 SQLite 和 CSV。

## 持久化

数据库：

```text
data/reviews.sqlite3
```

审核原图副本：

```text
backend/logs/images/
```

每条审核记录带稳定的 `workspace_id`，不同工作区不会共享自动审查区或已完成区。
工作区编号与名称保存在浏览器 `localStorage`，页面时间线只保留在当前标签页会话；
审核记录、人工处置和原图路径持久化在后端，页面刷新或后端重启后仍可通过各自
工作区队列恢复。旧版无工作区字段的记录迁移到 `default` 工作区。

“清空页面工作区记录”只清理当前页面时间线，不删除 SQLite 队列。
“清空队列”会删除当前工作区的 SQLite 记录及对应审计原图，但不改变页面时间线。

## API

```text
GET  /api/records
GET  /api/records/{id}
GET  /api/records/{id}/image
POST /api/records/{id}/feedback
POST /api/records/clear?workspace_id={id}
GET  /api/records/export.csv
```

反馈示例：

```json
{
  "resolution": "adjust",
  "final_action": "block",
  "adjustment_direction": "upgrade",
  "reason": "人工确认存在明确威胁"
}
```

## 文件夹审核

前端“导入文件夹”按 `webkitRelativePath` 或文件名自然排序，随后逐张串行提交。
同一时刻只运行一个审核请求，避免并发挤占 FG-CLIP、GLM-OCR 和显存。
