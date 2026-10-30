/* 图片审核工作台：页面时间线按会话保留，审核队列按工作区持久化。 */
(function () {
  "use strict";
  function createGroup(key, name) {
    return {
      key: String(key),
      name: String(name || "工作区"),
      records: [],
      selected: null,
      queueRecords: [],
      queueTotal: 0,
      queueSelection: null
    };
  }
  var workspaceRegistry = null;
  try { workspaceRegistry = JSON.parse(localStorage.getItem("imageguard-workspaces-v1") || "null"); } catch (_) {}
  var initialGroups = [];
  if (workspaceRegistry && Array.isArray(workspaceRegistry.groups)) {
    workspaceRegistry.groups.forEach(function (item) {
      if (!item || !/^[A-Za-z0-9_-]{1,64}$/.test(String(item.id || "")) || !String(item.name || "").trim()) return;
      initialGroups.push(createGroup(item.id, String(item.name).trim().slice(0, 32)));
    });
  }
  if (!initialGroups.length) initialGroups.push(createGroup("default", "工作区 1"));
  var selectedGroupKey = workspaceRegistry && String(workspaceRegistry.current || "");
  var firstGroup = initialGroups.find(function (group) { return group.key === selectedGroupKey; }) || initialGroups[0];
  var savedAppearance = "system";
  try { savedAppearance = localStorage.getItem("imageguard-theme") || "system"; } catch (_) {}
  if (!["system", "light", "dark"].includes(savedAppearance)) savedAppearance = "system";
  var savedSidebar = false;
  try { savedSidebar = localStorage.getItem("imageguard-sidebar") === "collapsed"; } catch (_) {}
  var state = { health: "checking", pending: null, groups: initialGroups, group: firstGroup, active: null, nextKey: 0, nextGroup: initialGroups.length + 1, menuGroup: null, renameTarget: null, deleteTarget: null, appearance: savedAppearance, sidebarCollapsed: savedSidebar, currentView: "home", landingScroll: 0, workspaceScroll: 0, queueLoading: false, queueSequence: 0, feedbackTarget: null, feedbackMode: "adjust", batch: null, batchTotal: 0, batchDone: 0 };
  var el = {};
  ["skipLink", "landingPage", "landingMain", "landingNav", "landingBrand", "landingWorkspaceLink", "landingThemeButton", "landingThemeIcon", "landingThemeLabel", "landingThemeMenu", "workspaceApp", "sidebar", "closeSidebar", "openSidebar", "sidebarBackdrop", "newWorkspace", "groupCount", "groupList", "groupMenu", "menuRename", "menuDelete", "deleteDialog", "deleteForm", "deleteQuestion", "confirmDelete", "cancelDelete", "closeDelete", "renameDialog", "renameForm", "workspaceName", "closeRename", "cancelRename", "helpButton", "helpDialog", "closeHelp", "dismissHelp", "helpLearnMore", "clearHistory", "historyCount", "historyList", "historyEmpty", "serviceStatus", "serviceText", "healthRetry", "workspaceTitle", "themeButton", "themeIcon", "themeLabel", "themeMenu", "modeBanner", "modeText", "copyAddress", "timelineViewport", "welcome", "welcomeUpload", "timeline", "dropOverlay", "dropZone", "fileInput", "folderInput", "chooseButton", "folderButton", "pendingFile", "pendingPreview", "pendingName", "pendingMeta", "removeFile", "composerHint", "submitButton", "localError", "announcement", "toast", "queueDomain", "queueAction", "queueRisk", "autoQueue", "confirmedQueue", "adjustedQueue", "autoQueueCount", "confirmedQueueCount", "adjustedQueueCount", "humanQueueCount", "clearQueue", "clearQueueDialog", "clearQueueForm", "clearQueueQuestion", "closeClearQueue", "cancelClearQueue", "exportQueue", "feedbackDialog", "feedbackForm", "feedbackTitle", "feedbackHelp", "feedbackResolution", "feedbackDirection", "feedbackAction", "feedbackReason", "closeFeedback", "cancelFeedback"].forEach(function (id) { el[id] = document.getElementById(id); });
  var API_ADDRESS = (window.REVIEW_API_BASE || location.origin).replace(/\/?$/, "/");
  var healthSequence = 0;
  function ready() { return state.health === "ready" || state.health === "mock"; }
  function saveWorkspaceRegistry() {
    try {
      localStorage.setItem("imageguard-workspaces-v1", JSON.stringify({
        current: state.group ? state.group.key : "default",
        groups: state.groups.map(function (group) {
          return { id: group.key, name: group.name };
        })
      }));
    } catch (_) {}
  }
  function newWorkspaceGroup() {
    var number = state.nextGroup++;
    return createGroup(
      "ws-" + Date.now().toString(36) + "-" + number,
      "工作区 " + number
    );
  }
  function node(tag, className, text) { var n = document.createElement(tag); if (className) n.className = className; if (text !== undefined) n.textContent = text; return n; }
  function button(text, action) { var b = node("button", "text-button", text); b.type = "button"; b.addEventListener("click", action); return b; }
  function timeText(date) { return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }); }
  function duration(seconds) { return String(Math.floor(seconds / 60)).padStart(2, "0") + ":" + String(seconds % 60).padStart(2, "0"); }
  function durationMs(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    return value < 1000 ? Math.round(value) + " ms" : (value / 1000).toFixed(value < 10000 ? 1 : 0) + " s";
  }
  function badge(text, className, dataName, dataValue) {
    var item = node("span", className || "status-badge", text);
    if (dataName && dataValue !== undefined) item.dataset[dataName] = String(dataValue);
    return item;
  }
  function actionTone(result) {
    if (result && ["safe", "warning", "review", "danger", "failed"].includes(result.tone)) return result.tone;
    return "neutral";
  }
  function metric(label, value, className) {
    var item = node("div", "decision-metric" + (className ? " " + className : ""));
    item.append(node("span", "decision-metric-label", label), node("strong", "decision-metric-value", value));
    return item;
  }
  function detailsBlock(className, summary, content) {
    var details = node("details", className);
    details.append(node("summary", "", summary), content);
    return details;
  }
  function preBlock(value, className) {
    return node("pre", className || "detail-json", typeof value === "string" ? value : JSON.stringify(value, null, 2));
  }
  function announce(text) { el.announcement.textContent = text; }
  function toast(text) { el.toast.textContent = text; el.toast.hidden = false; clearTimeout(toast.timer); toast.timer = setTimeout(function () { el.toast.hidden = true; }, 2200); }
  function copyButton(label, value) {
    var control = button(label, async function () {
      await copy(value);
      control.textContent = "已复制 ✓"; control.classList.add("is-copied");
      clearTimeout(control.copyTimer); control.copyTimer = setTimeout(function () { control.textContent = label; control.classList.remove("is-copied"); }, 1500);
    });
    return control;
  }
  var themeSymbols = {
    system: '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4"/></svg>',
    light: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
    dark: '<svg viewBox="0 0 24 24"><path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5 8.5 8.5 0 1 0 20.5 14.5Z"/></svg>'
  };
  var themeNames = { system: "跟随系统", light: "日间", dark: "夜间" };
  var themeControls = [
    { button: el.themeButton, icon: el.themeIcon, label: el.themeLabel, menu: el.themeMenu },
    { button: el.landingThemeButton, icon: el.landingThemeIcon, label: el.landingThemeLabel, menu: el.landingThemeMenu }
  ];
  function renderTheme() {
    document.documentElement.dataset.theme = state.appearance;
    themeControls.forEach(function (control) {
      control.icon.innerHTML = themeSymbols[state.appearance];
      control.label.textContent = themeNames[state.appearance];
      control.button.title = "页面外观：" + themeNames[state.appearance];
      control.button.setAttribute("aria-label", "选择页面外观，当前为" + themeNames[state.appearance]);
      control.menu.querySelectorAll(".theme-option").forEach(function (option) {
        var selected = option.dataset.themeValue === state.appearance;
        option.classList.toggle("is-selected", selected);
        option.setAttribute("aria-checked", String(selected));
      });
    });
  }
  function closeThemeMenus() { themeControls.forEach(function (control) { control.menu.hidden = true; control.button.setAttribute("aria-expanded", "false"); }); }
  function openThemeMenu(control) {
    closeGroupMenu();
    closeThemeMenus(); control.menu.hidden = false;
    control.button.setAttribute("aria-expanded", "true");
    var selected = control.menu.querySelector(".theme-option.is-selected");
    if (selected) selected.focus();
  }
  function requestedView() { return location.hash === "#home" ? "home" : "workspace"; }
  function updateLandingNav() { el.landingNav.classList.toggle("is-scrolled", scrollY > 24); }
  function restorePageScroll(top) {
    var previous = document.documentElement.style.scrollBehavior;
    document.documentElement.style.scrollBehavior = "auto";
    scrollTo(0, top);
    document.documentElement.style.scrollBehavior = previous;
  }
  function showView(view, initial) {
    if (!initial && state.currentView === "home") state.landingScroll = scrollY;
    if (!initial && state.currentView === "workspace") state.workspaceScroll = el.timelineViewport.scrollTop;
    state.currentView = view; closeThemeMenus(); closeGroupMenu(); hideDropOverlay();
    var workspace = view === "workspace";
    el.landingPage.hidden = workspace; el.landingPage.inert = workspace;
    el.workspaceApp.hidden = !workspace; el.workspaceApp.inert = !workspace;
    document.body.classList.toggle("is-workspace-view", workspace);
    el.skipLink.href = workspace ? "#chooseButton" : "#landingMain";
    el.skipLink.textContent = workspace ? "跳到图片提交区" : "跳到主要内容";
    document.title = workspace ? "ImageGuard · 图片审核工作台" : "ImageGuard · 多模态图片内容审核";
    requestAnimationFrame(function () {
      if (workspace) el.timelineViewport.scrollTop = state.workspaceScroll;
      else { restorePageScroll(state.landingScroll); updateLandingNav(); }
      if (!initial) {
        var target = workspace ? el.workspaceTitle : el.landingMain;
        if (target === el.workspaceTitle) target.setAttribute("tabindex", "-1");
        target.focus({ preventScroll: true });
        announce(workspace ? "已进入图片审核工作台。" : "已返回 ImageGuard 项目介绍。工作台记录仍然保留。");
      }
    });
    syncSidebar();
  }
  function route(initial) {
    if (!location.hash || !["#home", "#workspace"].includes(location.hash)) history.replaceState(null, "", "#workspace");
    showView(requestedView(), initial);
  }
  function closeGroupMenu() { el.groupMenu.hidden = true; state.menuGroup = null; }
  function openGroupMenu(group, anchor) {
    state.menuGroup = group; el.groupMenu.hidden = false;
    var box = anchor.getBoundingClientRect();
    el.groupMenu.style.left = Math.max(8, Math.min(box.right - 160, innerWidth - 184)) + "px";
    el.groupMenu.style.top = Math.min(box.bottom + 4, innerHeight - 142) + "px";
    el.menuRename.focus();
  }
  function renameGroup(group) {
    closeGroupMenu();
    state.renameTarget = group;
    el.workspaceName.value = group.name;
    el.workspaceName.setCustomValidity("");
    el.renameDialog.showModal();
    el.workspaceName.focus(); el.workspaceName.select();
  }
  function deleteGroup(group) {
    closeGroupMenu();
    if (ready()) {
      fetch(
        API_ADDRESS + "api/records/clear?workspace_id=" + encodeURIComponent(group.key),
        { method: "POST" }
      ).catch(function () {});
    }
    if (state.active && state.active.record.group === group) cancelActive("已停止等待", "后台可能仍在处理。");
    if (group === state.group) clearPending();
    group.records.forEach(function (record) { URL.revokeObjectURL(record.url); });
    state.groups = state.groups.filter(function (item) { return item !== group; });
    if (!state.groups.length) {
      state.groups.push(newWorkspaceGroup());
    }
    saveWorkspaceRegistry();
    if (group === state.group) switchGroup(state.groups[0], true);
    else renderControls();
    announce("已删除工作区" + group.name + "。");
  }
  function renderGroups() {
    el.groupList.replaceChildren();
    el.groupCount.textContent = state.groups.length;
    state.groups.forEach(function (group) {
      var row = node("div", "group-row"); row.dataset.groupKey = String(group.key);
      var item = node("button", "group-item" + (group === state.group ? " is-selected" : ""));
      item.type = "button"; item.title = "切换到" + group.name + "，已有审核记录会保留";
      item.setAttribute("aria-current", group === state.group ? "page" : "false");
      item.append(node("span", "group-symbol", "▣"), node("span", "group-name", group.name), node("span", "group-count", String(group.records.length)));
      item.addEventListener("click", function () { closeGroupMenu(); switchGroup(group); });
      var more = node("button", "group-more", "⋯"); more.type = "button"; more.title = group.name + "的操作菜单";
      more.setAttribute("aria-label", group.name + "的操作菜单");
      more.addEventListener("click", function () { openGroupMenu(group, row); });
      row.addEventListener("contextmenu", function (event) { event.preventDefault(); openGroupMenu(group, row); });
      row.append(item, more); el.groupList.append(row);
    });
    el.workspaceTitle.textContent = state.group.name;
  }
  function switchGroup(group, force) {
    if (group === state.group && !force) { if (el.sidebar.classList.contains("is-open")) sidebar(false); return; }
    clearPending(); state.group = group; closeGroupMenu();
    saveWorkspaceRegistry();
    el.timeline.replaceChildren(); el.historyList.replaceChildren();
    group.records.forEach(function (record) { el.timeline.append(record.article); el.historyList.append(record.history); });
    renderGroups(); renderControls(); renderQueue();
    if (ready()) loadQueue();
    if (group.records.length) {
      var chosen = group.records.find(function (r) { return r.key === group.selected; }) || group.records[group.records.length - 1];
      selectRecord(chosen, false);
    }
    el.timelineViewport.scrollTop = 0;
    if (el.sidebar.classList.contains("is-open")) sidebar(false);
    announce("已切换到" + group.name + "。");
  }
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); toast("已复制"); }
    catch (_) { var area = node("textarea"); area.value = text; area.className = "sr-only"; document.body.appendChild(area); area.select(); var ok = document.execCommand("copy"); area.remove(); toast(ok ? "已复制" : "复制失败，请手动选择文字复制"); }
  }
  function apiUrl(path) {
    return new URL(String(path || "").replace(/^\//, ""), API_ADDRESS).toString();
  }
  function queueStatusText(record) {
    if (record.manual_status === "confirmed") return "已通过";
    if (record.manual_status === "adjusted") return record.adjustment_direction === "upgrade" ? "已升级" : "已降级";
    if (record.manual_status === "rejected") return "已调整";
    return record.manual_status === "pending" || record.queue === "human" ? "待人工复核" : record.final_action_label;
  }
  function queueItem(record) {
    var item = node("button", "queue-item" + (state.group.queueSelection === record.id ? " is-selected" : ""));
    item.type = "button";
    item.dataset.action = record.final_action || record.action;
    item.setAttribute("aria-label", "打开审核记录 " + record.filename);
    var copyBlock = node("span", "queue-item-copy");
    copyBlock.append(
      node("strong", "", record.filename || record.id),
      node("small", "", (record.current_level_label || record.effective_risk || "未知等级") + " · " + (record.final_action_label || record.final_action))
    );
    item.append(
      node("span", "queue-item-dot"),
      copyBlock,
      node("span", "queue-item-status", queueStatusText(record))
    );
    item.addEventListener("click", function () { openQueueRecord(record); });
    return item;
  }
  function renderQueueSection(container, countNode, records) {
    container.replaceChildren();
    countNode.textContent = String(records.length);
    if (!records.length) {
      container.append(node("p", "queue-empty", "暂无记录"));
      return;
    }
    records.forEach(function (record) { container.append(queueItem(record)); });
  }
  function renderQueue() {
    var records = state.group.queueRecords || [];
    var auto = records.filter(function (record) {
      return record.manual_status !== "confirmed" && record.manual_status !== "adjusted" && record.manual_status !== "rejected";
    });
    var confirmed = records.filter(function (record) {
      return record.manual_status === "confirmed";
    });
    var adjusted = records.filter(function (record) {
      return record.manual_status === "adjusted" || record.manual_status === "rejected";
    });
    renderQueueSection(el.autoQueue, el.autoQueueCount, auto);
    renderQueueSection(el.confirmedQueue, el.confirmedQueueCount, confirmed);
    renderQueueSection(el.adjustedQueue, el.adjustedQueueCount, adjusted);
    el.humanQueueCount.textContent = String(confirmed.length + adjusted.length);
    el.clearQueue.disabled = !state.group.queueTotal;
    el.exportQueue.href = API_ADDRESS + "api/records/export.csv?workspace_id=" + encodeURIComponent(state.group.key);
  }
  async function loadQueue() {
    if (!ready()) return;
    var group = state.group;
    var sequence = ++state.queueSequence;
    var params = new URLSearchParams();
    params.set("workspace_id", group.key);
    if (el.queueDomain.value) params.set("domain", el.queueDomain.value);
    if (el.queueAction.value) params.set("action", el.queueAction.value);
    if (el.queueRisk.value) params.set("risk", el.queueRisk.value);
    state.queueLoading = true;
    try {
      var response = await fetch(API_ADDRESS + "api/records?" + params.toString(), { cache: "no-store" });
      if (!response.ok) throw new Error();
      var body = await response.json();
      if (sequence !== state.queueSequence || group !== state.group) return;
      group.queueRecords = Array.isArray(body.items) ? body.items : [];
      group.queueTotal = Number(body.total) || group.queueRecords.length;
    } catch (_) {
      if (sequence === state.queueSequence && group === state.group) {
        group.queueRecords = [];
        group.queueTotal = 0;
      }
    } finally {
      if (sequence === state.queueSequence) {
        state.queueLoading = false;
        renderQueue();
      }
    }
  }
  function findLocalRecord(recordId) {
    return state.group.records.find(function (record) {
      return record.result && record.result.id === recordId;
    }) || null;
  }
  function manualFromQueue(record) {
    if (!record || record.manual_status === "none") return null;
    return {
      manual_status: record.manual_status,
      final_action: record.final_action,
      final_action_label: record.final_action_label,
      current_level: record.current_level,
      current_level_label: record.current_level_label,
      adjustment_direction: record.adjustment_direction,
      feedback_reason: record.feedback_reason
    };
  }
  function createPersistedRecord(record) {
    var url = record.has_image ? apiUrl(record.image_url) : "";
    var created = record.created_at ? new Date(record.created_at) : new Date();
    var item = {
      key: ++state.nextKey,
      displayIndex: state.group.records.length + 1,
      group: state.group,
      file: { name: record.filename || record.id },
      url: url,
      createdAt: created,
      status: "success",
      persisted: true,
      manual: manualFromQueue(record),
      result: {
        id: record.id,
        suggestion: (record.response && record.response.suggestion) || "",
        structured: record.response
      }
    };
    item.article = node("article", "review-entry");
    item.article.tabIndex = -1;
    item.article.setAttribute("aria-label", "历史审核记录：" + item.file.name);
    var heading = node("div", "entry-heading");
    heading.append(node("strong", "", "历史记录"), node("time", "", timeText(created)));
    item.media = node("div", "review-media");
    item.mediaActions = node("aside", "review-media-actions");
    if (url) {
      var image = node("img", "submitted-image");
      image.src = url;
      image.alt = "历史审核图片：" + item.file.name;
      item.media.append(image);
    } else {
      item.media.classList.add("no-image");
    }
    item.media.append(item.mediaActions);
    item.article.append(heading, item.media);
    item.answer = node("div", "answer");
    item.article.append(node("p", "submitted-name", item.file.name), item.answer);
    item.history = node("button", "history-item");
    item.history.type = "button";
    if (url) {
      var thumb = node("img");
      thumb.src = url;
      thumb.alt = "";
      item.history.append(thumb);
    }
    var description = node("div");
    item.historyStatus = node("small");
    description.append(node("strong", "", item.file.name), item.historyStatus);
    item.history.append(description);
    item.history.addEventListener("click", function () { selectRecord(item, true); });
    state.group.records.push(item);
    el.timeline.append(item.article);
    el.historyList.append(item.history);
    renderAnswer(item);
    return item;
  }
  async function openQueueRecord(record) {
    state.group.queueSelection = record.id;
    renderQueue();
    var local = findLocalRecord(record.id);
    if (!local) {
      try {
        var response = await fetch(API_ADDRESS + "api/records/" + encodeURIComponent(record.id), { cache: "no-store" });
        if (!response.ok) throw new Error();
        record = await response.json();
      } catch (_) {
        toast("无法读取这条历史记录");
        return;
      }
      local = createPersistedRecord(record);
    } else if (manualFromQueue(record)) {
      local.manual = manualFromQueue(record);
      renderAnswer(local);
    }
    selectRecord(local, true);
  }
  function renderManualResult(record) {
    if (!record.manual) return null;
    var panel = node("div", "manual-result");
    panel.dataset.action = record.manual.final_action || "review";
    var labels = { pending: "等待人工处理", confirmed: "已通过", adjusted: "已人工调整", rejected: "已人工调整" };
    var direction = record.manual.adjustment_direction === "upgrade" ? "升级" : record.manual.adjustment_direction === "downgrade" ? "降级" : "";
    var level = node("span", "manual-result-level", record.manual.current_level_label || record.manual.final_action_label || "当前处置");
    panel.append(
      node("strong", "", labels[record.manual.manual_status] || "人工处理"),
      level,
      node("p", "", (direction ? direction + "；" : "") + "最终处置：" + (record.manual.final_action_label || record.manual.final_action) + "。原因：" + (record.manual.feedback_reason || "未填写"))
    );
    return panel;
  }
  var feedbackSeverity = { allow: 0, limit: 1, review: 2, block: 3 };
  var feedbackActionLabels = {
    allow: "L0 · 放行",
    limit: "L1 · 限流",
    review: "L2 · 人工复核",
    block: "L3 · 下架"
  };
  function currentFeedbackAction(record, result) {
    if (record.manual && feedbackSeverity[record.manual.final_action] !== undefined) {
      return record.manual.final_action;
    }
    return result.action;
  }
  function confirmedFeedbackAction(result) {
    return result.action === "review" ? result.recommended_action : result.action;
  }
  function feedbackDirectionValue() {
    var checked = el.feedbackDirection.querySelector('input[name="feedbackDirection"]:checked');
    return checked ? checked.value : "downgrade";
  }
  function setFeedbackDirection(value) {
    var input = el.feedbackDirection.querySelector('input[name="feedbackDirection"][value="' + value + '"]');
    if (input) input.checked = true;
  }
  function actionsForDirection(action, direction) {
    var baseline = feedbackSeverity[action];
    if (baseline === undefined) return [];
    return Object.keys(feedbackSeverity)
      .filter(function (candidate) {
        return direction === "upgrade"
          ? feedbackSeverity[candidate] > baseline
          : feedbackSeverity[candidate] < baseline;
      })
      .sort(function (a, b) { return feedbackSeverity[a] - feedbackSeverity[b]; });
  }
  function openFeedback(record, mode) {
    var result = record.result && record.result.structured && record.result.structured.result;
    if (!result) return;
    state.feedbackTarget = record;
    state.feedbackMode = mode;
    el.feedbackResolution.value = mode;
    var currentAction = currentFeedbackAction(record, result);
    var recommended = result.recommended_action && feedbackSeverity[result.recommended_action] !== undefined
      ? result.recommended_action
      : currentAction;
    var baseline = feedbackSeverity[currentAction];
    var proposed = feedbackSeverity[recommended];
    var previousDirection = record.manual && record.manual.adjustment_direction;
    var direction = proposed < baseline
      ? "downgrade"
      : proposed > baseline
        ? "upgrade"
        : previousDirection
          || (baseline > 0 ? "downgrade" : "upgrade");
    setFeedbackDirection(direction);
    el.feedbackReason.value = "";
    var titles = { confirm: "确认系统建议", adjust: "调整审核结果" };
    el.feedbackTitle.textContent = titles[mode];
    el.feedbackHelp.textContent = mode === "confirm"
      ? "确认后记录进入已完成·通过，并保留可再次调整的能力。"
      : "当前等级：" + (record.manual && record.manual.current_level_label ? record.manual.current_level_label : feedbackActionLabels[currentAction]) + "；调整后会进入已完成·调整。";
    populateFeedbackActions(mode === "confirm" ? confirmedFeedbackAction(result) : recommended, actionsForDirection(currentAction, direction));
    updateFeedbackControls();
    el.feedbackDialog.showModal();
  }
  function populateFeedbackActions(preferred, actions) {
    var allowed = actions || actionsForDirection(
      currentFeedbackAction(state.feedbackTarget, state.feedbackTarget.result.structured.result),
      feedbackDirectionValue()
    );
    el.feedbackAction.replaceChildren();
    allowed.forEach(function (action) {
      var option = node("option", "", feedbackActionLabels[action]);
      option.value = action;
      el.feedbackAction.append(option);
    });
    el.feedbackAction.value = allowed.includes(preferred) ? preferred : allowed[0];
  }
  function updateFeedbackControls() {
    var mode = el.feedbackResolution.value;
    var result = state.feedbackTarget && state.feedbackTarget.result && state.feedbackTarget.result.structured && state.feedbackTarget.result.structured.result;
    var currentAction = result ? currentFeedbackAction(state.feedbackTarget, result) : undefined;
    var recommended = result && result.recommended_action ? result.recommended_action : currentAction;
    var confirmOption = el.feedbackResolution.querySelector('option[value="confirm"]');
    var baseline = feedbackSeverity[currentAction];
    var downgrade = el.feedbackDirection.querySelector('input[value="downgrade"]');
    var upgrade = el.feedbackDirection.querySelector('input[value="upgrade"]');
    confirmOption.disabled = !recommended || recommended === "review";
    el.feedbackDirection.hidden = mode !== "adjust";
    el.feedbackDirection.disabled = mode !== "adjust";
    downgrade.disabled = baseline === undefined || baseline <= 0;
    upgrade.disabled = baseline === undefined || baseline >= 3;
    el.feedbackAction.disabled = mode !== "adjust";
    if (mode === "adjust") {
      var direction = feedbackDirectionValue();
      if (actionsForDirection(currentAction, direction).length === 0) {
        direction = !upgrade.disabled ? "upgrade" : "downgrade";
        setFeedbackDirection(direction);
      }
      populateFeedbackActions(el.feedbackAction.value || recommended, actionsForDirection(currentAction, direction));
    } else if (mode === "confirm") {
      var confirmAction = confirmedFeedbackAction(result);
      populateFeedbackActions(confirmAction, [confirmAction]);
      el.feedbackAction.value = confirmAction;
    }
    el.feedbackReason.required = mode === "adjust";
    el.feedbackReason.placeholder = mode === "confirm" ? "可选：补充确认说明" : "请填写调整原因";
  }
  async function submitFeedback(event) {
    event.preventDefault();
    var record = state.feedbackTarget;
    if (!record || !record.result) return;
    var resolution = el.feedbackResolution.value;
    var reason = el.feedbackReason.value.trim();
    if (resolution !== "confirm" && !reason) {
      el.feedbackReason.reportValidity();
      return;
    }
    try {
      var response = await fetch(API_ADDRESS + "api/records/" + encodeURIComponent(record.result.id) + "/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          resolution: resolution,
          final_action: el.feedbackAction.value,
          adjustment_direction: resolution === "adjust" ? feedbackDirectionValue() : "",
          reason: reason
        })
      });
      var body = await response.json();
      if (!response.ok) throw new Error(body.message || "保存失败");
      record.manual = manualFromQueue(body);
      state.group.queueSelection = body.id;
      el.feedbackDialog.close();
      renderAnswer(record);
      await loadQueue();
      toast("人工处理结果已保存");
    } catch (error) {
      toast(error.message || "人工处理结果保存失败");
    }
  }
  function enqueueFolder(files) {
    if (state.active || state.batch) return;
    var images = Array.from(files || []).filter(function (file) {
      return ["image/jpeg", "image/png", "image/webp"].includes(file.type) || /\.(jpe?g|png|webp)$/i.test(file.name);
    }).sort(function (a, b) {
      return String(a.webkitRelativePath || a.name).localeCompare(String(b.webkitRelativePath || b.name), "zh-CN", { numeric: true });
    });
    if (!images.length) { toast("文件夹中没有支持的图片"); return; }
    state.batch = images;
    state.batchTotal = images.length;
    state.batchDone = 0;
    startNextBatch();
  }
  function startNextBatch() {
    if (state.active || !state.batch) return;
    if (!state.batch.length) {
      var total = state.batchTotal;
      state.batch = null;
      renderControls();
      loadQueue();
      toast("文件夹审核完成：" + total + " 张");
      return;
    }
    var file = state.batch.shift();
    clearPending();
    state.pending = { file: file, url: URL.createObjectURL(file) };
    renderControls();
    submit();
  }
  function renderControls() {
    var busy = Boolean(state.active || state.batch);
    el.chooseButton.disabled = busy; el.welcomeUpload.disabled = busy; el.fileInput.disabled = busy; el.folderButton.disabled = busy; el.folderInput.disabled = busy; el.removeFile.disabled = busy;
    el.submitButton.disabled = busy || !state.pending || !ready();
    el.submitButton.textContent = state.active ? "正在审核…" : "开始审核 ↑";
    el.composerHint.textContent = state.batch ? "文件夹串行审核 " + state.batchDone + "/" + state.batchTotal : (busy ? "完成后可继续提交下一张" : "或拖拽图片到这里");
    el.pendingFile.hidden = !state.pending;
    if (state.pending) { el.pendingPreview.src = state.pending.url; el.pendingName.textContent = state.pending.file.name; el.pendingMeta.textContent = (state.pending.file.size / 1024).toFixed(1) + " KB · 待提交"; }
    else el.pendingPreview.removeAttribute("src");
    var records = state.group.records;
    el.clearHistory.disabled = !records.length && !state.pending;
    el.welcome.hidden = records.length > 0;
    el.historyEmpty.hidden = records.length > 0;
    el.historyCount.textContent = records.length;
    el.landingWorkspaceLink.textContent = state.groups.some(function (group) { return group.records.length; }) ? "返回工作台" : "进入工作台";
    state.groups.forEach(function (group) { group.records.forEach(function (r) { if (r.retryButton) r.retryButton.disabled = busy || !ready(); }); });
    renderGroups();
  }
  function renderHealth() {
    var texts = { checking: "正在连接服务", ready: "审核服务已连接", mock: "模拟服务已连接", "offline-preview": "本地界面预览", "not-ready": "服务尚未就绪", offline: "服务未连接" };
    el.serviceText.textContent = texts[state.health]; el.serviceStatus.dataset.state = state.health;
    el.healthRetry.hidden = !(state.health === "offline" || state.health === "not-ready");
    var preview = state.health === "offline-preview", mock = state.health === "mock";
    el.modeBanner.hidden = !preview && !mock; el.copyAddress.hidden = !preview;
    el.modeText.textContent = preview ? "本地预览：可选图查看界面。启动后端后访问 " + API_ADDRESS + " 进行审核。" : "模拟演示模式：结果仅用于流程联调，不代表真实模型效果。";
    renderControls();
  }
  async function checkHealth() {
    var sequence = ++healthSequence;
    if (location.protocol === "file:") { state.health = "offline-preview"; renderHealth(); return; }
    state.health = "checking"; renderHealth();
    var controller = new AbortController(), timer = setTimeout(function () { controller.abort(); }, 5000);
    try {
      var response = await fetch(API_ADDRESS + "health", { cache: "no-store", signal: controller.signal }); if (!response.ok) throw new Error();
      var body = await response.json(); if (sequence !== healthSequence) return;
      state.health = body && body.status === "ok" && body.service_ready === true ? (body.mock === true ? "mock" : body.model_loaded === true ? "ready" : "not-ready") : "not-ready";
    } catch (_) { if (sequence === healthSequence) state.health = "offline"; }
    finally {
      clearTimeout(timer);
      if (sequence === healthSequence) {
        renderHealth();
        if (ready()) loadQueue();
      }
    }
  }
  function clearPending() { if (state.pending) URL.revokeObjectURL(state.pending.url); state.pending = null; el.fileInput.value = ""; el.localError.hidden = true; renderControls(); }
  function acceptFiles(files) {
    if (state.active || !files || !files.length) return;
    var f = files[0], error = "";
    if (files.length !== 1) error = "一次只能提交一张图片。";
    else if (!f.size) error = "图片为空，请重新选择。";
    else if (f.size > 10 * 1024 * 1024) error = "图片超过 10MB，请压缩后重试。";
    else if (!["image/jpeg", "image/png", "image/webp"].includes(f.type) && !(!f.type && /\.(jpe?g|png|webp)$/i.test(f.name))) error = "请选择 JPG、PNG 或 WEBP 图片。";
    el.fileInput.value = "";
    if (error) { el.localError.textContent = error; el.localError.hidden = false; return; }
    clearPending(); state.pending = { file: f, url: URL.createObjectURL(f) }; renderControls(); el.submitButton.focus();
  }
  function mobileLayout() { return matchMedia("(max-width:760px)").matches; }
  function sidebar(open) {
    if (!mobileLayout()) return;
    el.sidebar.classList.toggle("is-open", open); el.sidebarBackdrop.hidden = !open;
    el.openSidebar.setAttribute("aria-expanded", String(open)); el.sidebar.inert = !open;
    el.openSidebar.setAttribute("aria-label", open ? "收起工作区导航" : "展开工作区导航");
    if (open) el.closeSidebar.focus(); else el.openSidebar.focus();
  }
  function toggleDesktopSidebar() {
    state.sidebarCollapsed = !state.sidebarCollapsed;
    try { localStorage.setItem("imageguard-sidebar", state.sidebarCollapsed ? "collapsed" : "expanded"); } catch (_) {}
    syncSidebar(); el.openSidebar.focus();
    announce(state.sidebarCollapsed ? "工作区导航已收起。" : "工作区导航已展开。");
  }
  function selectRecord(record, scroll) {
    if (record.group !== state.group) switchGroup(record.group);
    state.group.selected = record.key;
    state.group.records.forEach(function (r) { r.history.classList.toggle("is-selected", r.key === state.group.selected); r.history.setAttribute("aria-current", r.key === state.group.selected ? "true" : "false"); });
    if (scroll) {
      if (el.sidebar.classList.contains("is-open")) sidebar(false);
      record.article.scrollIntoView({ block: "start", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
      record.article.focus({ preventScroll: true });
    }
  }
  function renderTiming(timing) {
    if (!timing || !Array.isArray(timing.stages)) return null;
    var root = node("section", "timing-panel");
    var heading = node("div", "section-heading");
    var copy = node("div");
    copy.append(node("h3", "", "处理耗时"), node("p", "", "后端实测阶段耗时，不包含浏览器上传和网络等待。"));
    heading.append(copy, badge(durationMs(timing.total_ms), "total-duration", "tone", timing.total_ms > 15000 ? "warning" : "neutral"));
    root.append(heading);
    var list = node("div", "timing-list");
    var stages = timing.stages.filter(function (stage) { return stage.id !== "service.total"; });
    var max = Math.max.apply(null, stages.map(function (stage) { return Number(stage.duration_ms) || 0; }).concat([1]));
    stages.forEach(function (stage) {
      var row = node("div", "timing-row");
      var track = node("span", "timing-track");
      var bar = node("span", "timing-bar");
      bar.style.width = Math.max(2, Math.round((Number(stage.duration_ms) || 0) / max * 100)) + "%";
      track.append(bar);
      row.append(node("span", "timing-label", stage.label || stage.id), track, node("code", "timing-value", durationMs(Number(stage.duration_ms))));
      list.append(row);
    });
    root.append(detailsBlock("timing-details", "查看全部 " + stages.length + " 个阶段", list));
    return root;
  }
  function toolRow(tool) {
    var row = node("tr", "tool-row");
    row.dataset.toolId = tool.id;
    var name = node("th");
    name.scope = "row";
    name.append(node("strong", "", tool.label || tool.id));
    var outcome = node("td", "tool-outcome");
    outcome.append(
      badge(tool.status_label || tool.status, "status-badge", "status", tool.status),
      badge(tool.outcome_label || tool.outcome, "outcome-badge", "outcome", tool.outcome)
    );
    var summary = node("td", "tool-summary", tool.summary || "无结果说明");
    var durationCell = node("td", "tool-duration", durationMs(tool.duration_ms));
    var control = node("td", "tool-detail-cell");
    var detailsRow = node("tr", "tool-details-row");
    detailsRow.hidden = true;
    var detailCell = node("td");
    detailCell.colSpan = 5;
    var hasDetails = tool.details && Object.keys(tool.details).length;
    if (hasDetails) {
      detailCell.append(preBlock(tool.details));
      var toggle = button("查看", function () {
        detailsRow.hidden = !detailsRow.hidden;
        toggle.textContent = detailsRow.hidden ? "查看" : "收起";
        toggle.setAttribute("aria-expanded", String(!detailsRow.hidden));
      });
      toggle.setAttribute("aria-expanded", "false");
      control.append(toggle);
    } else {
      control.append(node("span", "muted-text", "无展开项"));
    }
    detailsRow.append(detailCell);
    row.append(name, outcome, summary, durationCell, control);
    return [row, detailsRow];
  }
  function renderEvidenceGroup(group) {
    var section = node("section", "evidence-group");
    section.dataset.group = group.id;
    var heading = node("div", "section-heading");
    var copy = node("div");
    copy.append(node("h3", "", group.label || group.id), node("p", "", group.description || ""));
    heading.append(copy);
    section.append(heading);
    var table = node("table", "tool-table");
    var thead = node("thead");
    var head = node("tr");
    ["工具", "执行 / 结论", "关键结果", "耗时", ""].forEach(function (label) { head.append(node("th", "", label)); });
    thead.append(head);
    var tbody = node("tbody");
    (group.tools || []).forEach(function (tool) {
      toolRow(tool).forEach(function (row) { tbody.append(row); });
    });
    table.append(thead, tbody);
    section.append(table);
    return section;
  }
  function renderStructuredResult(record) {
    var payload = record.result.structured;
    var result = payload.result || {};
    var panel = node("section", "decision-panel");
    panel.dataset.tone = actionTone(result);
    var header = node("div", "decision-heading");
    var title = node("div", "decision-title");
    title.append(node("span", "decision-eyebrow", "PROGRAM DECISION"), node("h3", "", result.action_label || "待确认"));
    var states = node("div", "decision-states");
    states.append(badge(result.uncertain ? "证据不确定" : "程序动作已锁定", "decision-state", "tone", result.uncertain ? "review" : "safe"));
    if (result.degraded) states.append(badge("流程降级", "decision-state", "tone", "warning"));
    header.append(title, states);
    panel.append(header);
    var metrics = node("div", "decision-metrics");
    metrics.append(
      metric("有效风险", result.effective_risk || "无有效预测", "is-risk"),
      metric("原始风险头", result.risk_raw || "无原始候选"),
      metric("图片域", result.domain_label || result.domain || "未分类"),
      metric("总处理", durationMs(payload.timing && payload.timing.total_ms)),
      metric("风险头置信度", typeof result.confidence === "number" ? (result.confidence * 100).toFixed(1) + "%" : "未提供")
    );
    panel.append(metrics);
    if (result.action === "review" && result.review_hint) {
      var review = node("div", "review-recommendation");
      var reviewText = node("div");
      reviewText.append(node("span", "review-recommendation-label", "复核建议"), node("strong", "", result.recommended_action_label || "建议人工判断"));
      review.append(reviewText, node("p", "", result.review_reason || "建议在人工确认前保留临时限制。"));
      panel.append(review);
    }
    var reasons = node("ul", "primary-reasons");
    (result.primary_reasons || []).slice(0, 3).forEach(function (reason) { reasons.append(node("li", "", reason)); });
    if (reasons.children.length) panel.append(reasons);
    var reasonDetails = node("div");
    reasonDetails.append(node("p", "", result.reason || "无程序理由"));
    if (Array.isArray(result.adjustments) && result.adjustments.length) {
      var adjustments = node("ul", "adjustment-list");
      result.adjustments.forEach(function (item) { adjustments.append(node("li", "", item)); });
      reasonDetails.append(node("h4", "", "有效风险调整"), adjustments);
    }
    panel.append(detailsBlock("decision-reason-details", "查看完整裁决依据", reasonDetails));
    return panel;
  }
  function createRecord(pending) {
    var record = { key: ++state.nextKey, displayIndex: state.group.records.length + 1, group: state.group, file: pending.file, url: pending.url, createdAt: new Date(), status: "loading", mock: state.health === "mock" };
    record.article = node("article", "review-entry"); record.article.tabIndex = -1;
    record.article.setAttribute("aria-label", "图片审核 " + record.displayIndex + "：" + record.file.name);
    var heading = node("div", "entry-heading"); heading.append(node("strong", "", "提交图片 " + String(record.displayIndex).padStart(2, "0")));
    var time = node("time", "", timeText(record.createdAt)); time.dateTime = record.createdAt.toISOString(); heading.append(time);
    var img = node("img", "submitted-image"); img.src = record.url; img.alt = "待审核图片：" + record.file.name;
    record.media = node("div", "review-media");
    record.mediaActions = node("aside", "review-media-actions");
    record.media.append(img, record.mediaActions);
    record.answer = node("div", "answer"); record.article.append(heading, record.media, node("p", "submitted-name", record.file.name), record.answer);
    record.history = node("button", "history-item"); record.history.type = "button";
    var thumb = node("img"); thumb.src = record.url; thumb.alt = "";
    var description = node("div"); record.historyStatus = node("small");
    description.append(node("strong", "", record.file.name), record.historyStatus); record.history.append(thumb, description);
    record.history.addEventListener("click", function () { selectRecord(record, true); });
    state.group.records.push(record); el.timeline.append(record.article); el.historyList.append(record.history); selectRecord(record, false); return record;
  }
  function renderAnswer(record) {
    record.answer.replaceChildren();
    if (record.mediaActions) record.mediaActions.replaceChildren();
    record.retryButton = null;
    record.answer.className = "answer" + (record.status === "error" ? " answer-error" : "");
    record.history.dataset.status = record.status;
    record.article.setAttribute("aria-busy", String(record.status === "loading"));
    var manualStatus = record.manual ? record.manual.manual_status : "none";
    record.article.classList.toggle("is-manually-adjusted", manualStatus === "adjusted" || manualStatus === "rejected");
    record.article.classList.toggle("is-human-confirmed", manualStatus === "confirmed");
    var historyText = { loading: "审核中", success: "已完成", error: "未完成" }[record.status];
    if (record.status === "success" && record.result && record.result.structured && record.result.structured.result) {
      var currentResult = record.result.structured.result;
      historyText = record.manual
        ? (manualStatus === "confirmed" ? "已通过" : "已调整") + " · " + (record.manual.final_action_label || record.manual.final_action)
        : currentResult.action_label || historyText;
      record.history.dataset.tone = record.manual
        ? ({ allow: "safe", limit: "warning", review: "review", block: "danger" }[record.manual.final_action] || "neutral")
        : currentResult.tone || "neutral";
    } else {
      delete record.history.dataset.tone;
    }
    record.historyStatus.textContent = timeText(record.createdAt) + " · " + historyText;
    var title = node("h3", "", record.status === "error" ? record.error.title : "审核结果");
    if (record.mock) title.append(node("span", "answer-label", "模拟模式")); record.answer.append(title);
    if (record.status === "loading") {
      var line = node("p", "loading-text"); record.elapsedNode = node("span", "elapsed", "00:00");
      line.append(node("span", "spinner"), node("span", "", "正在分析图片"), record.elapsedNode);
      record.answer.append(line, node("p", "loading-note", "通常需要 15–30 秒，复杂图片可能更久。"), button("取消等待", function () { cancelActive("已取消等待", "页面已停止等待，后台可能仍在处理。重试会发起一次新的审核。"); }));
    } else if (record.status === "success") {
      var structured = record.result.structured;
      if (structured && structured.result) {
        var decisionPanel = renderStructuredResult(record);
        if (manualStatus === "adjusted" || manualStatus === "rejected") {
          decisionPanel.prepend(node("p", "superseded-result-note", "原自动审核结果，已被人工调整取代。"));
        }
        record.answer.append(decisionPanel);
        var manualResult = renderManualResult(record);
        if (manualResult) record.answer.append(manualResult);
        var timingPanel = renderTiming(structured.timing);
        if (timingPanel) record.answer.append(timingPanel);
        var groupStack = node("div", "evidence-stack");
        (structured.groups || []).forEach(function (group) { groupStack.append(renderEvidenceGroup(group)); });
        record.answer.append(groupStack);
        var explanationBody = node("div");
        explanationBody.append(node("p", "suggestion", record.result.suggestion));
        record.answer.append(detailsBlock("suggestion-panel", "运营解释", explanationBody));
        record.answer.append(detailsBlock("raw-payload", "完整结构化响应（JSON）", preBlock(structured)));
      } else {
        var legacyNotice = node("div", "legacy-response-warning");
        legacyNotice.append(
          node("strong", "", "当前连接的是旧版后端响应"),
          node("p", "", "服务只返回了建议文本，没有结构化风险、工具状态和耗时。请重启后端后再审核；页面脚本已通过版本号绕过旧缓存。")
        );
        record.answer.append(legacyNotice);
        record.answer.append(node("p", "suggestion", record.result.suggestion));
      }
      var meta = node("div", "result-meta");
      meta.append(node("span", "", "审核编号"), node("code", "", record.result.id));
      if (structured && structured.result) {
        meta.append(node("span", "", "策略"), node("code", "", structured.result.policy_version || "未提供"));
        meta.append(node("span", "", "响应版本"), node("code", "", structured.schema_version || "未提供"));
      }
      record.answer.append(meta);
      if (structured && structured.result) {
        var result = structured.result;
        var confirmTarget = confirmedFeedbackAction(result);
        var canConfirm = (
          (manualStatus === "none" || manualStatus === "pending")
          && confirmTarget
          && feedbackSeverity[confirmTarget] !== undefined
          && confirmTarget !== "review"
        );
        var confirmButton = button(
          manualStatus === "confirmed" ? "已确认" : manualStatus === "adjusted" || manualStatus === "rejected" ? "已调整" : "确认",
          function () { openFeedback(record, "confirm"); }
        );
        confirmButton.className += " manual-action manual-action-confirm";
        confirmButton.disabled = !canConfirm;
        confirmButton.title = canConfirm ? "确认当前系统建议并归入已完成·通过" : "当前记录无需重复确认，可继续调整";
        var adjustButton = button(
          manualStatus === "adjusted" || manualStatus === "rejected" ? "再次调整" : "调整",
          function () { openFeedback(record, "adjust"); }
        );
        adjustButton.className += " manual-action manual-action-adjust";
        record.mediaActions.append(confirmButton, adjustButton);
      }
      var actions = node("div", "answer-actions"); actions.append(copyButton("复制建议", record.result.suggestion), copyButton("复制编号", record.result.id)); record.answer.append(actions);
    } else {
      record.answer.append(node("p", "", record.error.message));
      if (record.error.code || record.error.id) {
        var errorMeta = node("div", "result-meta");
        if (record.error.code) errorMeta.append(node("span", "", "错误代码 " + record.error.code));
        if (record.error.id) errorMeta.append(node("code", "", "请求编号 " + record.error.id)); record.answer.append(errorMeta);
      }
      record.retryButton = button("重新审核这张图片", function () { submit(record); }); record.answer.append(record.retryButton);
    }
  }
  function finish(active, error, result) {
    if (state.active !== active) return;
    clearInterval(active.interval); clearTimeout(active.timeout); state.active = null;
    active.record.status = error ? "error" : "success"; active.record.error = error; active.record.result = result;
    if (state.batch) {
      state.batchDone++;
    }
    renderAnswer(active.record); renderControls();
    if (state.batch) {
      setTimeout(startNextBatch, 0);
    } else {
      loadQueue();
      announce(error ? error.title : "审核完成，可继续提交下一张图片。");
    }
  }
  function cancelActive(title, message) { var active = state.active; if (!active) return; active.controller.abort(); finish(active, { title: title, message: message }); }
  async function submit(existing) {
    if (state.active || !ready() || (!existing && !state.pending)) return;
    var record = existing || createRecord(state.pending);
    if (record.group !== state.group) switchGroup(record.group);
    if (!existing) { state.pending = null; el.fileInput.value = ""; }
    el.localError.hidden = true; record.status = "loading"; record.mock = state.health === "mock";
    var active = { record: record, controller: new AbortController(), started: Date.now() }; state.active = active;
    renderAnswer(record); renderControls(); selectRecord(record, true); announce("图片已提交，审核服务正在处理中。");
    active.interval = setInterval(function () { if (state.active === active) record.elapsedNode.textContent = duration(Math.floor((Date.now() - active.started) / 1000)); }, 1000);
    active.timeout = setTimeout(function () { if (state.active === active) cancelActive("等待时间过长", "已等待 120 秒并停止等待，后台可能仍在处理。重试会发起一次新的审核。"); }, 120000);
    var form = new FormData();
    form.append("image", record.file);
    form.append("workspace_id", record.group.key);
    try {
      var response = await fetch(API_ADDRESS + "api/review", { method: "POST", body: form, signal: active.controller.signal });
      if (state.active !== active) return;
      var body;
      try { body = await response.json(); } catch (_) { finish(active, { title: "响应格式异常", message: "服务返回了无法解析的内容，请稍后重试。" }); return; }
      if (state.active !== active) return;
      if (!response.ok) finish(active, { title: "审核未完成", message: typeof body?.message === "string" ? body.message : "服务返回错误，请稍后重试。", code: body?.code, id: body?.id });
      else if (!body || typeof body.id !== "string" || typeof body.suggestion !== "string") finish(active, { title: "响应字段异常", message: "审核编号或建议字段不完整，请检查服务。" });
      else finish(active, null, { id: body.id, suggestion: body.suggestion, structured: body });
    } catch (_) { finish(active, { title: "无法连接审核服务", message: "请检查后端和网络连接，再重试这张图片。" }); }
  }
  function openClearQueueDialog() {
    if (!state.group.queueTotal) return;
    el.clearQueueQuestion.textContent =
      "将删除“" + state.group.name + "”中的 " + state.group.queueTotal +
      " 条持久化审核记录及对应审计原图。当前页面中的工作区时间线不会随之清空。";
    el.clearQueueDialog.showModal();
  }
  async function clearPersistentQueue(event) {
    event.preventDefault();
    var group = state.group;
    el.clearQueueDialog.close();
    try {
      var response = await fetch(
        API_ADDRESS + "api/records/clear?workspace_id=" + encodeURIComponent(group.key),
        { method: "POST" }
      );
      var body = await response.json();
      if (!response.ok) throw new Error(body.message || "清空队列失败");
      group.queueRecords = [];
      group.queueTotal = 0;
      group.queueSelection = null;
      renderQueue();
      toast("已清空当前工作区的持久化队列");
      announce("已清空" + group.name + "的持久化审核队列。");
    } catch (error) {
      toast(error.message || "清空队列失败");
    }
  }
  function clearWorkspace() {
    state.batch = null;
    state.batchDone = 0;
    state.batchTotal = 0;
    if (state.active && state.active.record.group === state.group) cancelActive("已停止等待", "后台可能仍在处理。");
    clearPending();
    state.group.records.forEach(function (r) { URL.revokeObjectURL(r.url); }); state.group.records = []; state.group.selected = null;
    el.timeline.replaceChildren(); el.historyList.replaceChildren(); renderControls(); el.timelineViewport.scrollTop = 0;
    announce("当前工作区的记录已清空，其他工作区保留。");
  }
  function fileDrag(event) { return event.dataTransfer && Array.from(event.dataTransfer.types || []).includes("Files"); }
  var dragDepth = 0;
  function hideDropOverlay() { dragDepth = 0; el.dropOverlay.hidden = true; el.dropZone.classList.remove("is-dragover"); }
  el.fileInput.addEventListener("change", function () { acceptFiles(el.fileInput.files); });
  el.folderInput.addEventListener("change", function () {
    enqueueFolder(el.folderInput.files);
    el.folderInput.value = "";
  });
  [el.chooseButton, el.welcomeUpload].forEach(function (b) { b.addEventListener("click", function () { if (!state.active) el.fileInput.click(); }); });
  el.folderButton.addEventListener("click", function () { if (!state.active && !state.batch) el.folderInput.click(); });
  el.removeFile.addEventListener("click", clearPending); el.submitButton.addEventListener("click", function () { submit(); });
  window.addEventListener("dragenter", function (event) {
    if (el.workspaceApp.hidden || !fileDrag(event)) return; event.preventDefault(); dragDepth++;
    if (!state.active) { el.dropOverlay.hidden = false; el.dropZone.classList.add("is-dragover"); }
  });
  window.addEventListener("dragover", function (event) { if (!el.workspaceApp.hidden && fileDrag(event)) { event.preventDefault(); event.dataTransfer.dropEffect = state.active ? "none" : "copy"; } });
  window.addEventListener("dragleave", function (event) { if (el.workspaceApp.hidden || !fileDrag(event)) return; dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) hideDropOverlay(); });
  window.addEventListener("drop", function (event) { if (el.workspaceApp.hidden || !fileDrag(event)) return; event.preventDefault(); var files = event.dataTransfer.files; hideDropOverlay(); acceptFiles(files); });
  window.addEventListener("paste", function (event) {
    if (el.workspaceApp.hidden || state.active || !event.clipboardData) return;
    var images = Array.from(event.clipboardData.files || []).filter(function (file) { return file.type.startsWith("image/"); });
    if (images.length) { event.preventDefault(); acceptFiles(images); announce("已从剪贴板读取图片，请确认后开始审核。"); }
  });
  el.healthRetry.addEventListener("click", checkHealth); el.copyAddress.addEventListener("click", function () { copy(API_ADDRESS); });
  [el.queueDomain, el.queueAction, el.queueRisk].forEach(function (control) {
    control.addEventListener("change", loadQueue);
  });
  el.exportQueue.href = API_ADDRESS + "api/records/export.csv";
  el.clearQueue.addEventListener("click", openClearQueueDialog);
  el.clearQueueForm.addEventListener("submit", clearPersistentQueue);
  [el.cancelClearQueue, el.closeClearQueue].forEach(function (control) {
    control.addEventListener("click", function () { el.clearQueueDialog.close(); });
  });
  el.clearQueueDialog.addEventListener("click", function (event) {
    if (event.target === el.clearQueueDialog) el.clearQueueDialog.close();
  });
  el.feedbackResolution.addEventListener("change", updateFeedbackControls);
  el.feedbackDirection.addEventListener("change", updateFeedbackControls);
  el.feedbackForm.addEventListener("submit", submitFeedback);
  [el.cancelFeedback, el.closeFeedback].forEach(function (control) {
    control.addEventListener("click", function () { el.feedbackDialog.close(); });
  });
  el.feedbackDialog.addEventListener("click", function (event) {
    if (event.target === el.feedbackDialog) el.feedbackDialog.close();
  });
  el.clearHistory.addEventListener("click", clearWorkspace);
  el.newWorkspace.addEventListener("click", function () {
    var group = newWorkspaceGroup();
    state.groups.push(group); switchGroup(group); el.chooseButton.focus();
  });
  el.menuRename.addEventListener("click", function () { if (state.menuGroup) renameGroup(state.menuGroup); });
  el.renameForm.addEventListener("submit", function (event) {
    event.preventDefault();
    var name = el.workspaceName.value.trim();
    if (!name) { el.workspaceName.setCustomValidity("请输入工作区名称"); el.workspaceName.reportValidity(); return; }
    if (!state.renameTarget) return;
    state.renameTarget.name = name;
    el.renameDialog.close();
    saveWorkspaceRegistry();
    renderGroups();
    announce("工作区已命名为" + name + "。");
  });
  el.workspaceName.addEventListener("input", function () { el.workspaceName.setCustomValidity(""); });
  [el.cancelRename, el.closeRename].forEach(function (item) { item.addEventListener("click", function () { el.renameDialog.close(); }); });
  el.renameDialog.addEventListener("close", function () { state.renameTarget = null; });
  el.menuDelete.addEventListener("click", function () {
    if (!state.menuGroup) return;
    state.deleteTarget = state.menuGroup;
    el.deleteQuestion.textContent = "“" + state.deleteTarget.name + "”的页面记录和持久化审核队列都会被删除，此操作不可撤销。";
    closeGroupMenu(); el.deleteDialog.showModal(); el.confirmDelete.focus();
  });
  el.deleteForm.addEventListener("submit", function (event) { event.preventDefault(); var target = state.deleteTarget; el.deleteDialog.close(); if (target) deleteGroup(target); });
  [el.cancelDelete, el.closeDelete].forEach(function (item) { item.addEventListener("click", function () { el.deleteDialog.close(); }); });
  el.deleteDialog.addEventListener("close", function () { state.deleteTarget = null; });
  el.helpButton.addEventListener("click", function () { closeThemeMenus(); el.helpDialog.showModal(); el.closeHelp.focus(); });
  [el.dismissHelp, el.closeHelp].forEach(function (item) { item.addEventListener("click", function () { el.helpDialog.close(); }); });
  el.helpLearnMore.addEventListener("click", function () { state.landingScroll = 0; el.helpDialog.close(); });
  el.helpDialog.addEventListener("click", function (event) { if (event.target === el.helpDialog) el.helpDialog.close(); });
  document.addEventListener("click", function (event) {
    if (!el.groupMenu.hidden && !el.groupMenu.contains(event.target) && !event.target.closest(".group-row")) closeGroupMenu();
    themeControls.forEach(function (control) { if (!control.menu.hidden && !control.menu.contains(event.target) && !control.button.contains(event.target)) closeThemeMenus(); });
  });
  themeControls.forEach(function (control) {
    control.button.addEventListener("click", function () { if (control.menu.hidden) openThemeMenu(control); else closeThemeMenus(); });
    control.menu.querySelectorAll(".theme-option").forEach(function (option) {
      option.addEventListener("click", function () {
        state.appearance = option.dataset.themeValue;
        try { localStorage.setItem("imageguard-theme", state.appearance); } catch (_) {}
        renderTheme(); closeThemeMenus(); control.button.focus();
        announce("页面已切换为" + themeNames[state.appearance] + "模式。");
      });
    });
    control.menu.addEventListener("keydown", function (event) {
      var items = Array.from(control.menu.querySelectorAll(".theme-option"));
      var index = items.indexOf(document.activeElement), next = index;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % items.length;
      else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (index - 1 + items.length) % items.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = items.length - 1;
      else return;
      event.preventDefault(); items[next].focus();
    });
  });
  el.groupMenu.addEventListener("keydown", function (event) {
    var items = [el.menuRename, el.menuDelete], index = items.indexOf(document.activeElement), next = index;
    if (event.key === "ArrowDown") next = (index + 1) % items.length;
    else if (event.key === "ArrowUp") next = (index - 1 + items.length) % items.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = items.length - 1;
    else return;
    event.preventDefault(); items[next].focus();
  });
  renderTheme();
  el.openSidebar.addEventListener("click", function () { if (mobileLayout()) sidebar(!el.sidebar.classList.contains("is-open")); else toggleDesktopSidebar(); });
  [el.closeSidebar, el.sidebarBackdrop].forEach(function (b) { b.addEventListener("click", function () { sidebar(false); }); });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && (el.renameDialog.open || el.deleteDialog.open || el.helpDialog.open)) return;
    var openTheme = themeControls.find(function (control) { return !control.menu.hidden; });
    if (e.key === "Escape" && openTheme) { closeThemeMenus(); openTheme.button.focus(); return; }
    if (e.key === "Escape" && !el.groupMenu.hidden) { closeGroupMenu(); return; }
    if (e.key === "Escape" && el.sidebar.classList.contains("is-open")) sidebar(false);
  });
  function syncSidebar() {
    var mobile = mobileLayout();
    if (mobile) {
      el.workspaceApp.classList.remove("is-collapsed");
      el.sidebar.inert = !el.sidebar.classList.contains("is-open");
      el.openSidebar.setAttribute("aria-expanded", String(el.sidebar.classList.contains("is-open")));
      var mobileLabel = el.sidebar.classList.contains("is-open") ? "收起工作区导航" : "展开工作区导航";
      el.openSidebar.setAttribute("aria-label", mobileLabel); el.openSidebar.title = mobileLabel;
    } else {
      el.sidebar.classList.remove("is-open"); el.sidebarBackdrop.hidden = true;
      el.workspaceApp.classList.toggle("is-collapsed", state.sidebarCollapsed);
      el.sidebar.inert = state.sidebarCollapsed;
      el.openSidebar.setAttribute("aria-expanded", String(!state.sidebarCollapsed));
      var desktopLabel = state.sidebarCollapsed ? "展开工作区导航" : "收起工作区导航";
      el.openSidebar.setAttribute("aria-label", desktopLabel); el.openSidebar.title = desktopLabel;
    }
  }
  window.addEventListener("resize", syncSidebar);
  window.addEventListener("hashchange", function () { route(false); });
  window.addEventListener("scroll", updateLandingNav, { passive: true });
  el.landingBrand.addEventListener("click", function (event) {
    if (state.currentView !== "home") return;
    event.preventDefault();
    scrollTo({ top: 0, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  });
  var landingLinks = Array.from(document.querySelectorAll(".landing-links a"));
  document.querySelectorAll('a[href="#principles"],a[href="#workflow"],a[href="#boundaries"],a[href="#usage"]').forEach(function (link) {
    link.addEventListener("click", function (event) {
      event.preventDefault(); var target = document.querySelector(link.getAttribute("href"));
      if (target) target.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
    });
  });
  var sectionObserver = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) { if (entry.isIntersecting) landingLinks.forEach(function (link) { link.classList.toggle("is-active", link.hash === "#" + entry.target.id); }); });
  }, { rootMargin: "-30% 0px -60%", threshold: 0 });
  ["principles", "workflow", "boundaries", "usage"].forEach(function (id) { sectionObserver.observe(document.getElementById(id)); });
  route(true);
  window.addEventListener("beforeunload", function () {
    if (state.active) { state.active.controller.abort(); clearInterval(state.active.interval); clearTimeout(state.active.timeout); }
    if (state.pending) URL.revokeObjectURL(state.pending.url);
    state.groups.forEach(function (group) { group.records.forEach(function (r) { URL.revokeObjectURL(r.url); }); });
  });
  renderControls(); checkHealth();
})();
