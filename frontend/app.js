/* 独立图片审核：页面内存记录，只调用 /health 和 /api/review。 */
(function () {
  "use strict";
  var firstGroup = { key: 1, name: "工作区 1", records: [], selected: null };
  var savedAppearance = "system";
  try { savedAppearance = localStorage.getItem("imageguard-theme") || "system"; } catch (_) {}
  if (!["system", "light", "dark"].includes(savedAppearance)) savedAppearance = "system";
  var savedSidebar = false;
  try { savedSidebar = localStorage.getItem("imageguard-sidebar") === "collapsed"; } catch (_) {}
  var state = { health: "checking", pending: null, groups: [firstGroup], group: firstGroup, active: null, nextKey: 0, nextGroup: 1, menuGroup: null, renameTarget: null, deleteTarget: null, appearance: savedAppearance, sidebarCollapsed: savedSidebar, currentView: "home", landingScroll: 0, workspaceScroll: 0 };
  var el = {};
  ["skipLink", "landingPage", "landingMain", "landingNav", "landingBrand", "landingWorkspaceLink", "landingThemeButton", "landingThemeIcon", "landingThemeLabel", "landingThemeMenu", "workspaceApp", "sidebar", "closeSidebar", "openSidebar", "sidebarBackdrop", "newWorkspace", "groupCount", "groupList", "groupMenu", "menuRename", "menuDelete", "deleteDialog", "deleteForm", "deleteQuestion", "confirmDelete", "cancelDelete", "closeDelete", "renameDialog", "renameForm", "workspaceName", "closeRename", "cancelRename", "helpButton", "helpDialog", "closeHelp", "dismissHelp", "helpLearnMore", "clearHistory", "historyCount", "historyList", "historyEmpty", "serviceStatus", "serviceText", "healthRetry", "workspaceTitle", "themeButton", "themeIcon", "themeLabel", "themeMenu", "modeBanner", "modeText", "copyAddress", "timelineViewport", "welcome", "welcomeUpload", "timeline", "dropOverlay", "dropZone", "fileInput", "chooseButton", "pendingFile", "pendingPreview", "pendingName", "pendingMeta", "removeFile", "composerHint", "submitButton", "localError", "announcement", "toast"].forEach(function (id) { el[id] = document.getElementById(id); });
  var API_ADDRESS = (window.REVIEW_API_BASE || location.origin).replace(/\/?$/, "/");
  var healthSequence = 0;
  function ready() { return state.health === "ready" || state.health === "mock"; }
  function node(tag, className, text) { var n = document.createElement(tag); if (className) n.className = className; if (text !== undefined) n.textContent = text; return n; }
  function button(text, action) { var b = node("button", "text-button", text); b.type = "button"; b.addEventListener("click", action); return b; }
  function timeText(date) { return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }); }
  function duration(seconds) { return String(Math.floor(seconds / 60)).padStart(2, "0") + ":" + String(seconds % 60).padStart(2, "0"); }
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
    if (state.active && state.active.record.group === group) cancelActive("已停止等待", "后台可能仍在处理。");
    if (group === state.group) clearPending();
    group.records.forEach(function (record) { URL.revokeObjectURL(record.url); });
    state.groups = state.groups.filter(function (item) { return item !== group; });
    if (!state.groups.length) {
      var key = ++state.nextGroup;
      state.groups.push({ key: key, name: "工作区 " + key, records: [], selected: null });
    }
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
    el.timeline.replaceChildren(); el.historyList.replaceChildren();
    group.records.forEach(function (record) { el.timeline.append(record.article); el.historyList.append(record.history); });
    renderGroups(); renderControls();
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
  function renderControls() {
    var busy = Boolean(state.active);
    el.chooseButton.disabled = busy; el.welcomeUpload.disabled = busy; el.fileInput.disabled = busy; el.removeFile.disabled = busy;
    el.submitButton.disabled = busy || !state.pending || !ready();
    el.submitButton.textContent = busy ? "正在审核…" : "开始审核 ↑";
    el.composerHint.textContent = busy ? "完成后可继续提交下一张" : "或拖拽图片到这里";
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
    finally { clearTimeout(timer); if (sequence === healthSequence) renderHealth(); }
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
  function createRecord(pending) {
    var record = { key: ++state.nextKey, displayIndex: state.group.records.length + 1, group: state.group, file: pending.file, url: pending.url, createdAt: new Date(), status: "loading", mock: state.health === "mock" };
    record.article = node("article", "review-entry"); record.article.tabIndex = -1;
    record.article.setAttribute("aria-label", "图片审核 " + record.displayIndex + "：" + record.file.name);
    var heading = node("div", "entry-heading"); heading.append(node("strong", "", "提交图片 " + String(record.displayIndex).padStart(2, "0")));
    var time = node("time", "", timeText(record.createdAt)); time.dateTime = record.createdAt.toISOString(); heading.append(time);
    var img = node("img", "submitted-image"); img.src = record.url; img.alt = "待审核图片：" + record.file.name;
    record.answer = node("div", "answer"); record.article.append(heading, img, node("p", "submitted-name", record.file.name), record.answer);
    record.history = node("button", "history-item"); record.history.type = "button";
    var thumb = node("img"); thumb.src = record.url; thumb.alt = "";
    var description = node("div"); record.historyStatus = node("small");
    description.append(node("strong", "", record.file.name), record.historyStatus); record.history.append(thumb, description);
    record.history.addEventListener("click", function () { selectRecord(record, true); });
    state.group.records.push(record); el.timeline.append(record.article); el.historyList.append(record.history); selectRecord(record, false); return record;
  }
  function renderAnswer(record) {
    record.answer.replaceChildren(); record.retryButton = null;
    record.answer.className = "answer" + (record.status === "error" ? " answer-error" : "");
    record.history.dataset.status = record.status;
    record.article.setAttribute("aria-busy", String(record.status === "loading"));
    record.historyStatus.textContent = timeText(record.createdAt) + " · " + ({ loading: "审核中", success: "已完成", error: "未完成" }[record.status]);
    var title = node("h3", "", record.status === "error" ? record.error.title : "审核建议");
    if (record.mock) title.append(node("span", "answer-label", "模拟模式")); record.answer.append(title);
    if (record.status === "loading") {
      var line = node("p", "loading-text"); record.elapsedNode = node("span", "elapsed", "00:00");
      line.append(node("span", "spinner"), node("span", "", "正在分析图片"), record.elapsedNode);
      record.answer.append(line, node("p", "loading-note", "通常需要 15–30 秒，复杂图片可能更久。"), button("取消等待", function () { cancelActive("已取消等待", "页面已停止等待，后台可能仍在处理。重试会发起一次新的审核。"); }));
    } else if (record.status === "success") {
      record.answer.append(node("p", "suggestion", record.result.suggestion));
      var meta = node("div", "result-meta"); meta.append(node("span", "", "审核编号"), node("code", "", record.result.id)); record.answer.append(meta);
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
    renderAnswer(active.record); renderControls(); announce(error ? error.title : "审核完成，可继续提交下一张图片。");
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
    var form = new FormData(); form.append("image", record.file);
    try {
      var response = await fetch(API_ADDRESS + "api/review", { method: "POST", body: form, signal: active.controller.signal });
      if (state.active !== active) return;
      var body;
      try { body = await response.json(); } catch (_) { finish(active, { title: "响应格式异常", message: "服务返回了无法解析的内容，请稍后重试。" }); return; }
      if (state.active !== active) return;
      if (!response.ok) finish(active, { title: "审核未完成", message: typeof body?.message === "string" ? body.message : "服务返回错误，请稍后重试。", code: body?.code, id: body?.id });
      else if (!body || typeof body.id !== "string" || typeof body.suggestion !== "string") finish(active, { title: "响应字段异常", message: "审核编号或建议字段不完整，请检查服务。" });
      else finish(active, null, { id: body.id, suggestion: body.suggestion });
    } catch (_) { finish(active, { title: "无法连接审核服务", message: "请检查后端和网络连接，再重试这张图片。" }); }
  }
  function clearWorkspace() {
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
  [el.chooseButton, el.welcomeUpload].forEach(function (b) { b.addEventListener("click", function () { if (!state.active) el.fileInput.click(); }); });
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
  el.clearHistory.addEventListener("click", clearWorkspace);
  el.newWorkspace.addEventListener("click", function () {
    var id = ++state.nextGroup, group = { key: id, name: "工作区 " + id, records: [], selected: null };
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
    renderGroups();
    announce("工作区已命名为" + name + "。");
  });
  el.workspaceName.addEventListener("input", function () { el.workspaceName.setCustomValidity(""); });
  [el.cancelRename, el.closeRename].forEach(function (item) { item.addEventListener("click", function () { el.renameDialog.close(); }); });
  el.renameDialog.addEventListener("close", function () { state.renameTarget = null; });
  el.menuDelete.addEventListener("click", function () {
    if (!state.menuGroup) return;
    state.deleteTarget = state.menuGroup;
    el.deleteQuestion.textContent = "“" + state.deleteTarget.name + "”中的 " + state.deleteTarget.records.length + " 条审核记录也会被删除。";
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
