/* ImageGuard 前端：原生状态管理，只使用 GET /health 与 POST /api/review。 */
(function () {
  "use strict";

  var MAX_IMAGE_BYTES = 10 * 1024 * 1024;
  var REVIEW_TIMEOUT_MS = 120000;
  var HEALTH_TIMEOUT_MS = 5000;
  var API_ADDRESS = "http://localhost:8000/";
  var ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
  var ACCEPTED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"];

  var state = {
    health: location.protocol === "file:" ? "offline-preview" : "checking",
    file: null,
    previewUrl: "",
    view: "empty", // empty | loading | success | error
    result: null,
    error: null,
    elapsed: 0,
    requestNumber: 0,
    activeRequest: null,
  };

  var ids = [
    "modeBanner", "modeBannerTitle", "modeBannerText", "copyAddressButton",
    "servicePill", "serviceText", "healthRetryButton", "fileInput", "dropZone",
    "emptyUpload", "previewWrap", "imagePreview", "fileHelp", "localError",
    "fileInfo", "fileName", "fileMeta", "removeButton", "reviewButton",
    "reviewButtonText", "resultPane", "resultTitle", "resultModeBadge", "emptyResult",
    "loadingResult", "loadingAnnouncement", "elapsedTime", "cancelButton",
    "successResult", "resultIcon", "displayHint", "suggestionText", "requestId",
    "copyIdButton", "restartButton", "errorResult", "errorTitle", "errorMessage",
    "errorMeta", "errorCodeWrap", "errorCode", "errorIdWrap", "errorId",
    "retryButton", "toast"
  ];
  var el = {};

  function collectElements() {
    var missing = [];
    ids.forEach(function (id) {
      el[id] = document.getElementById(id);
      if (!el[id]) missing.push(id);
    });
    if (missing.length) {
      document.body.innerHTML = '<main style="max-width:720px;margin:80px auto;padding:24px;font-family:sans-serif">' +
        '<h1>页面加载失败</h1><p>前端文件版本不一致，请重新部署页面资源。</p></main>';
      throw new Error("缺少页面元素: " + missing.join(", "));
    }
  }

  function isServiceReady() {
    return state.health === "ready" || state.health === "mock";
  }

  function formatTime(seconds) {
    var minutes = Math.floor(seconds / 60);
    var rest = seconds % 60;
    return String(minutes).padStart(2, "0") + ":" + String(rest).padStart(2, "0");
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
  }

  function setHidden(node, hidden) {
    node.hidden = hidden;
  }

  function renderHealth() {
    var config = {
      "offline-preview": ["service-pill--error", "离线界面预览"],
      "checking": ["service-pill--checking", "正在连接审核服务"],
      "ready": ["service-pill--ready", "审核服务已连接"],
      "mock": ["service-pill--mock", "Mock 服务已连接"],
      "not-ready": ["service-pill--error", "审核服务尚未就绪"],
      "offline": ["service-pill--error", "无法连接审核服务"]
    }[state.health];

    el.servicePill.className = "service-pill " + config[0];
    el.serviceText.textContent = config[1];
    el.healthRetryButton.hidden = !(state.health === "not-ready" || state.health === "offline");

    if (state.health === "offline-preview") {
      el.modeBanner.className = "mode-banner";
      el.modeBanner.hidden = false;
      el.modeBannerTitle.textContent = "本地界面预览";
      el.modeBannerText.textContent = "页面样式与选图预览可正常使用；启动后端后请访问 " + API_ADDRESS + " 进行真实审核。";
      el.copyAddressButton.hidden = false;
    } else if (state.health === "mock") {
      el.modeBanner.className = "mode-banner mode-banner--mock";
      el.modeBanner.hidden = false;
      el.modeBannerTitle.textContent = "模拟演示模式";
      el.modeBannerText.textContent = "当前建议由 Mock 服务生成，仅用于页面和流程联调，不代表模型效果或项目实验指标。";
      el.copyAddressButton.hidden = true;
    } else {
      el.modeBanner.hidden = true;
    }
    el.resultModeBadge.hidden = state.health !== "mock";
  }

  function renderFile() {
    var hasFile = Boolean(state.file);
    el.emptyUpload.hidden = hasFile;
    el.previewWrap.hidden = !hasFile;
    el.fileInfo.hidden = !hasFile;
    if (hasFile) {
      el.fileName.textContent = state.file.name || "未命名图片";
      var format = (state.file.type || extensionOf(state.file.name) || "未知格式")
        .replace("image/", "").toUpperCase();
      el.fileMeta.textContent = format + " · " + formatBytes(state.file.size);
    }
  }

  function renderResult() {
    el.emptyResult.hidden = state.view !== "empty";
    el.loadingResult.hidden = state.view !== "loading";
    el.successResult.hidden = state.view !== "success";
    el.errorResult.hidden = state.view !== "error";
    el.resultPane.setAttribute("aria-busy", state.view === "loading" ? "true" : "false");

    if (state.view === "loading") {
      el.elapsedTime.textContent = formatTime(state.elapsed);
    } else if (state.view === "success" && state.result) {
      el.suggestionText.textContent = state.result.suggestion;
      el.requestId.textContent = state.result.id;
      decorateSuggestion(state.result.suggestion);
    } else if (state.view === "error" && state.error) {
      el.errorTitle.textContent = state.error.title;
      el.errorMessage.textContent = state.error.message;
      el.errorMeta.hidden = !(state.error.code || state.error.id);
      el.errorCodeWrap.hidden = !state.error.code;
      el.errorCode.textContent = state.error.code || "";
      el.errorIdWrap.hidden = !state.error.id;
      el.errorId.textContent = state.error.id || "";
    }
  }

  function renderControls() {
    var loading = state.view === "loading";
    el.dropZone.disabled = loading;
    el.removeButton.disabled = loading;
    el.fileInput.disabled = loading;

    if (loading) {
      el.reviewButton.disabled = true;
      el.reviewButtonText.textContent = "正在审核 " + formatTime(state.elapsed);
    } else if (!state.file) {
      el.reviewButton.disabled = true;
      el.reviewButtonText.textContent = "请选择图片";
    } else if (!isServiceReady()) {
      el.reviewButton.disabled = true;
      el.reviewButtonText.textContent = state.health === "offline-preview" ? "启动后端后可审核" : "审核服务不可用";
    } else {
      el.reviewButton.disabled = false;
      el.reviewButtonText.textContent = "开始智能审核";
    }
  }

  function render() {
    renderHealth();
    renderFile();
    renderResult();
    renderControls();
  }

  async function checkHealth() {
    if (location.protocol === "file:") {
      state.health = "offline-preview";
      render();
      return;
    }

    state.health = "checking";
    render();
    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, HEALTH_TIMEOUT_MS);
    try {
      var response = await fetch("/health", { cache: "no-store", signal: controller.signal });
      if (!response.ok) throw new Error("health request failed");
      var body = await response.json();
      if (body.mock === true && body.service_ready === true) state.health = "mock";
      else if (body.status === "ok" && body.service_ready === true && body.model_loaded === true) state.health = "ready";
      else state.health = "not-ready";
    } catch (error) {
      state.health = "offline";
    } finally {
      clearTimeout(timer);
      render();
    }
  }

  function extensionOf(name) {
    var parts = String(name || "").toLowerCase().split(".");
    return parts.length > 1 ? parts.pop() : "";
  }

  function validateFile(file) {
    if (!file || file.size === 0) return "所选图片为空，请重新选择。";
    var typeAccepted = ACCEPTED_TYPES.indexOf(file.type) !== -1;
    var extensionAccepted = ACCEPTED_EXTENSIONS.indexOf(extensionOf(file.name)) !== -1;
    if (!typeAccepted && !extensionAccepted) return "暂不支持该文件，请选择 JPG、PNG 或 WEBP 图片。";
    if (file.size > MAX_IMAGE_BYTES) return "图片超过 10MB，请压缩后重试。";
    return "";
  }

  function revokePreview() {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = "";
  }

  function cancelActiveRequest() {
    if (!state.activeRequest) return;
    var active = state.activeRequest;
    state.requestNumber += 1;
    active.controller.abort();
    clearTimeout(active.timeoutTimer);
    clearInterval(active.elapsedTimer);
    state.activeRequest = null;
  }

  function resetReport() {
    state.view = "empty";
    state.result = null;
    state.error = null;
    state.elapsed = 0;
  }

  function acceptFile(file) {
    var problem = validateFile(file);
    el.fileInput.value = "";
    if (problem) {
      el.localError.textContent = problem + (state.file ? " 当前已选图片仍然保留。" : "");
      el.localError.hidden = false;
      return;
    }

    cancelActiveRequest();
    revokePreview();
    state.file = file;
    state.previewUrl = URL.createObjectURL(file);
    el.imagePreview.src = state.previewUrl;
    el.localError.hidden = true;
    resetReport();
    render();
  }

  function clearFile() {
    cancelActiveRequest();
    revokePreview();
    state.file = null;
    el.fileInput.value = "";
    el.imagePreview.removeAttribute("src");
    el.localError.hidden = true;
    resetReport();
    render();
    el.dropZone.focus();
  }

  function showError(title, message, code, id) {
    state.view = "error";
    state.error = { title: title, message: message, code: code || "", id: id || "" };
    render();
    el.errorTitle.focus();
  }

  function decorateSuggestion(text) {
    var tier = suggestionTier(text);
    el.successResult.className = "result-view result-view--success";
    el.resultIcon.className = "result-icon";
    var hint = "审核建议已生成";
    var icon = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v5m0-8h.01"/></svg>';

    if (tier === "safe") {
      el.successResult.classList.add("success-safe");
      el.resultIcon.classList.add("is-safe");
      hint = "展示提示 · 低关注";
      icon = '<svg viewBox="0 0 24 24"><path d="M5 12.5 10 17 19 7"/></svg>';
    } else if (tier === "watch" || tier === "degraded") {
      el.successResult.classList.add("success-watch");
      el.resultIcon.classList.add("is-watch");
      hint = tier === "degraded" ? "展示提示 · 服务降级" : "展示提示 · 建议关注";
      icon = '<svg viewBox="0 0 24 24"><path d="M12 9v4m0 4h.01"/><path d="M10.3 4 3.6 16a2 2 0 0 0 1.8 3h13.2a2 2 0 0 0 1.8-3L13.7 4a2 2 0 0 0-3.4 0Z"/></svg>';
    } else if (tier === "high") {
      el.successResult.classList.add("success-high");
      el.resultIcon.classList.add("is-high");
      hint = "展示提示 · 高关注";
      icon = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v6m0 4h.01"/></svg>';
    }
    el.displayHint.textContent = hint;
    el.resultIcon.innerHTML = icon;
  }

  function suggestionTier(text) {
    var value = String(text || "");
    if (/暂不可用|无法完成|稍后重试|流程不可用/.test(value)) return "degraded";
    if (/未发现严重违规|没有严重暴力|不建议下架|无需下架|建议放行|审核通过/.test(value)) return "safe";
    if (/严重|下架|拦截/.test(value)) return "high";
    if (/人工复核|建议复核|无法判定|中度|轻微|限流|关注/.test(value)) return "watch";
    if (/安全|放行|未发现/.test(value)) return "safe";
    return "neutral";
  }

  async function submitReview() {
    if (!state.file || state.view === "loading" || !isServiceReady()) return;

    cancelActiveRequest();
    var requestNumber = state.requestNumber + 1;
    state.requestNumber = requestNumber;
    var controller = new AbortController();
    var timedOut = false;
    var cancelledByUser = false;

    state.view = "loading";
    state.result = null;
    state.error = null;
    state.elapsed = 0;
    render();

    var elapsedTimer = setInterval(function () {
      if (!state.activeRequest || state.activeRequest.number !== requestNumber) return;
      state.elapsed += 1;
      el.elapsedTime.textContent = formatTime(state.elapsed);
      el.reviewButtonText.textContent = "正在审核 " + formatTime(state.elapsed);
    }, 1000);
    var timeoutTimer = setTimeout(function () {
      if (!state.activeRequest || state.activeRequest.number !== requestNumber) return;
      timedOut = true;
      controller.abort();
    }, REVIEW_TIMEOUT_MS);

    state.activeRequest = {
      number: requestNumber,
      controller: controller,
      elapsedTimer: elapsedTimer,
      timeoutTimer: timeoutTimer,
      cancel: function () { cancelledByUser = true; controller.abort(); }
    };

    var form = new FormData();
    form.append("image", state.file);

    try {
      var response = await fetch("/api/review", { method: "POST", body: form, signal: controller.signal });
      var body;
      try {
        body = await response.json();
      } catch (parseError) {
        if (state.requestNumber === requestNumber) {
          showError("响应格式异常", "审核服务返回了无法识别的内容，请检查后端运行日志。", "", "");
        }
        return;
      }
      if (state.requestNumber !== requestNumber) return;

      if (!response.ok) {
        showError("审核未完成", body.message || "审核服务返回错误，请稍后重试。", body.code, body.id);
      } else if (typeof body.id !== "string" || typeof body.suggestion !== "string") {
        showError("响应字段异常", "审核服务返回的 id 或 suggestion 字段不完整。", "", "");
      } else {
        state.view = "success";
        state.result = { id: body.id, suggestion: body.suggestion };
        state.error = null;
        render();
        el.resultTitle.focus();
      }
    } catch (error) {
      if (state.requestNumber !== requestNumber) return;
      if (error && error.name === "AbortError") {
        if (timedOut) showError("等待时间过长", "页面已等待 120 秒并停止本次请求。后端任务可能仍在处理中，重新提交会生成新的审核编号。", "", "");
        else if (cancelledByUser) showError("已取消等待", "页面已停止等待。后端任务可能仍在处理中，你可以稍后重新提交。", "", "");
        else showError("请求已终止", "本次审核请求已停止。", "", "");
      } else {
        showError("无法连接审核服务", "请确认后端已启动，并检查当前网络连接后重试。", "", "");
      }
    } finally {
      clearInterval(elapsedTimer);
      clearTimeout(timeoutTimer);
      if (state.activeRequest && state.activeRequest.number === requestNumber) state.activeRequest = null;
      renderControls();
    }
  }

  function userCancel() {
    if (!state.activeRequest) return;
    state.activeRequest.cancel();
  }

  async function copyText(text, successMessage) {
    try {
      await navigator.clipboard.writeText(text);
      showToast(successMessage);
    } catch (error) {
      var area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      var copied = document.execCommand("copy");
      area.remove();
      showToast(copied ? successMessage : "复制失败，请手动复制");
    }
  }

  function showToast(message) {
    el.toast.textContent = message;
    el.toast.hidden = false;
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(function () { el.toast.hidden = true; }, 1800);
  }

  function bindEvents() {
    el.fileInput.addEventListener("change", function () {
      acceptFile(el.fileInput.files && el.fileInput.files[0]);
    });
    el.dropZone.addEventListener("click", function () {
      if (state.view !== "loading") el.fileInput.click();
    });
    ["dragenter", "dragover"].forEach(function (name) {
      el.dropZone.addEventListener(name, function (event) {
        event.preventDefault();
        if (state.view !== "loading") el.dropZone.classList.add("is-dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      el.dropZone.addEventListener(name, function (event) {
        event.preventDefault();
        el.dropZone.classList.remove("is-dragover");
      });
    });
    el.dropZone.addEventListener("drop", function (event) {
      if (state.view === "loading") return;
      var files = event.dataTransfer && event.dataTransfer.files;
      if (!files || !files.length) return;
      if (files.length > 1) {
        el.localError.textContent = "一次只能审核一张图片，请重新选择。";
        el.localError.hidden = false;
        return;
      }
      acceptFile(files[0]);
    });
    el.removeButton.addEventListener("click", clearFile);
    el.reviewButton.addEventListener("click", submitReview);
    el.cancelButton.addEventListener("click", userCancel);
    el.retryButton.addEventListener("click", submitReview);
    el.restartButton.addEventListener("click", clearFile);
    el.healthRetryButton.addEventListener("click", checkHealth);
    el.copyAddressButton.addEventListener("click", function () { copyText(API_ADDRESS, "访问地址已复制"); });
    el.copyIdButton.addEventListener("click", function () {
      if (state.result) copyText(state.result.id, "审核编号已复制");
    });
    window.addEventListener("beforeunload", function () {
      cancelActiveRequest();
      revokePreview();
    });
  }

  function init() {
    collectElements();
    bindEvents();
    render();
    checkHealth();
  }

  init();
})();
