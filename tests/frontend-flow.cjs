// Run with NODE_PATH pointing to a directory containing Playwright.
const { chromium } = require('playwright');
const http = require('node:http');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j8WQAAAAASUVORK5CYII=', 'base64');
(async () => {
  const batchDir = fs.mkdtempSync(path.join(os.tmpdir(), 'imageguard-batch-'));
  fs.writeFileSync(path.join(batchDir, 'folder-02.png'), png);
  fs.writeFileSync(path.join(batchDir, 'folder-01.png'), png);
  const server = http.createServer((req, res) => {
    const requestPath = req.url.split('?')[0];
    const file = path.join(root, 'frontend', requestPath === '/' ? 'index.html' : requestPath.slice(1));
    if (!['index.html', 'app.js', 'config.js', 'styles.css', 'landing.css'].includes(path.basename(file))) { res.writeHead(404).end(); return; }
    res.setHeader('Content-Type', file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : 'text/html');
    res.end(fs.readFileSync(file));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({ channel: 'msedge', headless: true });
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.route('**/health', route => route.fulfill({ json: { status: 'ok', service_ready: true, model_loaded: true, mock: true } }));
    let mode = 'success', delayed, count = 0;
    const reviewWorkspace = {};
    const structured = {
      schema_version: '2026-10-27-review-v1',
      result: {
        action: 'review', action_label: '人工复核', tone: 'review',
        effective_risk: 'L1_轻微违规', risk_raw: 'L2_中度违规',
        domain: 'virtual', domain_label: '虚拟（二次元 / 插画 / 游戏场景等）', domain_score: 0.98,
        domain_source: 'style_router', confidence: 0.732,
        reason: '风险头给出 L2 候选，binary 明确阴性，但文字语境仍无法完成核验。',
        primary_reasons: ['binary 明确阴性，否认血腥维度', '文字检测命中但语义证据不足'],
        adjustments: ['虚拟风格导致的风险与内容假冲突已解除'],
        conflict_type: 'pseudo_binary_conflict', uncertain: true,
        policy_version: '2026-10-26-effective-v5',
        suggestion_source: 'llm_explanation', degraded: false,
        recommended_action: 'limit', recommended_action_label: '建议限流',
        review_hint: true, review_reason: '存在非轻度风险候选，建议人工确认前先限流。'
      },
      groups: [
        { id: 'basic_heads', label: 'CLIP 基础头', description: '风险、类型、伤害与质量。', tools: [
          { id: 'risk_head', label: '风险等级头', status: 'ok', status_label: '已完成', outcome: 'risk', outcome_label: '发现风险', summary: '原始候选 L2，置信度 0.732', duration_ms: null, details: { scores: { L2: 0.732 } } },
          { id: 'quality_fusion', label: '质量融合头', status: 'ok', status_label: '已完成', outcome: 'neutral', outcome_label: '中性证据', summary: '轻微模糊', duration_ms: null, details: { labels: ['轻微模糊'] } }
        ] },
        { id: 'specialized_context', label: '专项模型与语境', description: '二分类、虚拟风格路由与场景 Prompt。', tools: [
          { id: 'blood_binary', label: '血腥二分类头', status: 'ok', status_label: '已完成', outcome: 'safe', outcome_label: '支持安全', summary: '明确无血腥支持', duration_ms: 12, details: { p_blood: 0.05 } },
          { id: 'anime_router', label: '虚拟风格路由', status: 'ok', status_label: '已完成', outcome: 'neutral', outcome_label: '中性证据', summary: '判定为虚拟内容，分数 0.980', duration_ms: 8, details: { score: 0.98 } }
        ] },
        { id: 'text_rules', label: '文字与规则', description: '文字检测、OCR 与本地 RAG。', tools: [
          { id: 'text_detector', label: '轻量文字检测', status: 'detected', status_label: '已检出', outcome: 'risk', outcome_label: '发现风险', summary: '发现 2 个文字候选区域', duration_ms: 86, details: { box_count: 2 } },
          { id: 'ocr', label: 'OCR 文字识别', status: 'ok', status_label: '已完成', outcome: 'neutral', outcome_label: '中性证据', summary: '识别 2 段文字', duration_ms: 1450, details: { text_count: 2 } },
          { id: 'text_semantic_rag', label: '文字语义与 RAG', status: 'uncertain', status_label: '不确定', outcome: 'mixed', outcome_label: '证据冲突', summary: '文字语义无法可靠确认', duration_ms: 1800, details: { high_risk_rule_ids: ['R006'] } }
        ] },
        { id: 'branch_policy', label: '分支与最终裁决', description: '复核和程序动作。', tools: [
          { id: 'anime_visual_review', label: '语义视觉证据复核', status: 'skipped', status_label: '已跳过', outcome: 'not_applicable', outcome_label: '不适用', summary: '门控未通过', duration_ms: null, details: { gate: { eligible: false } } },
          { id: 'effective_policy', label: '有效风险与最终裁决', status: 'ok', status_label: '已完成', outcome: 'mixed', outcome_label: '证据冲突', summary: 'L2 → L1；最终动作 人工复核', duration_ms: 1, details: {} },
          { id: 'explanation', label: 'LLM 解释生成', status: 'template', status_label: '程序模板', outcome: 'neutral', outcome_label: '中性证据', summary: '未调用 LLM，由程序固定模板直接生成说明', duration_ms: null, details: { source: 'program_template' } }
        ] }
      ],
      timing: { total_ms: 8420, stages: [
        { id: 'analyze_image', label: 'FG-CLIP2 视觉推理', category: 'model', duration_ms: 680 },
        { id: 'ocr_image', label: 'OCR 文字识别', category: 'text', duration_ms: 1450 },
        { id: 'text.semantic', label: '文字语义复核', category: 'model', duration_ms: 1800 },
        { id: 'explanation', label: 'LLM 解释生成', category: 'model', duration_ms: 3200 },
        { id: 'service.total', label: '后端总处理', category: 'service', duration_ms: 8420 }
      ] }
    };
    await page.route('**/api/review', async route => {
      count++; const current = mode;
      assert.match(route.request().headers()['content-type'], /multipart/);
      const multipart = (route.request().postDataBuffer() || Buffer.alloc(0)).toString('utf8');
      const workspaceMatch = multipart.match(/name="workspace_id"\r\n\r\n([^\r\n]+)/);
      reviewWorkspace['review-' + count] = workspaceMatch ? workspaceMatch[1] : 'default';
      if (current === 'delay') { await new Promise(resolve => { delayed = resolve; }); }
      if (current === 'error') await route.fulfill({ status: 400, json: { code: 40001, message: '图片无法解析', id: 'error-id' } });
      else if (current === 'network') await route.abort();
      else if (current === 'invalid') await route.fulfill({ body: 'not-json', contentType: 'text/plain' });
      else if (current === 'legacy') await route.fulfill({ json: { id: 'legacy-' + count, suggestion: '旧版后端只返回建议文本。' } });
      else {
        const reviewId = 'review-' + count;
        const workspaceId = reviewWorkspace[reviewId] || 'default';
        queueRecords.unshift({
          id: reviewId, filename: ['first.png', 'second.png', 'failed.png', 'cancel.png', 'after-cancel.png', 'offline.png', 'invalid.png', 'timeout.png', 'clear-during-request.png', 'final.png', 'grouped.png'][Math.min(count - 1, 10)],
          workspace_id: workspaceId, domain: 'virtual', action: 'review', action_label: '人工复核',
          final_action: 'review', final_action_label: '人工复核', effective_risk: 'L2_中度违规',
          current_level: 'L2', current_level_label: 'L2 · 人工复核', risk_raw: 'L2_中度违规',
          queue: 'human', manual_status: 'pending', adjustment_direction: '', feedback_reason: '',
          has_image: false, image_url: `/api/records/${reviewId}/image`,
          response: { ...structured, id: reviewId, suggestion: '<script>unsafe</script>\n建议人工复核。' }
        });
        await route.fulfill({ json: { ...structured, id: reviewId, suggestion: '<script>unsafe</script>\n建议人工复核。' } }).catch(() => {});
      }
    });
    const persistedResponse = {
      ...structured,
      id: 'persisted-1',
      suggestion: '持久化的人工复核记录。'
    };
    let queueRecords = [
      {
        id: 'auto-1', filename: 'auto.png', domain: 'real',
        workspace_id: 'default',
        action: 'allow', action_label: '放行', final_action: 'allow',
        final_action_label: '放行', effective_risk: 'L0_安全', risk_raw: 'L0_安全',
        current_level: 'L0', current_level_label: 'L0 · 放行',
        queue: 'auto', manual_status: 'none', adjustment_direction: '',
        feedback_reason: '', has_image: false, image_url: '/api/records/auto-1/image',
        response: { ...structured, id: 'auto-1', suggestion: '自动放行。' }
      },
      {
        id: 'persisted-1', filename: 'human.png', domain: 'virtual',
        workspace_id: 'default',
        action: 'review', action_label: '人工复核', final_action: 'review',
        final_action_label: '人工复核', effective_risk: 'L2_中度违规',
        current_level: 'L2', current_level_label: 'L2 · 人工复核', risk_raw: 'L2_中度违规',
        queue: 'human', manual_status: 'pending',
        adjustment_direction: '', feedback_reason: '', has_image: false,
        image_url: '/api/records/persisted-1/image', response: persistedResponse
      }
    ];
    await page.route('**/api/records?*', route => {
      const params = new URL(route.request().url()).searchParams;
      const items = queueRecords.filter(record =>
        (!params.get('workspace_id') || record.workspace_id === params.get('workspace_id')) &&
        (!params.get('domain') || record.domain === params.get('domain')) &&
        (!params.get('action') || record.final_action === params.get('action')) &&
        (!params.get('risk') || record.effective_risk === params.get('risk'))
      );
      const workspaceTotal = queueRecords.filter(record =>
        !params.get('workspace_id') || record.workspace_id === params.get('workspace_id')
      ).length;
      return route.fulfill({ json: { items, total: workspaceTotal } });
    });
    await page.route('**/api/records/clear?*', route => {
      const workspaceId = new URL(route.request().url()).searchParams.get('workspace_id');
      const before = queueRecords.length;
      queueRecords = queueRecords.filter(record => record.workspace_id !== workspaceId);
      return route.fulfill({ json: { workspace_id: workspaceId, cleared: before - queueRecords.length } });
    });
    await page.route('**/api/records/export.csv', route => route.fulfill({ body: 'id,filename\npersisted-1,human.png', contentType: 'text/csv' }));
    await page.route('**/api/records/persisted-1', route => route.fulfill({ json: queueRecords[1] }));
    await page.route('**/api/records/*/feedback', async route => {
      const body = route.request().postDataJSON();
      const requestedId = decodeURIComponent(route.request().url().match(/records\/([^/]+)\/feedback/)[1]);
      const index = queueRecords.findIndex(item => item.id === requestedId);
      if (index < 0) {
        queueRecords.unshift({
          id: requestedId, filename: 'current.png', domain: 'virtual',
          workspace_id: reviewWorkspace[requestedId] || 'default',
          action: 'review', action_label: '人工复核', final_action: 'review',
          final_action_label: '人工复核', effective_risk: 'L1_轻微违规',
          current_level: 'L2', current_level_label: 'L2 · 人工复核',
          risk_raw: 'L2_中度违规', queue: 'human', manual_status: 'pending',
          adjustment_direction: '', feedback_reason: '', has_image: false,
          image_url: `/api/records/${requestedId}/image`, response: structured
        });
      }
      const targetIndex = queueRecords.findIndex(item => item.id === requestedId);
      queueRecords[targetIndex] = {
        ...queueRecords[targetIndex],
        final_action: body.resolution === 'reject' ? 'review' : body.final_action,
        final_action_label: body.resolution === 'reject' ? '人工复核' : ({ allow: '放行', limit: '限流', review: '人工复核', block: '下架' }[body.final_action]),
        current_level: ({ allow: 'L0', limit: 'L1', review: 'L2', block: 'L3' }[body.resolution === 'reject' ? 'review' : body.final_action]),
        current_level_label: `${({ allow: 'L0', limit: 'L1', review: 'L2', block: 'L3' }[body.resolution === 'reject' ? 'review' : body.final_action])} · ${body.resolution === 'reject' ? '人工复核' : ({ allow: '放行', limit: '限流', review: '人工复核', block: '下架' }[body.final_action])}`,
        queue: 'human',
        manual_status: body.resolution === 'adjust' ? 'adjusted' : body.resolution === 'reject' ? 'rejected' : 'confirmed',
        adjustment_direction: body.adjustment_direction || '',
        feedback_reason: body.reason || (body.resolution === 'confirm' ? '确认系统建议处置' : '')
      };
      await route.fulfill({ json: queueRecords[targetIndex] });
    });
    const origin = 'http://127.0.0.1:' + server.address().port;
    // Same-origin deployment: override the optional separate-server default.
    await page.route('**/config.js', route => route.fulfill({ body: 'window.REVIEW_API_BASE = location.origin;', contentType: 'text/javascript' }));
    await page.goto(origin); await page.waitForFunction(() => document.querySelector('#serviceText').textContent.includes('模拟'));
    assert.equal(page.url().endsWith('#workspace'), true);
    assert.equal(await page.locator('#workspaceApp').isVisible(), true); assert.equal(await page.locator('#landingPage').isVisible(), false);
    assert.equal(await page.locator('#landingPage').evaluate(n => n.inert), true);
    assert.equal(await page.locator('#openSidebar').getAttribute('title'), '收起工作区导航');
    assert.equal(await page.locator('#openSidebar').evaluate(n => getComputedStyle(n, '::after').content), 'none');
    const helpHitArea = await page.locator('#helpButton').evaluate(n => {
      const box = n.getBoundingClientRect(); const samples = [[8, 8], [box.width - 8, 8], [8, box.height - 8], [box.width - 8, box.height - 8], [box.width / 2, box.height / 2]];
      return samples.every(([x, y]) => document.elementFromPoint(box.left + x, box.top + y) === n);
    });
    assert.equal(helpHitArea, true);
    await page.locator('#helpButton').click();
    assert.equal(await page.locator('#helpDialog').evaluate(n => n.open), true);
    assert.match(await page.locator('#helpSummary').textContent(), /固定工作流/);
    assert.equal(await page.locator('.help-details > div').count(), 3);
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-help.png') });
    await page.keyboard.press('Escape'); assert.equal(await page.locator('#helpDialog').evaluate(n => n.open), false);
    await page.setViewportSize({ width: 390, height: 844 }); await page.locator('#helpButton').click();
    const helpBox = await page.locator('#helpDialog').boundingBox();
    assert.ok(helpBox.x >= 0 && helpBox.x + helpBox.width <= 390 && helpBox.y >= 0 && helpBox.y + helpBox.height <= 844);
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-help-mobile.png') });
    await page.locator('#dismissHelp').click(); await page.setViewportSize({ width: 1440, height: 900 });
    await page.locator('#helpButton').click(); await page.locator('#helpLearnMore').click(); await page.locator('#landingPage').waitFor();
    assert.equal(page.url().endsWith('#home'), true); assert.equal(await page.evaluate(() => scrollY), 0);
    assert.match(await page.title(), /多模态图片内容审核/);
    assert.equal(await page.locator('#finalTitle > span').count(), 2);
    await page.locator('#landingThemeButton').click();
    await page.locator('#landingThemeMenu .theme-option[data-theme-value="dark"]').click();
    const landingColors = await page.evaluate(() => ({
      canvas: getComputedStyle(document.body).backgroundColor,
      hero: getComputedStyle(document.querySelector('.landing-hero')).backgroundColor,
      sample: getComputedStyle(document.querySelector('.hero-result')).backgroundColor
    }));
    assert.notEqual(landingColors.hero, landingColors.canvas);
    assert.equal(landingColors.sample, 'rgb(255, 255, 255)');
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-landing-dark.png'), fullPage: true });
    await page.locator('#landingThemeButton').click();
    await page.locator('#landingThemeMenu .theme-option[data-theme-value="light"]').click();
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-landing.png'), fullPage: true });
    await page.locator('#landingThemeButton').click();
    await page.locator('#landingThemeMenu .theme-option[data-theme-value="system"]').click();
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-landing-mobile.png'), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.locator('.landing-links a[href="#workflow"]').click(); await page.waitForFunction(() => scrollY > 600);
    await page.locator('#landingBrand').click(); await page.waitForFunction(() => scrollY < 5);
    await page.locator('.landing-links a[href="#workflow"]').click(); await page.waitForFunction(() => scrollY > 600);
    const landingScroll = await page.evaluate(() => scrollY);
    await page.locator('#landingWorkspaceLink').click(); await page.locator('#workspaceApp').waitFor();
    assert.equal(page.url().endsWith('#workspace'), true);
    assert.equal(await page.locator('#landingPage').evaluate(n => n.inert), true); assert.match(await page.title(), /工作台/);
    await page.evaluate(() => {
      const transfer = new DataTransfer(); transfer.items.add(new File(['drag'], 'dragged.png', { type: 'image/png' }));
      const enter = new Event('dragenter', { bubbles: true }); Object.defineProperty(enter, 'dataTransfer', { value: transfer }); window.dispatchEvent(enter);
    });
    assert.equal(await page.locator('#dropOverlay').isVisible(), true);
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-drag.png') });
    await page.evaluate(() => {
      const transfer = new DataTransfer(); transfer.items.add(new File(['drag'], 'dragged.png', { type: 'image/png' }));
      const drop = new Event('drop', { bubbles: true }); Object.defineProperty(drop, 'dataTransfer', { value: transfer }); window.dispatchEvent(drop);
    });
    assert.equal(await page.locator('#pendingName').textContent(), 'dragged.png'); await page.locator('#removeFile').click();
    await page.evaluate(() => {
      const transfer = new DataTransfer(); transfer.items.add(new File(['paste'], 'pasted.png', { type: 'image/png' }));
      const paste = new Event('paste', { bubbles: true }); Object.defineProperty(paste, 'clipboardData', { value: transfer }); window.dispatchEvent(paste);
    });
    assert.equal(await page.locator('#pendingName').textContent(), 'pasted.png'); await page.locator('#removeFile').click();
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-desktop.png') });
    await page.waitForFunction(() => document.querySelectorAll('#autoQueue .queue-item').length === 2);
    assert.equal(await page.locator('#autoQueue .queue-item').count(), 2);
    assert.match(await page.locator('#autoQueue').textContent(), /待人工复核/);
    assert.equal(await page.locator('#confirmedQueue .queue-item').count(), 0);
    assert.equal(await page.locator('#adjustedQueue .queue-item').count(), 0);
    await page.locator('#queueDomain').selectOption('virtual');
    await page.waitForFunction(() => document.querySelectorAll('#autoQueue .queue-item').length === 1);
    assert.match(await page.locator('#autoQueue').textContent(), /human.png/);
    assert.equal(await page.locator('#autoQueue').textContent().then(value => value.includes('auto.png')), false);
    await page.locator('#queueDomain').selectOption('');
    await page.waitForFunction(() => document.querySelectorAll('#autoQueue .queue-item').length === 2);
    await page.locator('#openSidebar').click(); assert.equal(await page.locator('#workspaceApp').getAttribute('class'), 'app-layout is-collapsed');
    assert.equal(await page.locator('#sidebar').evaluate(n => n.inert), true);
    assert.equal(await page.evaluate(() => localStorage.getItem('imageguard-sidebar')), 'collapsed');
    assert.equal(await page.locator('#openSidebar').getAttribute('title'), '展开工作区导航');
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-collapsed.png') });
    await page.locator('#openSidebar').click(); assert.equal(await page.locator('#sidebar').evaluate(n => n.inert), false);
    const upload = async name => page.locator('#fileInput').setInputFiles({ name, mimeType: 'image/png', buffer: png });
    const submit = async name => { await upload(name); await page.locator('#submitButton').click(); };
    await submit('first.png'); await page.locator('.decision-panel').waitFor();
    assert.match(await page.locator('.decision-panel').textContent(), /人工复核/);
    assert.match(await page.locator('.decision-panel').textContent(), /虚拟（二次元 \/ 插画 \/ 游戏场景等）/);
    assert.match(await page.locator('.review-recommendation').textContent(), /建议限流/);
    assert.equal(await page.locator('.decision-metric').count(), 5);
    assert.equal(await page.locator('.evidence-group').count(), 4);
    assert.equal(await page.locator('.tool-row').count(), 10);
    assert.equal(await page.locator('.status-badge[data-status="ok"]').count() > 0, true);
    assert.equal(await page.locator('.outcome-badge[data-outcome="risk"]').count() > 0, true);
    assert.match(await page.locator('.total-duration').textContent(), /8\.4 s/);
    const mediaLayout = await page.locator('.review-media').evaluate(node => {
      const image = node.querySelector('.submitted-image').getBoundingClientRect();
      const actions = node.querySelector('.review-media-actions').getBoundingClientRect();
      return { actionsRightOfImage: actions.left >= image.right, actionsWidth: actions.width };
    });
    assert.equal(mediaLayout.actionsRightOfImage, true);
    assert.equal(mediaLayout.actionsWidth > 150, true);
    await page.locator('.review-media').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-manual-actions.png') });
    await page.locator('.decision-panel').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-structured-top.png') });
    assert.equal(await page.locator('.review-media-actions .manual-action').count(), 2);
    const historyListBox = await page.locator('#historyList').boundingBox();
    assert.ok(historyListBox && historyListBox.height >= 90);
    assert.match(await page.locator('.manual-action-confirm').textContent(), /确认/);
    assert.match(await page.locator('.manual-action-adjust').textContent(), /调整/);
    assert.equal(await page.getByText('驳回自动结果', { exact: true }).count(), 0);
    await page.locator('.manual-action-confirm').click();
    assert.equal(await page.locator('#feedbackDialog').evaluate(n => n.open), true);
    assert.equal(await page.locator('#feedbackAction').inputValue(), 'limit');
    await page.locator('#feedbackForm button[type=submit]').click();
    await page.locator('.manual-result').waitFor();
    assert.match(await page.locator('.manual-result').textContent(), /已通过/);
    await page.waitForFunction(() => document.querySelectorAll('#confirmedQueue .queue-item').length === 1);
    assert.equal(await page.locator('#adjustedQueue .queue-item').count(), 0);
    await page.locator('.manual-action-adjust').click();
    await page.locator('#feedbackResolution').selectOption('adjust');
    assert.equal(await page.locator('#feedbackDirectionDowngrade').isVisible(), true);
    assert.equal(await page.locator('#feedbackDirectionUpgrade').isVisible(), true);
    assert.equal(await page.locator('#feedbackDirectionDowngrade').isChecked(), true);
    assert.equal(await page.locator('#feedbackAction').inputValue(), 'allow');
    assert.deepEqual(await page.locator('#feedbackAction option').evaluateAll(options => options.map(option => option.value)), ['allow']);
    await page.locator('#feedbackDirectionUpgrade').check();
    assert.deepEqual(await page.locator('#feedbackAction option').evaluateAll(options => options.map(option => option.value)), ['review', 'block']);
    await page.locator('#feedbackAction').selectOption('block');
    await page.locator('#feedbackReason').fill('人工确认需要升级处置');
    assert.equal(await page.locator('#feedbackDialog').evaluate(n => n.scrollWidth <= n.clientWidth), true);
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-feedback-adjust.png') });
    await page.locator('#feedbackForm button[type=submit]').click();
    await page.waitForFunction(() => document.querySelector('.manual-result')?.textContent.includes('已人工调整'));
    assert.match(await page.locator('.manual-result').textContent(), /已人工调整/);
    assert.match(await page.locator('.manual-result').textContent(), /升级/);
    assert.match(await page.locator('.manual-result').textContent(), /下架/);
    assert.equal(await page.locator('.review-entry').first().getAttribute('class').then(value => value.includes('is-manually-adjusted')), true);
    await page.waitForFunction(() => document.querySelectorAll('#adjustedQueue .queue-item').length === 1);
    assert.equal(await page.locator('#confirmedQueue .queue-item').count(), 0);
    await page.locator('.timing-details summary').click();
    assert.equal(await page.locator('.timing-row').count(), 4);
    await page.locator('.tool-row[data-tool-id="risk_head"] button').click();
    assert.equal(await page.locator('.tool-row[data-tool-id="risk_head"] + .tool-details-row').isVisible(), true);
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-structured-details.png') });
    await page.getByText('复制建议', { exact: true }).click(); await page.getByText('已复制 ✓', { exact: true }).waitFor();
    await submit('second.png'); await page.waitForFunction(() => document.querySelectorAll('.decision-panel').length === 2);
    assert.equal(await page.locator('.review-entry').count(), 2);
    assert.equal(await page.locator('.suggestion script').count(), 0);
    await page.locator('.history-item').first().click();
    assert.equal(await page.locator('.history-item').first().getAttribute('aria-current'), 'true');
    mode = 'error'; await submit('failed.png'); await page.getByText('图片无法解析', { exact: true }).waitFor();
    mode = 'success'; await page.getByText('重新审核这张图片', { exact: true }).click();
    await page.waitForFunction(() => document.querySelectorAll('.decision-panel').length === 3);
    assert.equal(await page.locator('.review-entry').count(), 3);
    mode = 'delay'; await submit('cancel.png'); await page.getByText('取消等待', { exact: true }).waitFor();
    assert.equal(await page.locator('#chooseButton').isDisabled(), true);
    await page.getByText('取消等待', { exact: true }).click();
    mode = 'success'; await submit('after-cancel.png');
    await page.waitForFunction(() => document.querySelectorAll('.decision-panel').length === 4);
    delayed(); await page.waitForTimeout(100);
    assert.equal(await page.locator('.answer h3').filter({ hasText: '已取消等待' }).count(), 1);
    mode = 'network'; await submit('offline.png'); await page.locator('.answer h3').filter({ hasText: '无法连接审核服务' }).waitFor();
    mode = 'invalid'; await submit('invalid.png'); await page.locator('.answer h3').filter({ hasText: '响应格式异常' }).waitFor();
    await page.clock.install();
    mode = 'delay'; await submit('timeout.png'); await page.waitForFunction(() => document.querySelector('.loading-text'));
    await page.clock.runFor(120001); await page.locator('.answer h3').filter({ hasText: '等待时间过长' }).waitFor(); delayed();
    mode = 'delay'; await submit('clear-during-request.png'); await page.waitForFunction(() => document.querySelector('.loading-text'));
    await page.locator('#clearHistory').click(); delayed(); await page.waitForTimeout(50);
    assert.equal(await page.locator('.review-entry').count(), 0);
    mode = 'success'; await submit('final.png'); await page.locator('.decision-panel').waitFor();
    assert.match(await page.locator('#historyCount').getAttribute('title'), /图片张数/);
    await page.locator('#newWorkspace').click();
    assert.equal(await page.locator('#groupList .group-item').count(), 2);
    assert.equal(await page.locator('.review-entry').count(), 0);
    await page.waitForFunction(() => document.querySelectorAll('#autoQueue .queue-item').length === 0);
    assert.equal(await page.locator('#confirmedQueue .queue-item').count(), 0);
    await page.locator('#groupList .group-row').nth(1).click({ button: 'right' });
    await page.locator('#menuRename').click();
    assert.equal(await page.locator('#renameDialog').evaluate(n => n.open), true);
    const dialogBox = await page.locator('#renameDialog').boundingBox();
    assert.ok(dialogBox.x >= 0 && dialogBox.x + dialogBox.width <= 1440);
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-rename.png') });
    await page.locator('#workspaceName').fill('新闻图片');
    await page.locator('#renameForm button[type=submit]').click();
    assert.equal(await page.locator('#renameDialog').evaluate(n => n.open), false);
    assert.equal(await page.locator('#workspaceTitle').textContent(), '新闻图片');
    await submit('grouped.png'); await page.locator('.decision-panel').waitFor();
    assert.equal(await page.locator('.entry-heading strong').textContent(), '提交图片 01');
    await page.waitForFunction(() => document.querySelectorAll('#autoQueue .queue-item').length === 1);
    assert.match(await page.locator('#autoQueue').textContent(), /grouped.png/);
    await page.locator('#clearQueue').click();
    assert.equal(await page.locator('#clearQueueDialog').evaluate(node => node.open), true);
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-clear-queue.png') });
    await page.locator('#clearQueueForm button[type=submit]').click();
    await page.waitForFunction(() => document.querySelectorAll('#autoQueue .queue-item').length === 0);
    assert.equal(await page.locator('#clearQueue').isDisabled(), true);
    assert.equal(await page.locator('.decision-panel').count(), 1);
    await page.locator('#groupList .group-item').first().click();
    assert.equal(await page.locator('.submitted-name').first().textContent(), 'final.png');
    await page.waitForFunction(() => document.querySelectorAll('#autoQueue .queue-item').length > 0);
    await page.evaluate(() => {
      const nativeScrollTo = window.scrollTo.bind(window); window.__restoreBehaviors = [];
      window.scrollTo = function () { window.__restoreBehaviors.push(getComputedStyle(document.documentElement).scrollBehavior); return nativeScrollTo.apply(window, arguments); };
    });
    await page.locator('.brand').click(); await page.locator('#landingPage').waitFor();
    await page.waitForFunction(expected => Math.abs(scrollY - expected) < 5, landingScroll);
    assert.equal(await page.evaluate(() => window.__restoreBehaviors.at(-1)), 'auto');
    assert.equal(await page.locator('#landingWorkspaceLink').textContent(), '返回工作台');
    await page.locator('#landingWorkspaceLink').click(); await page.locator('#workspaceApp').waitFor();
    assert.equal(await page.locator('.submitted-name').first().textContent(), 'final.png');
    assert.equal(await page.locator('.review-entry').count(), 1);
    await page.locator('#groupList .group-item').nth(1).click();
    assert.equal(await page.locator('.submitted-name').first().textContent(), 'grouped.png');
    await page.locator('#clearHistory').click();
    assert.equal(await page.locator('.review-entry').count(), 0);
    await page.locator('#groupList .group-item').first().click();
    assert.equal(await page.locator('.submitted-name').first().textContent(), 'final.png');
    assert.equal(await page.locator('#workspaceCount').count(), 0);
    await page.locator('#themeButton').click();
    assert.equal(await page.locator('#themeMenu').isVisible(), true);
    assert.equal(await page.locator('#themeMenu .theme-option').count(), 3);
    assert.deepEqual(await page.locator('#themeMenu .theme-option').allTextContents(), ['跟随系统', '日间', '夜间']);
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-theme-menu.png') });
    await page.keyboard.press('ArrowRight'); assert.equal(await page.locator(':focus').getAttribute('data-theme-value'), 'light');
    await page.keyboard.press('ArrowLeft'); assert.equal(await page.locator(':focus').getAttribute('data-theme-value'), 'system');
    await page.locator('#themeMenu .theme-option[data-theme-value="dark"]').click();
    assert.equal(await page.locator('html').getAttribute('data-theme'), 'dark');
    assert.equal(await page.locator('#themeMenu').isVisible(), false);
    assert.match(await page.locator('#themeButton').getAttribute('aria-label'), /夜间/);
    assert.equal(await page.evaluate(() => localStorage.getItem('imageguard-theme')), 'dark');
    assert.equal(await page.locator('body').evaluate(n => getComputedStyle(n).backgroundColor), 'rgb(23, 27, 34)');
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-dark.png') });
    await page.locator('#themeButton').click();
    await page.locator('#themeMenu .theme-option[data-theme-value="light"]').click();
    assert.equal(await page.locator('html').getAttribute('data-theme'), 'light');
    assert.equal(await page.locator('body').evaluate(n => getComputedStyle(n).backgroundColor), 'rgb(255, 255, 255)');
    await page.locator('#themeButton').click();
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#themeMenu').isVisible(), false);
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.locator('#themeButton').click();
    await page.locator('#themeMenu .theme-option[data-theme-value="system"]').click();
    assert.equal(await page.locator('body').evaluate(n => getComputedStyle(n).backgroundColor), 'rgb(23, 27, 34)');
    await page.emulateMedia({ colorScheme: 'light' });
    await page.locator('#groupList .group-row').nth(1).locator('.group-more').click();
    await page.locator('#menuDelete').click();
    assert.equal(await page.locator('#deleteDialog').evaluate(n => n.open), true);
    assert.match(await page.locator('#deleteQuestion').textContent(), /新闻图片/);
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-delete.png') });
    await page.locator('#cancelDelete').click();
    assert.equal(await page.locator('#groupList .group-row').count(), 2);
    await page.locator('#groupList .group-row').nth(1).locator('.group-more').click();
    await page.locator('#menuDelete').click(); await page.locator('#confirmDelete').click();
    assert.equal(await page.locator('#groupList .group-row').count(), 1);
    assert.equal(await page.locator('.submitted-name').first().textContent(), 'final.png');
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-records.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('.decision-panel').first().scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-structured-mobile.png') });
    await page.locator('.evidence-group').first().scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-structured-mobile-evidence.png') });
    await page.locator('#openSidebar').click(); assert.equal(await page.locator('#sidebar').evaluate(n => n.inert), false);
    await page.locator('.history-item').first().click(); assert.equal(await page.locator('#sidebar').evaluate(n => n.inert), true);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-mobile.png') });
    await page.locator('#openSidebar').click(); await page.locator('#clearHistory').click();
    assert.equal(await page.locator('.review-entry').count(), 0);
    await page.locator('#groupList .group-more').first().click();
    await page.locator('#menuDelete').click(); await page.locator('#confirmDelete').click();
    assert.equal(await page.locator('#groupList .group-row').count(), 1);
    assert.equal(await page.locator('.review-entry').count(), 0);
    assert.equal(await page.locator('#sidebar').evaluate(n => n.inert), true);
    mode = 'success';
    await page.locator('#folderInput').setInputFiles(batchDir);
    await page.waitForFunction(() => document.querySelectorAll('.review-entry').length === 2);
    assert.match(await page.locator('#historyCount').textContent(), /2/);
    assert.equal(await page.locator('#folderButton').isDisabled(), false);
    await page.screenshot({ path: path.join(root, 'frontend', 'preview-batch.png') });
    await page.locator('#fileInput').setInputFiles({ name: 'bad.txt', mimeType: 'text/plain', buffer: Buffer.from('bad') });
    await page.locator('#localError').waitFor(); assert.equal(await page.locator('#submitButton').isDisabled(), true);
    await page.evaluate(() => localStorage.setItem('imageguard-theme', 'light'));
    await upload('refresh.png'); await page.reload(); assert.equal(await page.locator('#pendingFile').isVisible(), false);
    assert.equal(await page.locator('html').getAttribute('data-theme'), 'light');
    await page.waitForFunction(() => document.querySelector('#serviceText').textContent.includes('模拟'));
    mode = 'legacy'; await submit('legacy-response.png');
    await page.locator('.legacy-response-warning').waitFor();
    assert.match(await page.locator('.legacy-response-warning').textContent(), /旧版后端响应/);
    await page.goto('file:///' + path.join(root, 'frontend', 'index.html').replaceAll('\\', '/') + '#workspace');
    await upload('preview.png'); assert.equal(await page.locator('#submitButton').isDisabled(), true);
    assert.equal(errors.length, 0, errors.join('\n'));
    console.log('PASS: landing/workspace routing, retained state, collapsible sidebar, continuous reviews, dialogs, drag/paste input, persistent keyboard themes, timeout, mobile and validation.');
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
    fs.rmSync(batchDir, { recursive: true, force: true });
  }
})().catch(e => { console.error(e); process.exitCode = 1; });
