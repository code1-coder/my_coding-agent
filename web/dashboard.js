import { renderIcons } from './icons.js';

const $ = id => document.getElementById(id);
const statuses = { pending: '待处理', in_progress: '进行中', completed: '已完成' };
const viewNames = { board: '任务看板', chat: 'AI 对话', memory: '记忆空间', team: '智能体团队' };
const memoryTypes = { user: '用户偏好', feedback: '反馈记录', project: '项目知识', reference: '参考资料' };
const sessionId = sessionStorage.getItem('coding-agent-session') || crypto.randomUUID();
sessionStorage.setItem('coding-agent-session', sessionId);

const state = {
  tasks: [], workspace: null, filter: 'all', sending: false, controller: null,
  assistant: null, tools: new Map(), approvals: new Map(), refreshing: null,
  tasksSignature: '', workspaceSignature: '', wsConnected: false, httpReady: false,
  closing: false, ws: null, reconnect: null, poll: null,
};

// 所有动态文本先转义；模型输出、任务名称与记忆内容均不能作为 HTML 执行。
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === 'string' ? error.detail : '请求失败，请重试（' + response.status + '）');
  }
  return response.json();
}

function jsonOptions(method, body) {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
}

let toastTimer;
function toast(text) {
  $('toast').textContent = text;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3500);
}

function navigate(view) {
  if (!viewNames[view]) view = 'board';
  document.body.dataset.view = view;
  for (const name of ['board', 'memory', 'team']) $(name + '-view').hidden = name !== view;
  document.querySelectorAll('[data-nav]').forEach(button => {
    const active = button.dataset.nav === view;
    button.classList.toggle('active', active);
    if (active) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  $('breadcrumb-page').textContent = viewNames[view];
  closeAssistant();
  if (view === 'chat') {
    $('chat-nav-dot').hidden = true;
    setTab('conversation');
  }
}

function openAssistant() {
  if (document.body.dataset.view !== 'chat' && matchMedia('(max-width: 1050px)').matches) {
    document.body.classList.add('chat-open');
    $('chat-backdrop').hidden = false;
    $('workspace').inert = true;
  }
  $('assistant-toggle').setAttribute('aria-expanded', 'true');
}

function closeAssistant() {
  document.body.classList.remove('chat-open');
  $('chat-backdrop').hidden = true;
  $('workspace').inert = false;
  $('assistant-toggle').setAttribute('aria-expanded', 'false');
}

function setTab(tab, focus = false) {
  for (const name of ['conversation', 'activity']) {
    const active = name === tab;
    $(name + '-tab').setAttribute('aria-selected', String(active));
    $(name + '-tab').tabIndex = active ? 0 : -1;
    $(name + '-panel').hidden = !active;
  }
  if (focus) $(tab + '-tab').focus();
}

function draftPrompt(prompt) {
  openAssistant();
  setTab('conversation');
  $('prompt-input').value = prompt;
  $('prompt-input').focus();
}

function isBlocked(task) {
  return (task.blockedBy || []).some(id => state.tasks.find(other => other.id === id)?.status !== 'completed');
}

function taskRef(task) { return 'TASK-' + String(task.id).split('_').pop().slice(-6).toUpperCase(); }
function ownerLabel(owner) { return owner === 'manual' ? '我' : owner || '未认领'; }
function dateLabel(timestamp) {
  if (!Number.isFinite(timestamp)) return '';
  return new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit' }).format(new Date(timestamp * 1000));
}

// 任务计数来自完整快照，列内计数与搜索、认领人及依赖筛选保持一致。
function renderTasks() {
  const tasks = state.tasks.filter(task => Object.hasOwn(statuses, task.status));
  const query = $('task-search').value.trim().toLocaleLowerCase();
  const filtered = tasks.filter(task => {
    if (state.filter === 'mine' && task.owner !== 'manual') return false;
    if (state.filter === 'blocked' && !task.blockedBy?.length) return false;
    return !query || [task.subject, task.description, task.id, task.owner].join(' ').toLocaleLowerCase().includes(query);
  });
  const completed = tasks.filter(task => task.status === 'completed').length;
  const progress = tasks.length ? Math.round(completed / tasks.length * 100) : 0;
  $('nav-task-count').textContent = String(tasks.length);
  $('filter-count').textContent = String(tasks.length);
  $('overview-total').innerHTML = tasks.length + ' <small>个任务</small>';
  $('project-progress').style.width = progress + '%';
  document.querySelector('.progress-track').setAttribute('aria-valuenow', String(progress));
  $('progress-label').textContent = tasks.length ? progress + '% 已完成' : '等待开始';
  for (const status of Object.keys(statuses)) {
    const list = filtered.filter(task => task.status === status);
    $('count-' + status).textContent = String(list.length);
    const empty = {
      pending: ['circle-dashed', '等待新的任务', '把下一步计划放在这里'],
      in_progress: ['clock-3', '专注当下的工作', '认领任务后，在这里追踪进展'],
      completed: ['circle-check', '每一次完成都值得记录', '已完成的任务会归档到这里'],
    }[status];
    $('tasks-' + status).innerHTML = list.length ? list.map(task => {
      const blocked = isBlocked(task);
      const owner = ownerLabel(task.owner);
      return `<button class="task-card ${status}" data-task-id="${escapeHtml(task.id)}" aria-label="查看任务：${escapeHtml(task.subject)}">
        <div class="card-topline"><span class="task-ref">${escapeHtml(taskRef(task))}</span><span class="card-status-label">${blocked && status === 'pending' ? '等待依赖' : statuses[status]}</span></div>
        <h3>${escapeHtml(task.subject)}</h3>${task.description ? '<p>' + escapeHtml(task.description) + '</p>' : ''}
        ${task.blockedBy?.length ? `<span class="dependency-tag ${blocked ? '' : 'clear'}"><i data-icon="${blocked ? 'lock-keyhole' : 'circle-check'}"></i>${blocked ? '前置任务未完成' : '依赖已完成'}</span>` : ''}
        <div class="card-meta"><span class="card-owner"><span class="owner-avatar">${escapeHtml(owner.slice(0, 1).toUpperCase())}</span><span class="owner-name">${escapeHtml(owner)}</span></span><span class="card-date">${escapeHtml(dateLabel(task.completed_at || task.created_at))}</span></div>
      </button>`;
    }).join('') : `<div class="column-empty"><i data-icon="${empty[0]}"></i><h3>${query || state.filter !== 'all' ? '没有匹配的任务' : empty[1]}</h3><p>${query || state.filter !== 'all' ? '试试其他关键词或筛选条件' : empty[2]}</p></div>`;
  }
  $('task-list').setAttribute('aria-busy', 'false');
  renderIcons($('task-list'));
  if ($('detail-dialog').open) renderTaskDetail($('task-detail').dataset.taskId);
}

function renderMemories() {
  const memories = state.workspace?.memories || [];
  const query = $('memory-search').value.trim().toLocaleLowerCase();
  const opened = new Set([...$('memory-list').querySelectorAll('details[open]')].map(row => row.dataset.filename));
  $('memory-count').textContent = memories.length + ' 条记忆';
  const filtered = memories.filter(item => [item.name, item.description, item.body].join(' ').toLocaleLowerCase().includes(query));
  $('memory-list').innerHTML = filtered.length ? filtered.map(item => `<details class="memory-row" data-filename="${escapeHtml(item.filename)}" ${opened.has(item.filename) ? 'open' : ''}>
    <summary><span class="memory-type-icon"><i data-icon="brain"></i></span><div class="memory-title"><strong>${escapeHtml(item.name)}</strong><p>${escapeHtml(item.description)}</p></div><span class="memory-type">${escapeHtml(memoryTypes[item.type] || item.type)}</span><i data-icon="chevron-right"></i></summary>
    <div class="memory-body">${escapeHtml(item.body)}<div class="memory-file">${escapeHtml(item.filename)}</div></div></details>`).join('')
    : `<div class="simple-empty"><i data-icon="brain"></i><h3>${query ? '没有找到相关记忆' : '让协作从了解你开始'}</h3><p>${query ? '试试其他关键词。' : '在对话中告诉助手“记住我的偏好”。<br>保存成功后，记忆会显示在这里。'}</p></div>`;
  renderIcons($('memory-list'));
}

function renderWorkspace(snapshot) {
  state.workspace = snapshot;
  $('sidebar-project').textContent = snapshot.project;
  $('top-project').textContent = snapshot.project;
  $('model-name').textContent = snapshot.model;
  $('model-name').title = snapshot.model;
  $('workspace-path').textContent = snapshot.workdir;
  const totalTokens = snapshot.usage.input_tokens + snapshot.usage.output_tokens;
  $('usage-label').textContent = totalTokens ? new Intl.NumberFormat('zh-CN', { notation: 'compact' }).format(totalTokens) + ' tokens' : '上下文就绪';
  renderMemories();
  $('team-list').innerHTML = snapshot.team.map(agent => `<article class="team-row"><span class="team-avatar"><i data-icon="${agent.name === 'lead' ? 'sparkles' : 'bot'}"></i></span><div class="team-content"><h3>${escapeHtml(agent.name === 'lead' ? '主智能体' : agent.name)}</h3><p>${escapeHtml(agent.name === 'lead' ? '任务规划、代码执行与团队协调' : agent.role)}</p></div><span class="team-state ${agent.state === 'idle' ? '' : 'working'}"><span class="status-dot ${agent.state === 'idle' ? '' : 'purple'}"></span>${({ idle: '准备就绪', working: '执行中', active: '活跃' })[agent.state] || '活跃'}</span></article>`).join('');
  renderIcons($('team-list'));
}

function updateConnection() {
  const online = state.wsConnected && state.httpReady;
  $('connection-status').classList.toggle('offline', !online);
  $('connection-status').lastElementChild.textContent = online ? '已连接' : '重新连接中';
}

// 定时刷新与手动刷新共用同一请求，快照未变化时保留焦点和已展开的内容。
async function refreshWorkspace(force = false) {
  if (state.refreshing) {
    if (!force) return state.refreshing;
    await state.refreshing;
    return refreshWorkspace();
  }
  state.refreshing = (async () => {
    const results = await Promise.allSettled([
      api('/tasks'), api('/workspace?session_id=' + encodeURIComponent(sessionId)),
    ]);
    if (results[0].status === 'fulfilled') {
      const signature = JSON.stringify(results[0].value.tasks);
      if (signature !== state.tasksSignature) {
        state.tasksSignature = signature;
        state.tasks = results[0].value.tasks;
        renderTasks();
      }
    }
    if (results[1].status === 'fulfilled') {
      const signature = JSON.stringify(results[1].value);
      if (signature !== state.workspaceSignature) {
        state.workspaceSignature = signature;
        renderWorkspace(results[1].value);
      }
    }
    state.httpReady = results.every(result => result.status === 'fulfilled');
    $('data-error').hidden = state.httpReady;
    if (state.httpReady) $('sync-time').textContent = '已同步 ' + new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
    updateConnection();
  })().finally(() => { state.refreshing = null; });
  return state.refreshing;
}

function showTaskForm() {
  $('task-form').reset();
  $('task-form-error').hidden = true;
  $('task-dependency').innerHTML = '<option value="">无前置依赖</option>' + state.tasks.filter(task => task.status !== 'deleted').map(task => `<option value="${escapeHtml(task.id)}">${escapeHtml(task.subject)} · ${escapeHtml(statuses[task.status] || task.status)}</option>`).join('');
  $('task-dialog').showModal();
}

async function createTask(event) {
  event.preventDefault();
  if ($('create-task-btn').disabled) return;
  const subject = $('task-subject').value.trim();
  if (!subject) { $('task-subject').focus(); return; }
  $('create-task-btn').disabled = true;
  $('task-form-error').hidden = true;
  try {
    await api('/tasks', jsonOptions('POST', {
      subject, description: $('task-description').value,
      blockedBy: $('task-dependency').value ? [$('task-dependency').value] : [],
    }));
    $('task-dialog').close();
    // 新任务即使不满足先前筛选条件，也立即出现在全部任务中。
    $('task-search').value = '';
    setFilter('all');
    await refreshWorkspace(true);
    toast('任务已创建');
  } catch (error) {
    $('task-form-error').textContent = error.message;
    $('task-form-error').hidden = false;
  } finally { $('create-task-btn').disabled = false; }
}

function renderTaskDetail(taskId) {
  const task = state.tasks.find(item => item.id === taskId);
  if (!task) { $('detail-dialog').close(); return; }
  const blocked = isBlocked(task);
  const dependencies = (task.blockedBy || []).map(id => state.tasks.find(item => item.id === id));
  $('task-detail').dataset.taskId = taskId;
  $('task-detail').innerHTML = `<div class="modal-header"><div><div class="eyebrow">${escapeHtml(taskRef(task))}</div><h2 id="task-detail-title">${escapeHtml(task.subject)}</h2></div><button class="icon-button" data-close-dialog aria-label="关闭任务详情"><i data-icon="x"></i></button></div>
    <div class="detail-meta"><span><span class="column-dot ${escapeHtml(task.status)}"></span> ${escapeHtml(statuses[task.status] || task.status)}</span><span>负责人：${escapeHtml(ownerLabel(task.owner))}</span><span>创建于 ${escapeHtml(dateLabel(task.created_at))}</span></div>
    <div class="detail-description">${escapeHtml(task.description || '这个任务还没有补充描述。')}</div>
    ${dependencies.length ? '<div class="detail-dependencies">前置任务<ul>' + dependencies.map(dependency => '<li>' + escapeHtml(dependency ? dependency.subject + ' · ' + (statuses[dependency.status] || dependency.status) : '依赖任务不存在') + '</li>').join('') + '</ul></div>' : ''}
    ${blocked ? '<p class="form-help">请先完成前置任务，再开始处理当前任务。</p>' : ''}
    <p class="form-error" id="detail-error" role="alert" hidden></p>
    <div class="modal-footer"><button class="button secondary" id="task-to-ai"><i data-icon="sparkles"></i>交给 AI</button>${task.status === 'pending' ? '<button class="button primary" data-task-action="claim" ' + (blocked || task.owner ? 'disabled' : '') + '>认领并开始</button>' : task.status === 'in_progress' ? '<button class="button primary" data-task-action="complete"><i data-icon="circle-check"></i>完成任务</button>' : '<button class="button secondary" data-close-dialog>关闭</button>'}</div>`;
  renderIcons($('task-detail'));
}

async function taskAction(action, button) {
  const taskId = $('task-detail').dataset.taskId;
  button.disabled = true;
  try {
    await api('/tasks/' + encodeURIComponent(taskId), jsonOptions('PATCH', { action }));
    await refreshWorkspace(true);
    toast(action === 'claim' ? '已认领任务，可以开始处理' : '任务已完成');
  } catch (error) {
    $('detail-error').textContent = error.message;
    $('detail-error').hidden = false;
    button.disabled = false;
  }
}

function setFilter(filter) {
  state.filter = filter;
  document.querySelectorAll('[data-filter]').forEach(button => {
    const selected = button.dataset.filter === filter;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  renderTasks();
}

function scrollChat(force = false) {
  const panel = $('conversation-panel');
  // 用户向上查看历史时不强制拉回；发送新输入时才主动滚动到底部。
  if (force || panel.scrollHeight - panel.scrollTop - panel.clientHeight < 160) panel.scrollTop = panel.scrollHeight;
}

function appendMessage(role, text, variant = '') {
  $('chat-welcome').hidden = true;
  const node = document.createElement('div');
  node.className = 'message ' + role + (variant ? ' ' + variant : '');
  node.innerHTML = (role === 'system' ? '' : `<div class="role">${role === 'assistant' ? '<i data-icon="sparkles"></i>AI 助手' : '你'}</div>`) + '<div class="content"></div>';
  node.querySelector('.content').textContent = text;
  $('chat-area').appendChild(node);
  renderIcons(node);
  scrollChat();
  return node;
}

// 完成后以安全的 DOM 节点展示代码块；不把模型文本交给 innerHTML。
function finishAssistant(node) {
  if (!node) return;
  node.classList.add('finished');
  const content = node.querySelector('.content');
  const raw = content.textContent;
  const parts = raw.split(/```([^\n`]*)\n([\s\S]*?)```/g);
  content.replaceChildren();
  for (let index = 0; index < parts.length; index += 3) {
    content.append(document.createTextNode(parts[index]));
    if (index + 2 < parts.length) {
      const pre = document.createElement('pre');
      if (parts[index + 1].trim()) {
        const language = document.createElement('span');
        language.className = 'code-language';
        language.textContent = parts[index + 1].trim();
        pre.append(language);
      }
      const code = document.createElement('code');
      code.textContent = parts[index + 2];
      pre.append(code);
      content.append(pre);
    }
  }
  if (raw.trim()) {
    const copy = document.createElement('button');
    copy.className = 'message-copy';
    copy.textContent = '复制回复';
    copy.onclick = () => navigator.clipboard.writeText(raw).then(() => toast('回复已复制')).catch(() => toast('无法访问剪贴板，请手动选择文本复制'));
    node.append(copy);
  }
}

function setAgentState(label, working = false) {
  if ([...state.approvals.values()].some(entry => entry.state === 'pending')) {
    label = '等待你的审批';
    working = true;
  }
  $('agent-state').innerHTML = `<span class="status-dot ${working ? 'purple' : ''}"></span>` + escapeHtml(label);
}

function activityCount() {
  $('activity-count').textContent = String($('activity-list').querySelectorAll('.tool-call').length);
}

// SSE 与 WebSocket 可能同时送达；按每次执行的唯一 ID 合并状态，防止记录重复。
function updateTool(data, finished = false) {
  const key = data.execution_id || (data.agent || 'lead') + ':' + data.tool_id;
  let card = state.tools.get(key);
  if (!card) {
    $('activity-empty')?.remove();
    card = document.createElement('div');
    card.className = 'tool-call';
    card.innerHTML = `<button class="tool-summary" aria-expanded="false"><i data-icon="wrench"></i><span class="name"></span><span class="tool-state">执行中</span><i data-icon="chevron-right"></i></button><div class="details"></div>`;
    card.querySelector('.name').textContent = data.tool_name || '工具调用';
    card.querySelector('.details').textContent = JSON.stringify(data.input || {}, null, 2);
    card.querySelector('button').onclick = () => {
      const opened = card.classList.toggle('open');
      card.querySelector('button').setAttribute('aria-expanded', String(opened));
    };
    $('activity-list').append(card);
    state.tools.set(key, card);
    renderIcons(card);
    activityCount();
  }
  if (finished) {
    card.classList.add(data.is_error ? 'failed' : 'done');
    card.querySelector('.tool-state').textContent = data.is_error ? '执行失败' : '已完成';
    if (data.execution?.exit_code != null) {
      card.querySelector('.tool-state').textContent += ' · 退出码 ' + data.execution.exit_code;
    }
    card.querySelector('.details').textContent = data.execution && Object.keys(data.execution).length
      ? JSON.stringify({ ...data.execution, output: data.output }, null, 2)
      : data.output || '工具执行结束。';
  }
}

function showApproval(data) {
  if (state.approvals.has(data.approval_id)) return;
  const card = document.createElement('section');
  card.className = 'approval-card';
  card.innerHTML = `<div class="approval-title"><i data-icon="shield-check"></i><span>操作需要你的批准</span></div><p></p><pre></pre><div class="approval-note">120 秒内未处理将自动拒绝。</div><div class="approval-actions"><button class="button allow">允许</button><button class="button deny">拒绝</button></div>`;
  card.querySelector('p').textContent = data.tool_name + ' · ' + (data.reason || '请确认是否允许执行');
  card.querySelector('pre').textContent = JSON.stringify(data.input || {}, null, 2);
  const entry = { card, data, state: 'pending', choice: null };
  // 审批结束由后端通知；通知断开时，按钮也不会超过服务端的审批时限。
  entry.expiry = setTimeout(() => finishApproval(data.approval_id), 120000);
  state.approvals.set(data.approval_id, entry);
  card.querySelector('.allow').onclick = () => submitApproval(data.approval_id, true);
  card.querySelector('.deny').onclick = () => submitApproval(data.approval_id, false);
  $('approval-tray').append(card);
  $('approval-tray').hidden = false;
  $('chat-nav-dot').hidden = false;
  setAgentState('等待你的审批', true);
  renderIcons(card);
  openAssistant();
}

function finishApproval(approvalId) {
  const entry = state.approvals.get(approvalId);
  if (!entry || entry.state === 'finished') return;
  entry.state = 'finished';
  clearTimeout(entry.expiry);
  entry.card.remove();
  $('approval-tray').hidden = !$('approval-tray').children.length;
  $('activity-empty')?.remove();
  const record = document.createElement('div');
  record.className = 'tool-call open';
  record.innerHTML = '<div class="tool-summary"><i data-icon="shield-check"></i><span class="name"></span><span class="tool-state"></span></div><div class="details"></div>';
  record.querySelector('.name').textContent = entry.data.tool_name;
  record.querySelector('.tool-state').textContent = entry.choice === true ? '已允许' : entry.choice === false ? '已拒绝' : '审批已结束';
  record.querySelector('.details').textContent = JSON.stringify(entry.data.input || {}, null, 2);
  entry.record = record;
  $('activity-list').append(record);
  renderIcons(record);
  activityCount();
  setAgentState(state.sending ? '正在处理' : '准备就绪', state.sending);
}

async function submitApproval(approvalId, approved) {
  const entry = state.approvals.get(approvalId);
  if (!entry || entry.state !== 'pending') return;
  entry.card.querySelectorAll('button').forEach(button => { button.disabled = true; });
  try {
    await api('/approvals/' + encodeURIComponent(approvalId), jsonOptions('POST', { session_id: sessionId, approved }));
    entry.choice = approved;
    finishApproval(approvalId);
    if (entry.record) entry.record.querySelector('.tool-state').textContent = approved ? '已允许' : '已拒绝';
  } catch (error) {
    if (entry.state === 'pending') {
      entry.card.querySelector('.approval-note').textContent = error.message;
      entry.card.querySelectorAll('button').forEach(button => { button.disabled = false; });
    } else toast(error.message);
  }
}

function handleEvent(data) {
  const content = state.assistant?.querySelector('.content');
  if (data.type === 'text_delta' && content) {
    content.textContent += data.text || '';
    setAgentState('正在回复', true);
  } else if (data.type === 'text_reset' && content) content.textContent = data.text || '';
  else if (data.type === 'tool_start' || data.type === 'tool_use') {
    updateTool(data);
    if (state.sending) setAgentState('正在执行工具', true);
  } else if (data.type === 'tool_done') {
    updateTool(data, true);
  } else if (data.type === 'memory_saved') {
    appendMessage('system', '已将 ' + data.count + ' 条信息保存到项目记忆。', 'success');
  } else if (data.type === 'warning' || data.type === 'error') {
    appendMessage('system', data.text || data.message || '执行出现问题，请重试。', data.type === 'error' ? 'error' : '');
  } else if (data.type === 'approval_required') showApproval(data);
  else if (data.type === 'approval_finished') finishApproval(data.approval_id);
  scrollChat();
}

// 按完整 SSE 数据帧解析，正确保留跨网络分片的中文字符和未结束的帧。
async function send() {
  const prompt = $('prompt-input').value.trim();
  if (!prompt || state.sending) return;
  state.sending = true;
  state.controller = new AbortController();
  $('send-btn').disabled = true;
  $('stop-btn').hidden = false;
  $('prompt-input').value = '';
  openAssistant();
  setTab('conversation');
  appendMessage('user', prompt);
  state.assistant = appendMessage('assistant', '');
  const assistant = state.assistant;
  scrollChat(true);
  setAgentState('正在思考', true);
  let streamFailed = false;
  let receivedDone = false;
  const consume = frame => {
    const data = frame.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
    if (data) {
      const event = JSON.parse(data);
      if (event.type === 'error') streamFailed = true;
      if (event.type === 'done') receivedDone = true;
      handleEvent(event);
    }
  };
  try {
    const response = await fetch('/chat', {
      ...jsonOptions('POST', { prompt, session_id: sessionId }), signal: state.controller.signal,
    });
    if (!response.ok) throw new Error('聊天请求失败（' + response.status + '）');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() || '';
      frames.forEach(consume);
      if (done) { if (buffer.trim()) consume(buffer); break; }
    }
    if (!receivedDone && !streamFailed) throw new Error('响应提前中断，请重试或检查连接。');
  } catch (error) {
    streamFailed = true;
    appendMessage('system', error.name === 'AbortError' ? '已停止本轮生成。' : '连接出现问题：' + error.message, error.name === 'AbortError' ? '' : 'error');
  } finally {
    // 后端分别结束每项审批；主对话结束不能移除后台队友仍在等待的审批。
    finishAssistant(assistant);
    state.assistant = null;
    state.sending = false;
    state.controller = null;
    $('send-btn').disabled = false;
    $('stop-btn').hidden = true;
    setAgentState(streamFailed ? '本轮已结束' : '准备就绪');
    await refreshWorkspace(true);
    scrollChat();
  }
}

function connectWs() {
  if (state.closing) return;
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(protocol + '//' + location.host + '/ws?session_id=' + encodeURIComponent(sessionId));
  state.ws = socket;
  socket.onopen = () => { state.wsConnected = true; updateConnection(); };
  socket.onmessage = event => {
    if (event.data === 'pong') return;
    try { handleEvent(JSON.parse(event.data)); } catch { toast('收到无法识别的通知，将在下次同步时更新。'); }
  };
  socket.onclose = () => {
    state.wsConnected = false;
    updateConnection();
    if (!state.closing) state.reconnect = setTimeout(connectWs, 3000);
  };
}

// 将可见控件绑定到真实行为；事件委托也覆盖后续刷新的任务与弹窗。
document.addEventListener('click', event => {
  const target = event.target.closest('button, a');
  if (!target) return;
  if (target.dataset.nav) location.hash = target.dataset.nav;
  if (target.dataset.filter) setFilter(target.dataset.filter);
  if (target.dataset.taskId) {
    renderTaskDetail(target.dataset.taskId);
    $('detail-dialog').showModal();
  }
  if (target.dataset.prompt) draftPrompt(target.dataset.prompt);
  if (target.classList.contains('add-task') || target.id === 'new-task-btn') showTaskForm();
  if (target.hasAttribute('data-close-dialog')) target.closest('dialog').close();
  if (target.dataset.taskAction) taskAction(target.dataset.taskAction, target);
  if (target.id === 'task-to-ai') {
    const task = state.tasks.find(item => item.id === $('task-detail').dataset.taskId);
    $('detail-dialog').close();
    if (task) draftPrompt(`请处理任务 ${task.id}：${task.subject}\n${task.description || ''}\n先检查依赖与当前状态；任务尚未认领时使用 claim_task 认领，完成后用 complete_task 更新状态。`);
  }
});
$('task-form').addEventListener('submit', createTask);
$('task-search').addEventListener('input', renderTasks);
$('memory-search').addEventListener('input', renderMemories);
$('send-btn').addEventListener('click', send);
$('stop-btn').addEventListener('click', () => state.controller?.abort());
$('prompt-input').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); send(); }
});
$('assistant-toggle').onclick = () => {
  if (document.body.classList.contains('chat-open')) closeAssistant();
  else { openAssistant(); $('prompt-input').focus(); }
};
$('close-chat').onclick = () => { closeAssistant(); $('assistant-toggle').focus(); };
$('chat-backdrop').onclick = closeAssistant;
$('team-ask-btn').onclick = () => draftPrompt('请根据我的目标安排智能体协作。先确认任务范围，再按需要创建队友并分配任务。我的目标是：');
$('workspace-info').onclick = () => $('workspace-dialog').showModal();
$('refresh-btn').onclick = async () => { await refreshWorkspace(); toast(state.httpReady ? '工作空间已同步' : '同步失败，请检查服务连接'); };
for (const name of ['conversation', 'activity']) {
  $(name + '-tab').onclick = () => setTab(name);
  $(name + '-tab').addEventListener('keydown', event => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      setTab(event.key === 'Home' ? 'conversation' : event.key === 'End' ? 'activity' : name === 'conversation' ? 'activity' : 'conversation', true);
    }
  });
}
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !document.querySelector('dialog[open]')) closeAssistant();
});
window.addEventListener('hashchange', () => navigate(location.hash.slice(1)));
matchMedia('(min-width: 1051px)').addEventListener('change', event => { if (event.matches) closeAssistant(); });
window.addEventListener('pagehide', () => {
  state.closing = true;
  clearTimeout(state.reconnect);
  clearTimeout(state.poll);
  state.controller?.abort();
  state.ws?.close();
});

renderIcons();
navigate(location.hash.slice(1));
setTab('conversation');
connectWs();
async function poll() {
  await refreshWorkspace();
  if (!state.closing) state.poll = setTimeout(poll, 5000);
}
poll();
