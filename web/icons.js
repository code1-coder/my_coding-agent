// 仅加载界面使用的 Lucide 图标；图形和许可证随项目提供，无需访问 CDN。
import createElement from './vendor/lucide/createElement.js';
import CodeXml from './vendor/lucide/icons/code-xml.js';
import FolderOpen from './vendor/lucide/icons/folder-open.js';
import LayoutDashboard from './vendor/lucide/icons/layout-dashboard.js';
import MessagesSquare from './vendor/lucide/icons/messages-square.js';
import Brain from './vendor/lucide/icons/brain.js';
import Users from './vendor/lucide/icons/users.js';
import GitBranch from './vendor/lucide/icons/git-branch.js';
import FolderCode from './vendor/lucide/icons/folder-code.js';
import ChevronRight from './vendor/lucide/icons/chevron-right.js';
import PanelLeft from './vendor/lucide/icons/panel-left.js';
import RefreshCw from './vendor/lucide/icons/refresh-cw.js';
import Sparkles from './vendor/lucide/icons/sparkles.js';
import Plus from './vendor/lucide/icons/plus.js';
import Search from './vendor/lucide/icons/search.js';
import Activity from './vendor/lucide/icons/activity.js';
import CircleCheck from './vendor/lucide/icons/circle-check.js';
import Workflow from './vendor/lucide/icons/workflow.js';
import MessageSquarePlus from './vendor/lucide/icons/message-square-plus.js';
import Info from './vendor/lucide/icons/info.js';
import X from './vendor/lucide/icons/x.js';
import Terminal from './vendor/lucide/icons/terminal.js';
import FolderTree from './vendor/lucide/icons/folder-tree.js';
import ListTodo from './vendor/lucide/icons/list-todo.js';
import ScanSearch from './vendor/lucide/icons/scan-search.js';
import ArrowUpRight from './vendor/lucide/icons/arrow-up-right.js';
import Square from './vendor/lucide/icons/square.js';
import ArrowUp from './vendor/lucide/icons/arrow-up.js';
import CircleDashed from './vendor/lucide/icons/circle-dashed.js';
import Clock3 from './vendor/lucide/icons/clock-3.js';
import LockKeyhole from './vendor/lucide/icons/lock-keyhole.js';
import ShieldCheck from './vendor/lucide/icons/shield-check.js';
import Wrench from './vendor/lucide/icons/wrench.js';
import Bot from './vendor/lucide/icons/bot.js';

const iconSet = { CodeXml, FolderOpen, LayoutDashboard, MessagesSquare, Brain, Users,
  GitBranch, FolderCode, ChevronRight, PanelLeft, RefreshCw, Sparkles, Plus, Search,
  Activity, CircleCheck, Workflow, MessageSquarePlus, Info, X, Terminal, FolderTree,
  ListTodo, ScanSearch, ArrowUpRight, Square, ArrowUp, CircleDashed, Clock3,
  LockKeyhole, ShieldCheck, Wrench, Bot };

export function renderIcons(root = document) {
  root.querySelectorAll('[data-icon]').forEach(placeholder => {
    const name = placeholder.dataset.icon.replace(/(^|-)(\w)/g, (_, dash, letter) => letter.toUpperCase());
    const definition = iconSet[name];
    if (!definition) return;
    const svg = createElement(definition, {
      class: 'icon ' + (placeholder.className || ''), 'aria-hidden': 'true', focusable: 'false',
    });
    placeholder.replaceWith(svg);
  });
}
