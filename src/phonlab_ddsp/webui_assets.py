"""Self-contained browser assets for the dependency-free PhonLab web UI."""

FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect width="64" height="64" rx="16" fill="#d84c68"/>
<path d="M21 49V15h13c9 0 15 5 15 13s-6 13-15 13h-5v8zm8-16h5c4 0 7-2 7-5s-3-5-7-5h-5z" fill="white"/>
</svg>"""

GUI_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; connect-src 'self'; media-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'; form-action 'self'">
<title>PhonLab-DDSP 语音实验台</title>
<style>
:root{
  --bg:#f4f6f8;--surface:#fff;--surface-2:#f8fafb;--sidebar:#f8f9fb;
  --ink:#18232c;--muted:#687681;--line:#dfe5e9;--line-strong:#cbd4da;
  --primary:#d84c68;--primary-dark:#ad304b;--primary-soft:#fff0f3;
  --teal:#16766f;--teal-soft:#eaf7f5;--blue:#3269a8;--blue-soft:#edf4fc;
  --warning:#9a6415;--warning-soft:#fff6df;--danger:#a43838;--danger-soft:#fff0ef;
  --success:#26734e;--success-soft:#eaf7f0;--shadow:0 1px 2px rgba(15,31,42,.06),0 8px 24px rgba(31,47,58,.05);
  --radius:12px;--sidebar-width:268px
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,"Noto Sans SC","PingFang SC","Microsoft YaHei",system-ui,-apple-system,sans-serif}
button,input,select,textarea{font:inherit}
button,a,input,select,textarea{transition:border-color .15s,box-shadow .15s,background .15s,opacity .15s}
a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}
.skip{position:fixed;left:1rem;top:-4rem;z-index:100;background:var(--ink);color:#fff;padding:.6rem 1rem;border-radius:8px}
.skip:focus{top:1rem}
.app{min-height:100vh}
.sidebar{position:fixed;inset:0 auto 0 0;width:var(--sidebar-width);overflow:auto;background:var(--sidebar);border-right:1px solid var(--line);padding:1.25rem 1rem;z-index:20}
.brand{display:flex;align-items:center;gap:.7rem;padding:.3rem .45rem 1.25rem}
.brand-mark{display:grid;place-items:center;width:38px;height:38px;border-radius:11px;background:linear-gradient(145deg,var(--primary),#ef8497);color:#fff;font-weight:850;box-shadow:0 5px 14px rgba(216,76,104,.24)}
.brand strong{display:block;font-size:1.02rem}.brand small{display:block;color:var(--muted)}
.side-label{margin:1rem .55rem .35rem;color:#87939b;font-size:.72rem;font-weight:750;letter-spacing:.08em;text-transform:uppercase}
.nav{display:grid;gap:.15rem}.nav a{display:flex;align-items:center;gap:.65rem;padding:.58rem .65rem;border-radius:8px;color:#40505a;font-weight:560}
.nav a:hover,.nav a:focus{background:#edf0f3;text-decoration:none;color:var(--ink)}
.nav .active{background:var(--primary-soft);color:var(--primary-dark)}
.nav-icon{width:1.35rem;text-align:center;font-size:1.03rem}
.side-state{margin-top:1rem;padding:.75rem;border:1px solid var(--line);border-radius:10px;background:#fff}
.state-row{display:flex;align-items:flex-start;gap:.55rem}.dot{width:9px;height:9px;border-radius:50%;background:#b8c0c5;margin-top:.38rem;flex:0 0 auto}
.dot.good{background:#30a46c;box-shadow:0 0 0 3px #dff3e8}.dot.busy{background:#d89826;box-shadow:0 0 0 3px #faedcf}.dot.bad{background:#d14949;box-shadow:0 0 0 3px #f8dddd}
.side-state strong{font-size:.82rem}.side-state p{margin:.18rem 0 0;color:var(--muted);font-size:.76rem;overflow-wrap:anywhere}
.privacy{margin:1rem .55rem;color:#839099;font-size:.74rem}
.main{margin-left:var(--sidebar-width);min-width:0}
.topbar{position:sticky;top:0;z-index:12;display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.8rem clamp(1rem,3vw,2.3rem);background:rgba(255,255,255,.94);border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}
.breadcrumbs{color:var(--muted);font-size:.84rem}.breadcrumbs strong{color:var(--ink)}
.top-actions{display:flex;gap:.55rem;align-items:center}.model-badge{display:none;padding:.3rem .55rem;border-radius:999px;background:var(--teal-soft);color:var(--teal);font-size:.78rem;font-weight:750}
.content{width:min(1260px,100%);margin:0 auto;padding:2.2rem clamp(1rem,3vw,2.4rem) 5rem}
.hero{display:flex;justify-content:space-between;align-items:flex-start;gap:2rem;margin-bottom:1.5rem}
.hero h1{margin:0 0 .35rem;font-size:clamp(1.7rem,3vw,2.45rem);line-height:1.16;letter-spacing:-.025em}.hero p{max-width:760px;margin:0;color:var(--muted);font-size:1rem}
.eyebrow{color:var(--primary-dark);font-size:.76rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.45rem}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);margin-bottom:1.1rem;overflow:hidden}
.panel-head{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;padding:1.15rem 1.25rem;border-bottom:1px solid var(--line)}
.panel-head h2,.panel-head h3{margin:0;font-size:1.08rem}.panel-head p{margin:.2rem 0 0;color:var(--muted);font-size:.84rem}
.panel-body{padding:1.25rem}.panel-foot{display:flex;justify-content:flex-end;align-items:center;gap:.65rem;padding:.9rem 1.25rem;border-top:1px solid var(--line);background:var(--surface-2)}
.path-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.85rem}.span-2{grid-column:1/-1}
.workspace-browser{display:grid;grid-template-columns:auto 150px minmax(220px,1fr) auto auto;gap:.55rem;align-items:end;margin-bottom:1rem;padding:.75rem;border:1px solid var(--line);border-radius:9px;background:var(--surface-2)}.workspace-browser strong{align-self:center}.workspace-browser label{min-width:0}
label{display:flex;flex-direction:column;gap:.3rem;color:#43525c;font-size:.8rem;font-weight:650}
.hint{color:var(--muted);font-size:.75rem;font-weight:450}
input,select,textarea{width:100%;min-width:0;padding:.62rem .7rem;color:var(--ink);background:#fff;border:1px solid var(--line-strong);border-radius:8px}
input:focus,select:focus,textarea:focus{outline:0;border-color:#e27488;box-shadow:0 0 0 3px rgba(216,76,104,.12)}
input[type=range]{padding:0;border:0;accent-color:var(--primary);box-shadow:none;background:transparent}
textarea{min-height:6.5rem;resize:vertical}
.check{display:flex;flex-direction:row;align-items:flex-start;gap:.55rem;font-weight:550}.check input{width:auto;margin-top:.2rem;accent-color:var(--primary)}
.button{display:inline-flex;align-items:center;justify-content:center;gap:.42rem;border:1px solid transparent;border-radius:8px;padding:.58rem .9rem;background:var(--primary);color:#fff;font-weight:720;cursor:pointer;white-space:nowrap}
.button:hover{background:var(--primary-dark);text-decoration:none}.button:disabled{opacity:.5;cursor:not-allowed}
.button.secondary{background:#fff;border-color:var(--line-strong);color:#34434d}.button.secondary:hover{background:#f5f7f8}
.button.ghost{background:transparent;color:#56656e}.button.ghost:hover{background:#eef1f3}
.button.danger{background:#fff;border-color:#e7c1c1;color:var(--danger)}.button.danger:hover{background:var(--danger-soft)}
.button.small{padding:.36rem .58rem;font-size:.78rem}.button.wide{width:100%}
.button-row{display:flex;flex-wrap:wrap;align-items:center;gap:.55rem}
.notice{display:none;margin-top:.9rem;padding:.75rem .85rem;border-radius:8px;background:var(--blue-soft);color:#2a5888;white-space:pre-wrap;overflow-wrap:anywhere}.notice.show{display:block}.notice.error{background:var(--danger-soft);color:var(--danger)}.notice.success{background:var(--success-soft);color:var(--success)}
.section-title{margin:2.2rem 0 .85rem;scroll-margin-top:5rem}.section-title h2{margin:0 0 .2rem;font-size:1.35rem}.section-title p{margin:0;color:var(--muted)}
.steps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.65rem;margin:1rem 0 1.4rem}.step{display:flex;align-items:center;gap:.55rem;padding:.65rem .75rem;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--muted);font-size:.8rem}.step b{display:grid;place-items:center;width:1.45rem;height:1.45rem;border-radius:50%;background:#edf0f2;color:#61717a}.step.active{border-color:#edbcc6;background:var(--primary-soft);color:var(--primary-dark)}.step.active b{background:var(--primary);color:#fff}
.cap-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem}
.cap-card{border:1px solid var(--line);border-radius:10px;padding:.85rem;background:var(--surface-2)}.cap-card.enabled{border-color:#ec9aaa;background:#fff8f9}.cap-top{display:flex;justify-content:space-between;gap:.5rem;align-items:flex-start}.cap-name{font-weight:750}.cap-code{display:block;color:var(--muted);font:11px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace}.cap-description{min-height:2.45em;margin:.42rem 0;color:var(--muted);font-size:.76rem}.cap-range{display:grid;grid-template-columns:1fr 78px;gap:.5rem;align-items:center}.cap-bounds{display:flex;justify-content:space-between;color:#8a969e;font-size:.68rem}.toggle{display:flex;align-items:center;gap:.3rem;color:var(--muted);font-size:.72rem}.toggle input{width:auto;accent-color:var(--primary)}
.empty{padding:1.3rem;border:1px dashed var(--line-strong);border-radius:9px;text-align:center;color:var(--muted);background:var(--surface-2)}
.builder-layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(250px,1fr);gap:1rem}.builder-help{padding:.85rem;border-radius:9px;background:var(--teal-soft);color:#315d59;font-size:.8rem}.builder-help h4{margin:0 0 .35rem}.builder-help ul{margin:.3rem 0 0;padding-left:1.1rem}.builder-help code{font-size:.72rem}
code{padding:.08rem .27rem;border-radius:4px;background:#edf0f2;color:#31424d;font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:9px}table{width:100%;border-collapse:collapse;min-width:610px}th,td{padding:.67rem .72rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:var(--surface-2);color:#697780;font-size:.73rem;letter-spacing:.02em}tr:last-child td{border-bottom:0}.condition-name{font-weight:720}.chips{display:flex;flex-wrap:wrap;gap:.3rem}.chip{display:inline-flex;padding:.18rem .42rem;border-radius:999px;background:var(--blue-soft);color:#315e8d;font:11px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}.row-actions{display:flex;gap:.3rem}
.job-layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(300px,.85fr);gap:1rem}.confirm-box{padding:.85rem;border:1px solid #edc971;border-radius:9px;background:var(--warning-soft)}.confirm-box h4{margin:0 0 .35rem;color:#755016}.confirm-box p{margin:.2rem 0 .65rem;color:#7d642f;font-size:.78rem}.bundle{padding:.6rem;border:1px solid var(--line);border-radius:7px;background:#fff;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}
.job-status{display:flex;align-items:center;gap:.55rem;margin-bottom:.65rem}.status-pill{display:inline-flex;align-items:center;border-radius:999px;padding:.22rem .52rem;background:#edf0f2;color:#596971;font-size:.75rem;font-weight:780}.status-pill.running{background:var(--blue-soft);color:var(--blue)}.status-pill.done{background:var(--success-soft);color:var(--success)}.status-pill.failed{background:var(--danger-soft);color:var(--danger)}
pre.log,pre.json{margin:.5rem 0 0;padding:.8rem;border-radius:8px;background:#172129;color:#dbe5e9;font:11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow:auto;max-height:300px}.log-empty{color:#8b989f}
.result-summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.65rem;margin-bottom:1rem}.metric{padding:.75rem;border:1px solid var(--line);border-radius:9px;background:var(--surface-2)}.metric b{display:block;font-size:1.1rem}.metric span{color:var(--muted);font-size:.73rem}
.listen-toolbar{display:grid;grid-template-columns:minmax(180px,1fr) minmax(180px,1fr) auto;gap:.65rem;align-items:end;margin-bottom:.9rem}.nav-buttons{display:flex;gap:.35rem}
.audio-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem}.audio-card{padding:1rem;border:1px solid var(--line);border-radius:10px;background:var(--surface-2)}.audio-card.variant{border-color:#ecc1c9;background:#fff9fa}.audio-card h4{margin:0}.audio-card .file{margin:.2rem 0 .65rem;color:var(--muted);font-size:.75rem;overflow-wrap:anywhere}.audio-card audio{width:100%;height:38px}.audio-meta{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.65rem;color:var(--muted);font-size:.72rem}.meta-pill{padding:.16rem .38rem;border-radius:999px;background:#edf0f2}.meta-pill.clip{background:var(--danger-soft);color:var(--danger)}
.download-row{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:.75rem}.downloads{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7rem}.download-card{padding:.9rem;border:1px solid var(--line);border-radius:9px}.download-card h4{margin:0 0 .25rem}.download-card p{min-height:2.5em;margin:0 0 .7rem;color:var(--muted);font-size:.76rem}
.provenance-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem}.report-list{display:flex;flex-wrap:wrap;gap:.45rem}.report-link{display:inline-flex;align-items:center;padding:.42rem .62rem;border:1px solid var(--line);border-radius:7px;background:#fff}
.advanced-shell{border:1px solid var(--line);border-radius:var(--radius);background:#fff;overflow:hidden}.advanced-shell>summary{cursor:pointer;padding:1.1rem 1.25rem;font-weight:760;list-style:none}.advanced-shell>summary::-webkit-details-marker{display:none}.advanced-shell[open]>summary{border-bottom:1px solid var(--line)}.advanced-intro{padding:1rem 1.25rem 0;color:var(--muted)}
.legacy-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem;padding:1rem}.legacy-card{border:1px solid var(--line);border-radius:10px;padding:1rem;background:var(--surface-2)}.legacy-card h3{margin:0 0 .15rem;font-size:1rem}.legacy-card>p{margin:.15rem 0 .8rem;color:var(--muted);font-size:.77rem}.legacy-fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.55rem}.legacy-fields .wide{grid-column:1/-1}.legacy-card details{margin-top:.6rem}.legacy-card summary{cursor:pointer;color:var(--blue);font-size:.78rem}.legacy-result{display:none;margin-top:.7rem;padding:.65rem;border-radius:7px;background:var(--success-soft);color:#275d43;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.legacy-result.show{display:block}.legacy-result.error{background:var(--danger-soft);color:var(--danger)}
.toast{position:fixed;right:1.2rem;bottom:1.2rem;z-index:60;max-width:min(420px,calc(100vw - 2rem));padding:.75rem .9rem;border-radius:9px;background:#1d2b34;color:#fff;box-shadow:0 9px 28px rgba(0,0,0,.18);opacity:0;transform:translateY(10px);pointer-events:none}.toast.show{opacity:1;transform:none}
.sr-only{position:absolute!important;width:1px!important;height:1px!important;padding:0!important;margin:-1px!important;overflow:hidden!important;clip:rect(0,0,0,0)!important;white-space:nowrap!important;border:0!important}
@media(max-width:1050px){.cap-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.builder-layout,.job-layout{grid-template-columns:1fr}.result-summary{grid-template-columns:repeat(2,minmax(0,1fr))}.workspace-browser{grid-template-columns:1fr 1fr}.workspace-browser strong{grid-column:1/-1}}
@media(max-width:820px){:root{--sidebar-width:0px}.sidebar{position:static;width:auto;padding:.65rem 1rem;border-right:0;border-bottom:1px solid var(--line)}.brand{padding:.2rem 0 .65rem}.side-label,.side-state,.privacy{display:none}.nav{display:flex;overflow:auto;padding-bottom:.2rem}.nav a{flex:0 0 auto}.main{margin-left:0}.topbar{top:0}.steps{grid-template-columns:repeat(2,minmax(0,1fr))}.legacy-grid{grid-template-columns:1fr}}
@media(max-width:620px){.content{padding-top:1.35rem}.hero{display:block}.hero .button{margin-top:1rem}.path-grid,.cap-grid,.audio-grid,.provenance-grid,.downloads,.legacy-fields,.workspace-browser{grid-template-columns:1fr}.workspace-browser strong{grid-column:auto}.span-2,.legacy-fields .wide{grid-column:auto}.listen-toolbar{grid-template-columns:1fr}.result-summary{grid-template-columns:repeat(2,minmax(0,1fr))}.panel-head{display:block}.panel-head .button-row{margin-top:.65rem}.top-actions .button span{display:none}}
</style>
</head>
<body>
<a class="skip" href="#main-content">跳到主要内容</a>
<div class="app">
<aside class="sidebar" aria-label="主导航">
  <div class="brand"><div class="brand-mark">P</div><div><strong>PhonLab-DDSP</strong><small>语音实验台</small></div></div>
  <div class="side-label">主要工作</div>
  <nav class="nav">
    <a class="active" href="#workspace"><span class="nav-icon">⌂</span>项目与路径</a>
    <a href="#manipulation"><span class="nav-icon">⌁</span>Manipulation</a>
    <a href="#jobs"><span class="nav-icon">◷</span>Slurm 作业</a>
    <a href="#results"><span class="nav-icon">▶</span>试听与导出</a>
  </nav>
  <div class="side-label">完整流程</div>
  <nav class="nav">
    <a href="#advanced"><span class="nav-icon">☷</span>数据准备 0–7</a>
    <a href="#provenance"><span class="nav-icon">✓</span>来源与报告</a>
  </nav>
  <div class="side-state" aria-live="polite">
    <div class="state-row"><span class="dot" id="service-dot"></span><div><strong id="service-title">本机服务就绪</strong><p id="service-detail">训练与推理不会在网页进程中运行。</p></div></div>
  </div>
  <p class="privacy">路径与音频只发送到当前回环地址服务。远程使用时请建立 SSH 隧道。</p>
</aside>

<main class="main" id="main-content">
  <header class="topbar">
    <div class="breadcrumbs">工作台 / <strong id="page-location">Manipulation 实验</strong></div>
    <div class="top-actions"><span class="model-badge" id="model-badge">模型</span><button class="button secondary small" id="doctor-button" type="button"><span>环境检查</span> ⟳</button></div>
  </header>
  <div class="content">
    <section class="hero">
      <div><div class="eyebrow">No-code acoustic workflow</div><h1>听见每一个参数改变</h1><p>为语音学研究者准备的本地 WebUI：选择实验、建立命名条件、提交 GPU 后处理作业，再按语料条目比较 baseline 与 manipulation，并把 WAV 和完整来源记录一起保存。</p></div>
      <a class="button" href="#manipulation">开始建立条件 ↓</a>
    </section>

    <section id="workspace" class="section-title"><h2>项目与实验路径</h2><p>先填写一次，后续能力查询、作业和结果浏览会自动复用。</p></section>
    <div class="panel">
      <div class="panel-body">
        <div class="workspace-browser" aria-label="现有项目">
          <strong>现有项目</strong>
          <label>类型<select id="workspace-kind"><option value="experiments">实验</option><option value="checkpoints">Checkpoint</option><option value="results">结果目录</option><option value="datasets">数据集</option></select></label>
          <label>仓库内已发现路径<select id="workspace-entry"><option value="">点击“刷新”扫描</option></select></label>
          <button class="button secondary" id="workspace-use" type="button" disabled>填入路径</button>
          <button class="button secondary" id="workspace-refresh" type="button">刷新</button>
        </div>
        <div class="path-grid">
          <label>项目目录 <span class="hint">用于辨认当前工作，不会被自动改写</span><input id="project-path" data-persist="project" placeholder="/path/to/your-project"></label>
          <label>实验目录 <span class="hint">包含 experiment.json</span><input id="experiment-path" list="available-experiments" data-persist="experiment" placeholder="/path/to/experiment"></label>
          <label>Checkpoint <span class="hint">例如 runs/checkpoints/last.ckpt</span><input id="checkpoint-path" list="available-checkpoints" data-persist="checkpoint" placeholder="/path/to/last.ckpt"></label>
          <label>新结果目录 <span class="hint">必须是尚不存在的输出目录</span><input id="output-path" data-persist="output" placeholder="/path/to/control-postprocess"></label>
        </div>
        <datalist id="available-experiments"></datalist><datalist id="available-checkpoints"></datalist><datalist id="available-results"></datalist><datalist id="available-datasets"></datalist>
        <div class="button-row" style="margin-top:.85rem"><button class="button" id="load-capabilities" type="button">读取实验与模型能力</button><span class="hint" id="path-hint">不会上传文件；这里只把路径交给本机服务。</span></div>
        <div class="notice" id="workspace-notice" role="status"></div>
      </div>
    </div>

    <section id="manipulation" class="section-title"><h2>Manipulation 条件</h2><p>每个 condition 是一组有名字的参数变化；baseline 保持 checkpoint 原始输出。</p></section>
    <div class="steps" aria-label="后处理流程">
      <div class="step active" id="step-capabilities"><b>1</b>读取模型能力</div><div class="step" id="step-conditions"><b>2</b>建立条件</div><div class="step" id="step-job"><b>3</b>生成并提交</div><div class="step" id="step-listen"><b>4</b>试听和保存</div>
    </div>

    <div class="panel">
      <div class="panel-head"><div><h3>模型支持的参数</h3><p>滑杆由 API 返回的模型 capability 动态生成；不支持的参数不会出现。</p></div><div class="button-row"><span class="model-badge" id="cap-model-badge">尚未读取模型</span><button class="button secondary small" id="reload-capabilities" type="button">重新读取</button></div></div>
      <div class="panel-body">
        <div id="capability-empty" class="empty">请先在上方填写实验目录并点击“读取实验与模型能力”。</div>
        <div id="capability-grid" class="cap-grid" aria-live="polite"></div>
        <p class="hint" style="margin:.8rem 0 0">控制全集：<code>pitch_semitones</code>（F0）、<code>output_gain_db</code>（输出电平）、<code>noise_gain_db</code>（噪声）、<code>glottal_rd_scale</code>（R<sub>d</sub>）、<code>f1_scale</code>（F1）、<code>f2_scale</code>（F2）、<code>tilt_alpha_delta</code>（谱倾斜）。实际可用项以当前实验模型声明为准。</p>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head"><div><h3>命名 condition builder</h3><p>启用一个或多个参数，给它一个易懂且可复现的名字。</p></div><button class="button secondary small" id="reset-builder" type="button">清空滑杆</button></div>
      <div class="panel-body">
        <div class="builder-layout">
          <div>
            <label>Condition 名称 <span class="hint">英文字母或数字开头，可含下划线和连字符</span><input id="condition-name" maxlength="255" placeholder="例如 less_noise 或 vowel_up"></label>
            <div id="builder-controls" class="cap-grid" style="margin-top:.8rem"></div>
            <div class="button-row" style="margin-top:.85rem"><button class="button" id="add-condition" type="button">加入条件表</button><button class="button secondary" id="cancel-edit" type="button" hidden>取消编辑</button></div>
            <div class="notice" id="builder-notice" role="status"></div>
          </div>
          <aside class="builder-help"><h4>建议一次只改变一个问题</h4><ul><li>F0：<code>pitch_semitones=-4</code></li><li>输出响度：<code>output_gain_db=-6</code></li><li>噪声分支：<code>noise_gain_db=6</code></li><li>GOLF 声门：<code>glottal_rd_scale=1.2</code></li><li>ARIA-GOLF 共振峰：<code>f1_scale=1.1</code>、<code>f2_scale=.95</code></li><li>组合条件也可保存，但解释时应明确参数共同变化。</li></ul></aside>
        </div>
      </div>
      <div class="panel-body" style="padding-top:0">
        <div class="table-wrap"><table><thead><tr><th style="width:22%">Condition</th><th>参数和值</th><th style="width:130px">操作</th></tr></thead><tbody id="condition-rows"><tr id="condition-placeholder"><td colspan="3" class="empty">尚无条件。先读取 capability，再用滑杆加入第一个条件。</td></tr></tbody></table></div>
      </div>
    </div>

    <section id="jobs" class="section-title"><h2>生成与提交 Slurm 作业</h2><p>生成作业包不会启动计算；只有勾选确认并再次点击，才会调用 <code>sbatch</code>。</p></section>
    <div class="panel">
      <div class="panel-body">
        <div class="job-layout">
          <div>
            <h3 style="margin-top:0">1. 生成 postprocess 作业</h3>
            <div class="path-grid">
              <label>Partition<input id="slurm-partition" value="gpu-short"></label><label>GPU GRES<input id="slurm-gres" value="gpu:l4:1"></label>
              <label>时间上限<input id="slurm-time" value="00:30:00"></label><label>CPU<input id="slurm-cpus" type="number" min="1" value="4"></label>
              <label>内存<input id="slurm-memory" value="24G"></label><label>排除节点（可选）<input id="slurm-exclude" placeholder="node857"></label>
            </div>
            <div class="button-row" style="margin-top:.85rem"><button class="button" id="create-job" type="button">生成 postprocess 作业</button><span class="hint"><span id="condition-count">0</span> 个 manipulation 条件</span></div>
            <div class="notice" id="create-job-notice" role="status"></div>
          </div>
          <div class="confirm-box">
            <h4>2. 明确确认后提交</h4><p>请先核对作业包、checkpoint、输出目录和条件。GPU 资源可能产生排队与计算成本。</p>
            <div class="bundle" id="job-bundle">尚未生成作业包</div>
            <label class="check" style="margin:.7rem 0"><input id="submit-confirm" type="checkbox">我已核对以上路径和参数，确认向 Slurm 提交</label>
            <button class="button wide" id="submit-job" type="button" disabled>确认并提交作业</button>
          </div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head"><div><h3>作业状态与日志</h3><p>提交后每 5 秒自动刷新；完成、失败或取消时停止轮询。</p></div><div class="button-row"><label class="check"><input id="auto-poll" type="checkbox" checked>自动刷新</label><button class="button secondary small" id="refresh-job" type="button">立即刷新</button></div></div>
      <div class="panel-body">
        <div class="job-status"><span class="status-pill" id="job-state">尚未提交</span><strong id="job-id">Job ID —</strong><span class="hint" id="job-elapsed"></span></div>
        <pre class="log" id="job-log"><span class="log-empty">作业日志会显示在这里。</span></pre>
        <div class="notice" id="job-notice" role="status"></div>
      </div>
    </div>

    <section id="results" class="section-title"><h2>结果试听与 WAV 保存</h2><p>加载已完成的结果目录，逐条比较 baseline 和一个 manipulation condition。</p></section>
    <div class="panel">
      <div class="panel-body">
        <div class="path-grid"><label class="span-2">结果目录 <span class="hint">通常与上方“新结果目录”相同，也可从“现有项目”选择</span><input id="results-path" list="available-results" data-persist="results" placeholder="/path/to/control-postprocess"></label></div>
        <div class="button-row" style="margin-top:.8rem"><button class="button" id="load-results" type="button">加载结果目录</button><span class="hint" id="results-loaded">尚未加载</span></div><div class="notice" id="results-notice" role="status"></div>
      </div>
    </div>

    <div class="panel" id="results-browser" hidden>
      <div class="panel-body">
        <div class="result-summary"><div class="metric"><b id="metric-items">0</b><span>语料条目</span></div><div class="metric"><b id="metric-conditions">0</b><span>manipulation 条件</span></div><div class="metric"><b id="metric-files">0</b><span>可用 WAV</span></div><div class="metric"><b id="metric-clipping">—</b><span>clipped samples / files</span></div></div>
        <div class="listen-toolbar">
          <label>当前 condition<select id="listen-condition"></select></label><label>当前语料条目<select id="listen-item"></select></label><div class="nav-buttons"><button class="button secondary" id="previous-item" type="button" title="上一条">←</button><button class="button secondary" id="next-item" type="button" title="下一条">→</button></div>
        </div>
        <div class="audio-grid">
          <article class="audio-card"><h4>Baseline</h4><div class="file" id="baseline-file">—</div><audio id="baseline-audio" controls preload="metadata"></audio><div class="audio-meta" id="baseline-meta"></div><div class="download-row"><a class="button secondary small" id="baseline-download" href="#" download>下载 baseline WAV</a></div></article>
          <article class="audio-card variant"><h4 id="variant-heading">Manipulation</h4><div class="file" id="variant-file">—</div><audio id="variant-audio" controls preload="metadata"></audio><div class="audio-meta" id="variant-meta"></div><div class="download-row"><a class="button small" id="wav-download" href="#" download>浏览器下载当前 WAV</a><button class="button secondary small" id="export-wav" type="button">服务端另存</button></div></article>
        </div>
      </div>
    </div>

    <div class="panel" id="download-panel" hidden>
      <div class="panel-head"><div><h3>批量保存</h3><p>导出会由服务端校验结果路径；ZIP 适合传给同事或附在研究记录中。</p></div></div>
      <div class="panel-body">
        <label style="margin-bottom:.85rem">服务端另存目录 <span class="hint">必须是当前仓库内尚不存在的新路径；“下载当前 WAV”浏览器按钮不使用这里</span><input id="export-destination" data-persist="export-destination" placeholder="/path/inside/repository/new-export"></label>
        <div class="downloads">
        <div class="download-card"><h4>服务端另存单 WAV</h4><p>把当前 item 的当前 condition 复制到“另存目录”，并保留可追踪文件名。</p><button class="button secondary wide" id="export-current" type="button">另存当前单 WAV</button></div>
        <div class="download-card"><h4>服务端另存 condition</h4><p>把当前 condition 的全部 WAV 另存到新目录，便于批量声学分析。</p><button class="button secondary wide" id="export-condition" type="button">另存当前条件</button></div>
        <div class="download-card"><h4>生成下载 ZIP</h4><p>打包当前 condition、manipulation 元数据和来源记录。</p><button class="button wide" id="create-zip" type="button">生成下载 ZIP</button></div>
      </div><div class="notice" id="download-notice" role="status"></div></div>
    </div>

    <section id="provenance" class="section-title"><h2>削波、来源与报告</h2><p>重要质量信息与试听放在同一页，不把 provenance 藏在日志目录里。</p></section>
    <div class="panel">
      <div class="panel-body"><div class="provenance-grid"><div><h3 style="margin-top:0">Clipping 检查</h3><pre class="json" id="clipping-json">尚未加载结果。</pre></div><div><h3 style="margin-top:0">Provenance</h3><pre class="json" id="provenance-json">尚未加载结果。</pre></div></div><h3>可视化报告</h3><div class="report-list" id="report-links"><span class="hint">加载结果后显示 manipulation、reconstruction 和 loss/metrics 报告。</span></div></div>
    </div>

    <section id="advanced" class="section-title"><h2>高级工作流：数据准备 0–7</h2><p>保留原有下载、切分、参数提取、训练与报告入口；首次使用建议按编号顺序执行。</p></section>
    <details class="advanced-shell">
      <summary>展开完整 0–7 数据准备与训练工具</summary>
      <p class="advanced-intro">这些表单继续调用与 CLI 相同的 Python 核心。目录应位于当前项目内；训练与推理仍需生成并显式提交 Slurm 作业。</p>
      <div class="legacy-grid">
        <section class="legacy-card"><h3>0 · 获取可复现示例语料</h3><p>固定 CMU ARCTIC 30–60 分钟子集。</p><form data-action="corpus"><div class="legacy-fields"><label class="wide">新输出目录<input name="output" required placeholder="/path/to/cmu-arctic-slt"></label><label class="wide">已有官方压缩包（可选）<input name="archive"></label><label>目标分钟<input name="target_minutes" type="number" step=".5" value="30"></label><label>最长分钟<input name="max_minutes" type="number" step=".5" value="60"></label><label>话语间静音秒<input name="silence_gap" type="number" step=".05" value=".35"></label></div><button class="button small" type="submit" style="margin-top:.7rem">获取并校验语料</button><div class="legacy-result"></div></form></section>
        <section class="legacy-card"><h3>1 · 音频切分</h3><p>静音边界或固定窗口，输出独立 WAV。</p><form data-action="split"><div class="legacy-fields"><label class="wide">源音频文件或目录<input name="source" required></label><label class="wide">新输出目录<input name="output" required></label><label>切分模式<select name="mode"><option value="silence">静音边界</option><option value="fixed">固定时长</option></select></label><label>输出采样率<input name="sample_rate" type="number" placeholder="16000"></label><label>片段秒数<input name="segment_seconds" type="number" step=".05" value="2"></label><label>重叠秒数<input name="overlap_seconds" type="number" step=".05" value="0"></label></div><details><summary>静音阈值与边界参数</summary><div class="legacy-fields"><label>静音阈值 dBFS<input name="silence_threshold_db" type="number" value="-40"></label><label>最短静音秒<input name="min_silence_seconds" type="number" step=".05" value=".30"></label><label>边界留白秒<input name="padding_seconds" type="number" step=".01" value=".05"></label><label>最短片段秒<input name="min_duration_seconds" type="number" step=".05" value=".25"></label><label>最长片段秒<input name="max_duration_seconds" type="number" step=".5" value="15"></label><label>F0 帧移秒<input name="f0_hop_seconds" type="number" step=".001" value=".005"></label><label class="check"><input name="keep_tail" type="checkbox" checked>保留末尾片段</label><label class="check"><input name="split_f0_sidecars" type="checkbox">同步切分 .pv F0</label></div></details><button class="button small" type="submit" style="margin-top:.7rem">开始切分</button><div class="legacy-result"></div></form></section>
        <section class="legacy-card"><h3>2 · 准备数据集</h3><p>统一 WAV、提取 F0、确定划分和数据指纹。</p><form data-action="prepare"><div class="legacy-fields"><label class="wide">录音目录<input name="source" required></label><label class="wide">新数据集目录<input name="output" required></label><label>F0 方法<select name="f0_method"><option value="autocorr">自动相关（易安装）</option><option value="sidecar">同名 .pv</option><option value="auto">自动选择</option><option value="pyworld">pyworld</option></select></label><label>采样率<input name="sample_rate" type="number" value="16000"></label><label>F0 下限 Hz<input name="f0_floor" type="number" value="60"></label><label>F0 上限 Hz<input name="f0_ceiling" type="number" value="500"></label><label>验证比例<input name="validation_ratio" type="number" step=".01" value=".1"></label><label>测试比例<input name="test_ratio" type="number" step=".01" value=".1"></label><label>随机种子<input name="seed" type="number" value="42"></label><label>最短录音秒<input name="min_duration" type="number" step=".05" value=".25"></label><label>峰值归一化<input name="normalize_peak" type="number" step=".05" placeholder="默认关闭"></label></div><button class="button small" type="submit" style="margin-top:.7rem">准备数据</button><div class="legacy-result"></div></form></section>
        <section class="legacy-card"><h3>3 · 质检与试听</h3><p>生成时长、F0、削波、来源和逐条试听报告。</p><form data-action="inspect"><div class="legacy-fields"><label class="wide">已准备的数据集<input name="dataset" required></label><label class="wide">报告路径（可留空）<input name="output" placeholder="dataset/report.html"></label></div><button class="button small" type="submit" style="margin-top:.7rem">生成报告</button><div class="legacy-result"></div></form></section>
        <section class="legacy-card"><h3>4 · 建立训练实验</h3><p>生成配置、provenance 和 Slurm 启动器。</p><form data-action="experiment"><div class="legacy-fields"><label class="wide">已准备的数据集<input name="dataset" required></label><label class="wide">新实验目录<input name="output" required></label><label>模型<select name="model"><option value="golf">GOLF</option><option value="ddsp">DDSP</option><option value="aria-golf">ARIA-GOLF</option></select></label><label>Batch size<input name="batch_size" type="number" value="32"></label><label>训练步数<input name="max_steps" type="number" value="40000"></label><label>Workers<input name="workers" type="number" value="4"></label><label>F0 下限 Hz<input name="f0_min" type="number" value="60"></label><label>F0 上限 Hz<input name="f0_max" type="number" value="500"></label><label>随机种子<input name="seed" type="number" value="42"></label><label>Partition<input name="partition" value="gpu-short"></label><label>GPU GRES<input name="gres" value="gpu:l4:1"></label><label>时间上限<input name="time_limit" value="04:00:00"></label><label>CPU<input name="cpus" type="number" value="8"></label><label>内存<input name="memory" value="32G"></label><label>排除节点<input name="exclude" value="node857"></label></div><button class="button small" type="submit" style="margin-top:.7rem">生成实验</button><div class="legacy-result"></div></form></section>
        <section class="legacy-card"><h3>5 · Loss 与训练指标</h3><p>绘制 train/validation loss、学习率及 NaN/Inf。</p><form data-action="metrics"><div class="legacy-fields"><label class="wide">实验目录<input name="experiment" required></label><label class="wide">报告路径（可留空）<input name="output"></label><label>日志版本<input name="version" placeholder="latest"></label></div><button class="button small" type="submit" style="margin-top:.7rem">生成指标报告</button><div class="legacy-result"></div></form></section>
        <section class="legacy-card"><h3>6 · Slurm 作业中心</h3><p>显式确认提交；查询状态和末尾日志。</p><form data-action="job-submit"><div class="legacy-fields"><label class="wide">训练实验或作业包目录<input name="experiment" required></label><label class="check wide"><input name="confirm" type="checkbox">确认向 Slurm 提交该作业</label></div><button class="button small" type="submit" style="margin-top:.7rem">提交作业</button><div class="legacy-result"></div></form><details><summary>查询状态与末尾 200 行日志</summary><form data-action="job-status"><div class="legacy-fields"><label>Job ID<input name="job_id" required></label><label>实验/作业包目录<input name="experiment"></label></div><button class="button secondary small" type="submit" style="margin-top:.7rem">刷新状态</button><div class="legacy-result"></div></form></details><details><summary>取消作业</summary><form data-action="job-cancel"><div class="legacy-fields"><label>Job ID<input name="job_id" required></label><label>输入 CANCEL 确认<input name="confirmation" required></label></div><button class="button danger small" type="submit" style="margin-top:.7rem">取消作业</button><div class="legacy-result"></div></form></details></section>
        <section class="legacy-card"><h3>7 · 推理与 Manipulation</h3><p>新的可视化 builder 位于本页上方，仍可用旧格式建立作业。</p><form data-action="control-list"><div class="legacy-fields"><label class="wide">实验目录<input name="experiment" required></label></div><button class="button secondary small" type="submit" style="margin-top:.7rem">查看实验模型声明的参数</button><div class="legacy-result"></div></form><details><summary>旧格式：每行 condition:参数=值</summary><form data-action="postprocess"><div class="legacy-fields"><label class="wide">实验目录<input name="experiment" required></label><label class="wide">Checkpoint<input name="checkpoint" required></label><label class="wide">新输出目录<input name="output" required></label><label>半音偏移<input name="semitones" value="-4, 4"></label><label class="wide">其他参数条件<textarea name="variants" placeholder="less_noise:noise_gain_db=-6"></textarea></label><label>Partition<input name="partition" value="gpu-short"></label><label>GRES<input name="gres" value="gpu:l4:1"></label><label>时间<input name="time_limit" value="00:30:00"></label><label>CPU<input name="cpus" type="number" value="4"></label><label>内存<input name="memory" value="24G"></label><label>排除节点<input name="exclude"></label></div><button class="button small" type="submit" style="margin-top:.7rem">生成后处理作业</button><div class="legacy-result"></div></form></details></section>
      </div>
    </details>
  </div>
</main>
</div>
<div class="toast" id="toast" role="status" aria-live="polite"></div>
<script>
"use strict";
const state={model:"",capabilities:[],conditions:[],editing:null,jobBundle:"",jobId:"",pollTimer:null,polling:false,results:null,itemIndex:0,condition:"",workspace:{datasets:[],experiments:[],checkpoints:[],results:[]}};
const $=id=>document.getElementById(id);
const CONTROL_LABELS={pitch_semitones:"F0 / 音高",output_gain_db:"输出电平",noise_gain_db:"噪声分支",glottal_rd_scale:"声门 R_d",f1_scale:"第一共振峰 F1",f2_scale:"第二共振峰 F2",tilt_alpha_delta:"谱倾斜"};
const CONTROL_SHORT={pitch_semitones:"F0",output_gain_db:"Output",noise_gain_db:"Noise",glottal_rd_scale:"R_d",f1_scale:"F1",f2_scale:"F2",tilt_alpha_delta:"Tilt"};
const TERMINAL_STATES=new Set(["COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL","DEADLINE"]);

function text(value,fallback=""){return value===undefined||value===null?fallback:String(value)}
function numberFrom(object,keys,fallback){for(const key of keys){const value=Number(object&&object[key]);if(Number.isFinite(value))return value}return fallback}
function showNotice(id,message,type="success"){const box=$(id);box.textContent=message;box.className="notice show"+(type==="error"?" error":type==="success"?" success":"")}
function hideNotice(id){const box=$(id);box.textContent="";box.className="notice"}
let toastTimer=null;
function toast(message){const box=$("toast");box.textContent=message;box.classList.add("show");clearTimeout(toastTimer);toastTimer=setTimeout(()=>box.classList.remove("show"),3500)}
function setService(mode,title,detail){$("service-dot").className="dot "+mode;$("service-title").textContent=title;$("service-detail").textContent=detail}
function setButtonBusy(button,busy,label="处理中…"){if(busy){button.dataset.oldLabel=button.textContent;button.textContent=label;button.disabled=true}else{button.textContent=button.dataset.oldLabel||button.textContent;button.disabled=false}}
async function api(action,payload={}){const response=await fetch("/api/"+action,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});let data;try{data=await response.json()}catch(error){throw new Error("服务返回了无法解析的响应（HTTP "+response.status+"）")}if(!response.ok||data.ok===false)throw new Error(data.error||("请求失败（HTTP "+response.status+"）"));return data}

function persistPaths(){document.querySelectorAll("[data-persist]").forEach(input=>{input.addEventListener("change",()=>{try{localStorage.setItem("phonlab.webui."+input.dataset.persist,input.value)}catch(error){/* private mode */}});try{const saved=localStorage.getItem("phonlab.webui."+input.dataset.persist);if(saved&&!input.value)input.value=saved}catch(error){/* private mode */}})}
function pathValue(id,name){const value=$(id).value.trim();if(!value)throw new Error("请填写"+name);return value}
function syncResultPath(){if(!$("results-path").value.trim()&&$("output-path").value.trim())$("results-path").value=$("output-path").value.trim()}
function workspacePath(entry){if(typeof entry==="string")return entry;if(!entry||typeof entry!=="object")return"";return text(entry.path||entry.relative_path||entry.value||entry.name)}
function renderWorkspaceList(){const kind=$("workspace-kind").value,entries=state.workspace[kind]||[],select=$("workspace-entry");select.textContent="";const placeholder=document.createElement("option");placeholder.value="";placeholder.textContent=entries.length?"选择一个已发现路径":"没有发现此类路径";select.appendChild(placeholder);entries.forEach(entry=>{const path=workspacePath(entry);if(!path)return;const option=document.createElement("option");option.value=path;option.textContent=path;select.appendChild(option)});$("workspace-use").disabled=!entries.length}
function renderWorkspaceDatalists(){for(const kind of ["datasets","experiments","checkpoints","results"]){const list=$("available-"+kind);list.textContent="";(state.workspace[kind]||[]).forEach(entry=>{const path=workspacePath(entry);if(!path)return;const option=document.createElement("option");option.value=path;list.appendChild(option)})}renderWorkspaceList()}
async function scanWorkspace(quiet=false){const button=$("workspace-refresh");setButtonBusy(button,true,"扫描中…");try{const data=await api("workspace-scan",{});for(const kind of ["datasets","experiments","checkpoints","results"])state.workspace[kind]=Array.isArray(data[kind])?data[kind]:[];renderWorkspaceDatalists();const total=Object.values(state.workspace).reduce((sum,items)=>sum+items.length,0);if(!quiet)toast("扫描完成：发现 "+total+" 个可用路径")}catch(error){if(!quiet)showNotice("workspace-notice","扫描现有项目失败："+error.message,"error")}finally{setButtonBusy(button,false)}}
function useWorkspaceEntry(){const kind=$("workspace-kind").value,path=$("workspace-entry").value;if(!path)return;if(kind==="experiments")$("experiment-path").value=path;else if(kind==="checkpoints")$("checkpoint-path").value=path;else if(kind==="results")$("results-path").value=path;else $("project-path").value=path;toast("已填入："+path)}

function capabilityStep(spec){const span=numberFrom(spec,["maximum","max_value"],1)-numberFrom(spec,["minimum","min_value"],0);if(spec.name==="pitch_semitones")return 1;if(spec.name.includes("gain_db"))return .5;if(span<=.6)return .01;return .05}
function localizedDescription(spec){return text(spec.description,"由当前模型声明的可控参数")}
function appendBoundLabels(container,labels){labels.forEach(label=>{const span=document.createElement("span");span.textContent=label;container.appendChild(span)})}
function controlCard(spec,mode){const card=document.createElement("div");card.className="cap-card";card.dataset.control=spec.name;const minimum=numberFrom(spec,["minimum","min_value"],0);const maximum=numberFrom(spec,["maximum","max_value"],1);const defaultValue=numberFrom(spec,["default","default_value"],0);const top=document.createElement("div");top.className="cap-top";const title=document.createElement("div");title.className="cap-name";title.textContent=CONTROL_LABELS[spec.name]||spec.name;const code=document.createElement("span");code.className="cap-code";code.textContent=spec.name;title.appendChild(code);top.appendChild(title);if(mode==="builder"){const toggle=document.createElement("label");toggle.className="toggle";const enabled=document.createElement("input");enabled.type="checkbox";enabled.className="control-enable";enabled.setAttribute("aria-label","启用 "+spec.name);toggle.append(enabled,document.createTextNode("启用"));top.appendChild(toggle)}else{const badge=document.createElement("span");badge.className="chip";badge.textContent=text(spec.unit);top.appendChild(badge)}card.appendChild(top);const description=document.createElement("div");description.className="cap-description";description.textContent=localizedDescription(spec);card.appendChild(description);if(mode==="builder"){const rangeRow=document.createElement("div");rangeRow.className="cap-range";const range=document.createElement("input");range.type="range";range.min=minimum;range.max=maximum;range.step=capabilityStep(spec);range.value=defaultValue;range.className="control-range";range.setAttribute("aria-label",spec.name+" 滑杆");const numeric=document.createElement("input");numeric.type="number";numeric.min=minimum;numeric.max=maximum;numeric.step=capabilityStep(spec);numeric.value=defaultValue;numeric.className="control-number";numeric.setAttribute("aria-label",spec.name+" 数值");rangeRow.append(range,numeric);card.appendChild(rangeRow);const bounds=document.createElement("div");bounds.className="cap-bounds";appendBoundLabels(bounds,[minimum,"默认 "+defaultValue+" "+text(spec.unit),maximum]);card.appendChild(bounds);const enabled=card.querySelector(".control-enable");function sync(source,target){target.value=source.value;enabled.checked=true;card.classList.add("enabled")}range.addEventListener("input",()=>sync(range,numeric));numeric.addEventListener("input",()=>sync(numeric,range));enabled.addEventListener("change",()=>card.classList.toggle("enabled",enabled.checked))}else{const bounds=document.createElement("div");bounds.className="cap-bounds";appendBoundLabels(bounds,["最小 "+minimum,"默认 "+defaultValue,"最大 "+maximum]);card.appendChild(bounds)}return card}
function renderCapabilities(){const summary=$("capability-grid"),builder=$("builder-controls");summary.textContent="";builder.textContent="";$("capability-empty").hidden=state.capabilities.length>0;state.capabilities.forEach(spec=>{summary.appendChild(controlCard(spec,"summary"));builder.appendChild(controlCard(spec,"builder"))});const label=state.model?state.model.toUpperCase():"尚未读取模型";for(const id of ["model-badge","cap-model-badge"]){$(id).textContent=label;$(id).style.display=state.model?"inline-flex":"none"}if(state.capabilities.length){$("step-capabilities").classList.add("active");$("step-conditions").classList.add("active")}}
async function loadCapabilities(){const experiment=pathValue("experiment-path","实验目录");const button=$("load-capabilities");setButtonBusy(button,true,"读取中…");hideNotice("workspace-notice");setService("busy","正在读取实验","检查 experiment.json 与模型 capability。");try{const data=await api("control-list",{experiment});state.model=text(data.model);state.capabilities=Array.isArray(data.controls)?data.controls:[];state.conditions=[];state.editing=null;renderCapabilities();renderConditions();showNotice("workspace-notice",data.message||("已读取 "+state.model+"，支持 "+state.capabilities.length+" 个控制参数。"));setService("good","实验已载入",state.model+" · "+state.capabilities.length+" 个可控参数");syncResultPath()}catch(error){showNotice("workspace-notice",error.message,"error");setService("bad","读取实验失败",error.message)}finally{setButtonBusy(button,false)}}

function resetBuilder(){$("condition-name").value="";document.querySelectorAll("#builder-controls .cap-card").forEach((card,index)=>{const spec=state.capabilities[index];const enabled=card.querySelector(".control-enable");if(enabled)enabled.checked=false;card.classList.remove("enabled");if(spec){const value=numberFrom(spec,["default","default_value"],0);card.querySelector(".control-range").value=value;card.querySelector(".control-number").value=value}});state.editing=null;$("add-condition").textContent="加入条件表";$("cancel-edit").hidden=true;hideNotice("builder-notice")}
function collectBuilder(){const name=$("condition-name").value.trim();if(!/^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$/.test(name))throw new Error("Condition 名称须以英文字母或数字开头，并且只含字母、数字、_ 或 -");const controls={};document.querySelectorAll("#builder-controls .cap-card").forEach(card=>{const enabled=card.querySelector(".control-enable");if(!enabled||!enabled.checked)return;const spec=state.capabilities.find(item=>item.name===card.dataset.control);const value=Number(card.querySelector(".control-number").value);const minimum=numberFrom(spec,["minimum","min_value"],-Infinity),maximum=numberFrom(spec,["maximum","max_value"],Infinity);if(!Number.isFinite(value)||value<minimum||value>maximum)throw new Error(card.dataset.control+" 必须在 "+minimum+" 到 "+maximum+" 之间");controls[card.dataset.control]=value});if(!Object.keys(controls).length)throw new Error("请至少启用一个参数");return{name,controls}}
function addCondition(){try{if(!state.capabilities.length)throw new Error("请先读取模型 capability");const condition=collectBuilder();const duplicate=state.conditions.findIndex((item,index)=>item.name===condition.name&&index!==state.editing);if(duplicate>=0)throw new Error("Condition 名称已存在，请使用不同名称");if(state.editing===null)state.conditions.push(condition);else state.conditions[state.editing]=condition;renderConditions();resetBuilder();showNotice("builder-notice","条件已加入表格。","success")}catch(error){showNotice("builder-notice",error.message,"error")}}
function editCondition(index){const condition=state.conditions[index];if(!condition)return;resetBuilder();state.editing=index;$("condition-name").value=condition.name;document.querySelectorAll("#builder-controls .cap-card").forEach(card=>{if(!Object.prototype.hasOwnProperty.call(condition.controls,card.dataset.control))return;card.querySelector(".control-enable").checked=true;card.classList.add("enabled");card.querySelector(".control-range").value=condition.controls[card.dataset.control];card.querySelector(".control-number").value=condition.controls[card.dataset.control]});$("add-condition").textContent="保存修改";$("cancel-edit").hidden=false;$("condition-name").focus()}
function removeCondition(index){state.conditions.splice(index,1);if(state.editing===index)resetBuilder();renderConditions()}
function renderConditions(){const body=$("condition-rows");body.textContent="";if(!state.conditions.length){const row=document.createElement("tr"),cell=document.createElement("td");cell.colSpan=3;cell.className="empty";cell.textContent="尚无条件。先读取 capability，再用滑杆加入第一个条件。";row.appendChild(cell);body.appendChild(row)}else state.conditions.forEach((condition,index)=>{const row=document.createElement("tr");const name=document.createElement("td");name.className="condition-name";name.textContent=condition.name;const controls=document.createElement("td"),chips=document.createElement("div");chips.className="chips";Object.entries(condition.controls).forEach(([key,value])=>{const chip=document.createElement("span");chip.className="chip";chip.textContent=(CONTROL_SHORT[key]||key)+" = "+value;chip.title=key;chips.appendChild(chip)});controls.appendChild(chips);const actions=document.createElement("td"),wrap=document.createElement("div");wrap.className="row-actions";const edit=document.createElement("button");edit.className="button secondary small";edit.type="button";edit.textContent="编辑";edit.addEventListener("click",()=>editCondition(index));const remove=document.createElement("button");remove.className="button danger small";remove.type="button";remove.textContent="删除";remove.addEventListener("click",()=>removeCondition(index));wrap.append(edit,remove);actions.appendChild(wrap);row.append(name,controls,actions);body.appendChild(row)});$("condition-count").textContent=state.conditions.length;$("step-job").classList.toggle("active",state.conditions.length>0)}
function serializeConditions(){return state.conditions.map(item=>item.name+":"+Object.entries(item.controls).map(([key,value])=>key+"="+value).join(",")).join("\n")}

async function createJob(){const button=$("create-job");hideNotice("create-job-notice");try{if(!state.conditions.length)throw new Error("请先建立至少一个 manipulation 条件");const payload={experiment:pathValue("experiment-path","实验目录"),checkpoint:pathValue("checkpoint-path","Checkpoint"),output:pathValue("output-path","新结果目录"),semitones:"",variants:serializeConditions(),partition:$("slurm-partition").value.trim(),gres:$("slurm-gres").value.trim(),time_limit:$("slurm-time").value.trim(),cpus:Number($("slurm-cpus").value),memory:$("slurm-memory").value.trim(),exclude:$("slurm-exclude").value.trim()};setButtonBusy(button,true,"正在生成…");setService("busy","正在生成作业包","只写入配置和 Slurm 脚本，尚未提交。");const data=await api("postprocess",payload);state.jobBundle=text(data.job_bundle||data.bundle);if(!state.jobBundle)throw new Error("服务未返回 job_bundle");$("job-bundle").textContent=state.jobBundle;$("submit-confirm").checked=false;$("submit-job").disabled=true;showNotice("create-job-notice",data.message||"作业包已生成；请核对后明确确认提交。");setService("good","作业包已生成","尚未提交到 Slurm");cascade(data)}catch(error){showNotice("create-job-notice",error.message,"error");setService("bad","作业生成失败",error.message)}finally{setButtonBusy(button,false)}}
function updateSubmitState(){$("submit-job").disabled=!state.jobBundle||!$("submit-confirm").checked}
async function submitJob(){const button=$("submit-job");if(!state.jobBundle||!$("submit-confirm").checked)return;setButtonBusy(button,true,"提交中…");try{const data=await api("job-submit",{experiment:state.jobBundle,confirm:true});state.jobId=text(data.job_id||(data.status&&data.status.job_id));if(!state.jobId)throw new Error("服务未返回 Slurm Job ID");$("job-id").textContent="Job ID "+state.jobId;showNotice("job-notice",data.message||("作业 "+state.jobId+" 已提交。"));setService("busy","Slurm 作业已提交","Job "+state.jobId+" 正在排队或运行");renderJobStatus(data.status||{});startPolling();await pollJob()}catch(error){showNotice("job-notice",error.message,"error");setService("bad","提交失败",error.message)}finally{setButtonBusy(button,false);updateSubmitState()}}
function stateClass(name){const upper=text(name).toUpperCase();if(upper==="COMPLETED")return"done";if(TERMINAL_STATES.has(upper))return"failed";if(upper&&upper!=="UNKNOWN")return"running";return""}
function renderJobStatus(status){const name=text(status.state||status.status,"UNKNOWN").toUpperCase();$("job-state").textContent=name;$("job-state").className="status-pill "+stateClass(name);$("job-elapsed").textContent=text(status.elapsed||status.elapsed_time||status.runtime);if(name==="COMPLETED"){setService("good","后处理已完成","现在可以加载结果目录试听。");$("step-listen").classList.add("active");syncResultPath()}else if(TERMINAL_STATES.has(name)){setService("bad","作业 "+name,"请检查下方日志。")}}
async function pollJob(){if(!state.jobId||state.polling)return;state.polling=true;try{const data=await api("job-status",{job_id:state.jobId,experiment:state.jobBundle});renderJobStatus(data.status||{});if(data.log!==undefined)$("job-log").textContent=text(data.log,"日志为空");const name=text(data.status&&data.status.state).toUpperCase();if(TERMINAL_STATES.has(name))stopPolling()}catch(error){showNotice("job-notice",error.message,"error")}finally{state.polling=false}}
function startPolling(){stopPolling();if($("auto-poll").checked)state.pollTimer=setInterval(pollJob,5000)}
function stopPolling(){if(state.pollTimer){clearInterval(state.pollTimer);state.pollTimer=null}}

function normalizeReports(raw){if(Array.isArray(raw))return raw.map((entry,index)=>typeof entry==="string"?{label:"报告 "+(index+1),url:entry}:entry);if(raw&&typeof raw==="object")return Object.entries(raw).map(([label,value])=>typeof value==="string"?{label,url:value}:{label,...value});return[]}
function normalizeConditions(data,items){let raw=Array.isArray(data.conditions)?data.conditions:Array.isArray(data.variants)?data.variants:[];let names=raw.map(entry=>typeof entry==="string"?entry:text(entry.name||entry.slug||entry.condition)).filter(Boolean);if(!names.length&&items.length){const variants=items[0].variants||items[0].conditions||{};names=Object.keys(variants)}return names}
function normalizeItems(data){const raw=Array.isArray(data.items)?data.items:Array.isArray(data.files)?data.files:[];return raw.map((item,index)=>{if(typeof item==="string")return{id:item,label:item,baseline:item,variants:{}};const audio=Array.isArray(item.audio)?item.audio:[];const baseline=item.baseline||audio.find(entry=>text(entry.condition).toLowerCase()==="baseline")||null;const rawVariants=item.variants||item.conditions||item.manipulations||audio.filter(entry=>text(entry.condition).toLowerCase()!=="baseline");const variants={};if(Array.isArray(rawVariants))rawVariants.forEach(entry=>{if(!entry)return;const condition=text(entry.condition||entry.variant||entry.slug||entry.name);if(condition)variants[condition]=entry});else if(rawVariants&&typeof rawVariants==="object")Object.assign(variants,rawVariants);return{...item,id:text(item.id||item.item_id||item.item||item.name||item.filename,index+1),label:text(item.label||item.name||item.filename||item.id||item.item_id,"条目 "+(index+1)),baseline,variants}})}
function mediaObject(value,fallback={}){if(typeof value==="string")return{file_url:value,download_url:value,name:value};return value&&typeof value==="object"?value:fallback}
function itemBaseline(item){return mediaObject(item.baseline||item.reconstruction||item.baseline_audio,{file_url:item.baseline_file_url||item.baseline_url,download_url:item.baseline_download_url,name:item.baseline_name})}
function itemVariant(item,condition){const variants=item.variants||{};return mediaObject(variants[condition]||(item.variant&&item.condition===condition?item.variant:null),{file_url:item.file_url||item.audio_url,download_url:item.download_url,name:item.filename})}
function mediaUrl(media){return text(media.file_url||media.audio_url||media.url)}
function downloadUrl(media){return text(media.download_url||media.file_url||media.audio_url||media.url)}
function fileLabel(media){return text(media.name||media.filename||media.path||media.file_url,"—")}
function clippingValue(media){return media.clipped_samples!==undefined?media.clipped_samples:media.clipping}
function renderMeta(container,media){container.textContent="";const values=[];if(media.duration_s!==undefined)values.push(Number(media.duration_s).toFixed(2)+" s");if(media.sample_rate!==undefined)values.push(media.sample_rate+" Hz");if(media.controls&&typeof media.controls==="object")Object.entries(media.controls).forEach(([name,value])=>values.push(name+"="+value));const clipping=clippingValue(media);if(clipping!==undefined)values.push("clipping: "+text(clipping));values.forEach(value=>{const span=document.createElement("span");span.className="meta-pill"+(String(value).startsWith("clipping:")&&Number(clipping)>0?" clip":"");span.textContent=value;container.appendChild(span)})}
function setAudio(audio,media){const url=mediaUrl(media);audio.pause();audio.removeAttribute("src");if(url)audio.src=url;audio.load();audio.closest(".audio-card").style.opacity=url?"1":".55"}
function renderAudio(){if(!state.results||!state.results.items.length)return;const item=state.results.items[state.itemIndex],condition=state.condition||state.results.conditions[0];state.condition=condition;const baseline=itemBaseline(item),variant=itemVariant(item,condition);$("variant-heading").textContent=condition||"Manipulation";$("baseline-file").textContent=fileLabel(baseline);$("variant-file").textContent=fileLabel(variant);setAudio($("baseline-audio"),baseline);setAudio($("variant-audio"),variant);renderMeta($("baseline-meta"),baseline);renderMeta($("variant-meta"),variant);for(const [id,media] of [["baseline-download",baseline],["wav-download",variant]]){const link=$(id),url=downloadUrl(media);link.href=url||"#";link.hidden=!url}$("listen-item").value=String(state.itemIndex);$("listen-condition").value=condition}
function renderReports(reports){const box=$("report-links");box.textContent="";if(!reports.length){const empty=document.createElement("span");empty.className="hint";empty.textContent="结果中没有可用报告链接。";box.appendChild(empty);return}reports.forEach((report,index)=>{const url=text(report.url||report.report_url||report.file_url);if(!url)return;const link=document.createElement("a");link.className="report-link";link.href=url;link.target="_blank";link.rel="noopener";link.textContent=text(report.label||report.name,"报告 "+(index+1))+" ↗";box.appendChild(link)})}
function aggregateClipping(data){const entries=[data.baseline,...(Array.isArray(data.conditions)?data.conditions:[])].filter(Boolean),summary={clipped_samples:0,samples:0,files_with_clipping:0,files_checked:0};entries.forEach(entry=>{const clipping=entry.clipping||{};summary.clipped_samples+=Number(clipping.clipped_samples)||0;summary.samples+=Number(clipping.samples)||0;summary.files_with_clipping+=Number(clipping.files_with_clipping)||0;summary.files_checked+=Number(entry.file_count)||0});summary.clipped_fraction=summary.samples?summary.clipped_samples/summary.samples:0;return summary}
function renderResults(data){const items=normalizeItems(data),conditions=normalizeConditions(data,items);state.results={raw:data,items,conditions,reports:normalizeReports(data.reports||data.report_urls)};state.itemIndex=0;state.condition=conditions[0]||"";$("results-browser").hidden=false;$("download-panel").hidden=false;$("metric-items").textContent=items.length;$("metric-conditions").textContent=conditions.length;$("metric-files").textContent=text(data.wav_count||data.file_count||(items.length*(conditions.length+1)));const supplied=data.clipping||data.clipping_summary,clipping=supplied&&Object.keys(supplied).length?supplied:aggregateClipping(data);$("metric-clipping").textContent=text(clipping.clipped_samples??clipping.clipped_files??data.clipped_samples,"0")+" / "+text(clipping.files_with_clipping,"0");$("clipping-json").textContent=JSON.stringify(clipping,null,2);$("provenance-json").textContent=JSON.stringify(data.provenance||data.metadata||{},null,2);const conditionSelect=$("listen-condition");conditionSelect.textContent="";conditions.forEach(name=>{const option=document.createElement("option");option.value=name;option.textContent=name;conditionSelect.appendChild(option)});const itemSelect=$("listen-item");itemSelect.textContent="";items.forEach((item,index)=>{const option=document.createElement("option");option.value=String(index);option.textContent=item.label;itemSelect.appendChild(option)});renderReports(state.results.reports);if(items.length&&conditions.length)renderAudio();else showNotice("results-notice","结果目录已读取，但缺少可试听的 item 或 condition。","error");$("step-listen").classList.add("active")}
async function loadResults(){const button=$("load-results");hideNotice("results-notice");try{const output=pathValue("results-path","结果目录");setButtonBusy(button,true,"加载中…");setService("busy","正在读取结果","索引 baseline、conditions 与 provenance。");const data=await api("results-load",{output});renderResults(data);$("results-loaded").textContent=(state.results.items.length+" 条 · "+state.results.conditions.length+" 个条件");showNotice("results-notice",data.message||"结果已加载，可以逐条试听。","success");setService("good","结果已加载",state.results.items.length+" 条语料可试听")}catch(error){showNotice("results-notice",error.message,"error");setService("bad","结果加载失败",error.message)}finally{setButtonBusy(button,false)}}
function selectAdjacent(delta){if(!state.results||!state.results.items.length)return;state.itemIndex=(state.itemIndex+delta+state.results.items.length)%state.results.items.length;renderAudio()}
function currentResultPayload(scope){if(!state.results)throw new Error("请先加载结果目录");const item=state.results.items[state.itemIndex];return{output:$("results-path").value.trim(),item:item.id,condition:state.condition,scope}}
function triggerDownload(url){if(!url)throw new Error("服务没有返回下载地址");const link=document.createElement("a");link.href=url;link.download="";link.rel="noopener";document.body.appendChild(link);link.click();link.remove()}
async function exportResult(scope,button){hideNotice("download-notice");setButtonBusy(button,true,"另存中…");try{const payload=currentResultPayload(scope);payload.destination=pathValue("export-destination","服务端另存目录");const data=await api("results-export",payload);const saved=data.destination||data.output||payload.destination;showNotice("download-notice",(data.message||"服务端另存完成。")+"\n"+saved,"success")}catch(error){showNotice("download-notice",error.message,"error")}finally{setButtonBusy(button,false)}}
async function createZip(){const button=$("create-zip");hideNotice("download-notice");setButtonBusy(button,true,"打包中…");try{const payload=currentResultPayload("condition");const data=await api("results-zip",payload);showNotice("download-notice",data.message||"ZIP 已生成。","success");triggerDownload(data.download_url||data.file_url)}catch(error){showNotice("download-notice",error.message,"error")}finally{setButtonBusy(button,false)}}

function formValues(form){const output={};new FormData(form).forEach((value,key)=>output[key]=value);form.querySelectorAll("input[type=checkbox]").forEach(input=>output[input.name]=input.checked);return output}
function fillLegacy(action,name,value){if(!value)return;document.querySelectorAll('form[data-action="'+action+'"] [name="'+name+'"]').forEach(input=>{if(!input.value)input.value=value})}
function cascade(data){fillLegacy("split","source",data.continuous_audio);fillLegacy("split","output",data.suggested_segments);fillLegacy("prepare","source",data.segments_audio);fillLegacy("prepare","output",data.suggested_dataset);["inspect","experiment"].forEach(action=>fillLegacy(action,"dataset",data.dataset));fillLegacy("experiment","output",data.suggested_experiment);["metrics","job-submit","postprocess","control-list"].forEach(action=>fillLegacy(action,"experiment",data.experiment));fillLegacy("postprocess","output",data.suggested_postprocess);fillLegacy("job-submit","experiment",data.job_bundle);["job-status","job-cancel"].forEach(action=>fillLegacy(action,"job_id",data.job_id));if(data.experiment&&!$("experiment-path").value)$("experiment-path").value=data.experiment;if(data.job_bundle){state.jobBundle=data.job_bundle;$("job-bundle").textContent=data.job_bundle;updateSubmitState()}}
function appendReportLink(box,data){const url=data.report_url;if(!url)return;const link=document.createElement("a");link.href=url;link.target="_blank";link.rel="noopener";link.textContent="\n打开报告 ↗";box.appendChild(link)}
document.querySelectorAll("form[data-action]").forEach(form=>form.addEventListener("submit",async event=>{event.preventDefault();const button=form.querySelector('button[type="submit"]'),box=form.querySelector(".legacy-result");setButtonBusy(button,true);box.className="legacy-result";try{const data=await api(form.dataset.action,formValues(form));cascade(data);const copy={...data};delete copy.ok;box.textContent=JSON.stringify(copy,null,2);box.className="legacy-result show";appendReportLink(box,data)}catch(error){box.textContent="错误："+error.message;box.className="legacy-result show error"}finally{setButtonBusy(button,false)}}));

async function doctor(){const button=$("doctor-button");setButtonBusy(button,true);setService("busy","正在检查环境","检查 CPU 工具、Slurm 与可选依赖。");try{const data=await api("doctor",{});const checks=Array.isArray(data.checks)?data.checks:[];const failed=checks.filter(item=>!item.ok);if(failed.length)setService("bad","环境检查有 "+failed.length+" 项提示",failed.map(item=>item.name).join(" · "));else setService("good","环境检查通过",checks.map(item=>"✓ "+item.name).join(" · "));toast(data.message||"环境检查完成")}catch(error){setService("bad","环境检查失败",error.message)}finally{setButtonBusy(button,false)}}

$("load-capabilities").addEventListener("click",loadCapabilities);$("reload-capabilities").addEventListener("click",loadCapabilities);$("reset-builder").addEventListener("click",resetBuilder);$("add-condition").addEventListener("click",addCondition);$("cancel-edit").addEventListener("click",resetBuilder);$("create-job").addEventListener("click",createJob);$("submit-confirm").addEventListener("change",updateSubmitState);$("submit-job").addEventListener("click",submitJob);$("refresh-job").addEventListener("click",pollJob);$("auto-poll").addEventListener("change",()=>$("auto-poll").checked?startPolling():stopPolling());$("load-results").addEventListener("click",loadResults);$("listen-condition").addEventListener("change",event=>{state.condition=event.target.value;renderAudio()});$("listen-item").addEventListener("change",event=>{state.itemIndex=Number(event.target.value);renderAudio()});$("previous-item").addEventListener("click",()=>selectAdjacent(-1));$("next-item").addEventListener("click",()=>selectAdjacent(1));$("export-wav").addEventListener("click",event=>exportResult("wav",event.currentTarget));$("export-current").addEventListener("click",event=>exportResult("wav",event.currentTarget));$("export-condition").addEventListener("click",event=>exportResult("condition",event.currentTarget));$("create-zip").addEventListener("click",createZip);$("doctor-button").addEventListener("click",doctor);$("output-path").addEventListener("change",syncResultPath);$("workspace-kind").addEventListener("change",renderWorkspaceList);$("workspace-entry").addEventListener("change",()=>$("workspace-use").disabled=!$("workspace-entry").value);$("workspace-use").addEventListener("click",useWorkspaceEntry);$("workspace-refresh").addEventListener("click",()=>scanWorkspace(false));
document.querySelectorAll(".nav a[href^='#']").forEach(link=>link.addEventListener("click",()=>{document.querySelectorAll(".nav a").forEach(item=>item.classList.remove("active"));link.classList.add("active");$("page-location").textContent=link.textContent.trim()}));
persistPaths();syncResultPath();renderConditions();scanWorkspace(true);
</script>
</body>
</html>
"""
