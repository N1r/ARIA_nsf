const ASSET = "assets/";
const PLOT_BG = "#ffffff";
const FONT = { family: "Inter, -apple-system, Segoe UI, sans-serif", color: "#6b6660", size: 12 };
const GRID = "#e7e1d6", AXTITLE = "#1c1b19";   // = CSS --line
const EXC = ["#2f5d8c", "#9c4f3f", "#5a7d54"]; // steel-blue accent + muted terracotta/sage (paper palette)

// every studio (main + each voice) and the source player registers a stop fn here, so starting
// one playback stops all others — no overlapping audio across the page.
const STOPPERS = new Set();
function stopOthers(keep) { STOPPERS.forEach(f => { if (f !== keep) f(); }); }
const srcAudio = new Audio();                  // one reusable element for the natural-source playback
function srcStop() { srcAudio.pause(); }
STOPPERS.add(srcStop);

// one-line phonetic gloss per control, shown under the chooser
const GLOSS = {
  pitch: "<b>F0</b> — intonation &amp; register. The natural tone shape is preserved; only pitch level scales.",
  f1: "<b>F1</b> — tongue height / vowel openness (open ↔ close). F2 is held.",
  f2: "<b>F2</b> — backness (front ↔ back): a bright front /i/ darkens toward a back /u/. F1 is held.",
  vowel: "<b>F1 and F2 together</b> — the formants walk a perceptually even (Bark) path to <i>another</i> vowel. A large manipulation (formants scaled ×0.3 to ×1.7, ~70%), well beyond the single-formant controls: /a/→/i/ on the diagonal (F1↓, F2↑); /i/→/u/ in backness (F2↓).",
  breathy: "<b>R<sub>d</sub></b> — glottal source shape: modal ↔ breathy phonation. The formants are held.",
  prominence: "<b>energy + F0 + effort</b> — stress as a joint cue (timing is fixed, so duration is excluded).",
  nasal: "<b>nasalisation</b> — A1 reduction + widened F1 bandwidth + anti-formants (×3): oral ↔ nasalised. Oral F1/F2 frequencies held."
};

let M, state = { ci: 0, step: 3 };

fetch(ASSET + "manifest.json")
  .then(r => { if (!r.ok) throw new Error("manifest " + r.status); return r.json(); })
  .then(m => { M = m; init(); })
  .catch(() => {
    const row = document.getElementById("stimRow");
    if (row) row.innerHTML = '<p class="muted">Couldn’t load the interactive demo — please reload.</p>';
  });

function init() {
  STOPPERS.add(stopSeq);
  buildPills("contPills", M.continua.map((c, i) => ({ k: i, label: c.label })),
    state.ci, k => { stopSeq(); state.ci = +k; clampStep(); render(); });
  document.getElementById("scale").addEventListener("input", e => { stopSeq(); state.step = +e.target.value; updateStep(); });
  document.getElementById("playAll").addEventListener("click", playSequence);
  document.getElementById("cmpRef").addEventListener("click", playSource);
  document.addEventListener("keydown", onKey);
  render();
  renderRecon();
}

// natural-vs-ARIA resynthesis A/B (synthesis-quality section) — assets already built (manifest.recon)
function renderRecon() {
  const host = document.getElementById("reconAB");
  if (!host || !M.recon) return;
  host.innerHTML = M.recon.map(r => `
    <figure class="recon">
      <figcaption class="recon-head">sustained ${r.ipa}</figcaption>
      <div class="recon-pair">
        <div class="recon-one"><span class="recon-tag">natural</span>
          <img src="${ASSET}spec/${r.id}_ref.png" alt="natural ${r.ipa} spectrogram" loading="lazy"/>
          <audio controls preload="none" src="${ASSET}audio/${r.id}_ref.wav"></audio></div>
        <div class="recon-one"><span class="recon-tag">ARIA resynthesis</span>
          <img src="${ASSET}spec/${r.id}_aria.png" alt="ARIA resynthesis of ${r.ipa}" loading="lazy"/>
          <audio controls preload="none" src="${ASSET}audio/${r.id}_aria.wav"></audio></div>
      </div>
    </figure>`).join("");
}

function buildPills(id, items, active, cb) {
  const el = document.getElementById(id); el.innerHTML = "";
  el.setAttribute("role", "radiogroup"); el.setAttribute("aria-label", "Phonetic control");
  items.forEach(it => {
    const b = document.createElement("button");
    b.className = "pill" + (it.k === active ? " active" : "");
    b.textContent = it.label; b.dataset.k = it.k;
    b.setAttribute("role", "radio"); b.setAttribute("aria-checked", it.k === active ? "true" : "false");
    b.onclick = () => {
      [...el.children].forEach(c => { c.classList.remove("active"); c.setAttribute("aria-checked", "false"); });
      b.classList.add("active"); b.setAttribute("aria-checked", "true"); cb(it.k);
    };
    el.appendChild(b);
  });
}

function cont() { return M.continua[state.ci]; }
function exs() { return cont().examples; }
function nsteps() { return exs()[0].steps.length; }
function clampStep() { state.step = Math.min(state.step, nsteps() - 1); }
function isPitch() { return exs()[0].steps[0].f0 != null; }

function valsOf(s) {
  if (s.f0 != null) return `F0 ${Math.round(s.f0)} Hz`;
  if (s.f1 || s.f2) return `F1 ${s.f1 ? Math.round(s.f1) : "–"} · F2 ${s.f2 ? Math.round(s.f2) : "–"} Hz`;
  return "";
}

function render() {                 // on control change: rebuild the 3 plates
  const c = cont();
  document.getElementById("contMeta").innerHTML =
    `<span>${c.left}</span><span class="mono axis">${c.axis}</span><span>${c.right}</span>`;
  document.getElementById("ctrlNote").innerHTML = GLOSS[c.id] || c.axis || "";
  const sl = document.getElementById("scale"); sl.max = nsteps() - 1; sl.value = state.step;
  const row = document.getElementById("stimRow"); row.innerHTML = "";
  exs().forEach((ex, i) => {
    const fig = document.createElement("figure"); fig.className = "plate";
    fig.innerHTML =
      `<figcaption class="plate-cap">
         <span class="plate-syl"><span class="dot" aria-hidden="true" style="color:${EXC[i]}">●</span> ${ex.syl}</span>
         <span class="plate-vals"></span>
         <button class="plate-play" title="play this syllable's 1→7 sweep">▶ sweep</button>
       </figcaption>
       <img class="plate-spec" alt="" loading="lazy" onerror="this.classList.add('img-missing')"/>
       <audio class="plate-audio" controls preload="none"></audio>`;
    fig.querySelector(".plate-play").onclick = () => playExample(i);
    row.appendChild(fig);
  });
  exs().forEach(ex => ex.steps.forEach(s => { new Image().src = ASSET + s.spec; }));  // preload sweep frames
  updateStep();
}

function updateStep() {             // on step change: swap spec/audio/values for the 3 plates
  const n = nsteps();
  document.getElementById("scaleVal").textContent = `${state.step + 1} / ${n}`;
  const sl = document.getElementById("scale");
  if (sl) sl.setAttribute("aria-valuetext", `step ${state.step + 1} of ${n}`);
  const cols = [...document.querySelectorAll("#stimRow .plate")];
  exs().forEach((ex, i) => {
    const s = ex.steps[state.step], col = cols[i]; if (!col) return;
    const img = col.querySelector(".plate-spec");
    img.classList.remove("img-missing"); img.src = ASSET + s.spec;
    img.alt = `Spectrogram of ${ex.syl} — ${cont().label}, step ${state.step + 1} of ${n}`;
    const au = col.querySelector(".plate-audio");
    au.src = ASSET + s.audio; au.setAttribute("aria-label", `Audio: ${ex.syl}, step ${state.step + 1} of ${n}`);
    col.querySelector(".plate-vals").textContent = valsOf(s);
  });
  drawCompanion();
}

// companion figure under the plates: a static A1–P0 spectrum for the nasal control,
// otherwise the live F1–F2 / F0 Plotly chart.
function drawCompanion() {
  const c = cont();
  const chart = document.getElementById("vowelChart");
  const ltas = document.getElementById("ltasFig");
  const cap = document.getElementById("chartCap");
  if (c.ltas) {
    chart.style.display = "none"; ltas.style.display = "block"; ltas.src = ASSET + c.ltas;
    cap.innerHTML = "Long-term spectrum, <b>oral → nasal</b>: the F1 peak amplitude <b>A1 drops</b> and " +
      "F1 broadens as the anti-formant deepens — the <b>A1–P0</b> nasalisation cue. Oral formant " +
      "<i>frequencies</i> are held; the dotted line marks the ~280 Hz P0 nasal-pole region (reference — " +
      "that additive pole is not synthesised, only its A1↓ correlate).";
  } else {
    ltas.style.display = "none"; chart.style.display = "block";
    (isPitch() ? drawPitch() : drawVowel());
    cap.innerHTML = "Measured trajectory of the three syllables; the current step is ringed. " + (
      c.id === "prominence" ? "Prominence deliberately <b>bundles cues</b> — F0 (shown), energy and effort move together; the one control that is not single-parameter." :
      c.id === "breathy"    ? "Phonation (R<sub>d</sub>) changes the <b>voice source</b>, not the vowel — the F1–F2 dots stay put by design; <i>listen</i> rather than look." :
      isPitch()             ? "F0 traces the continuum while the formants stay put — only pitch moves." :
                              "The formants trace a path while F0 stays put — only the vowel moves.");
  }
}

// sweep ONE syllable through its 7 steps (audio + spectrogram follow); independent per plate
let playing = false, playIdx = -1;
function sweepCol(i, done) {
  const col = [...document.querySelectorAll("#stimRow .plate")][i];
  if (!col) return done && done();
  const ex = exs()[i], au = col.querySelector(".plate-audio"), img = col.querySelector(".plate-spec");
  const sl = document.getElementById("scale"), sv = document.getElementById("scaleVal");
  col.classList.add("playing");
  let k = 0;
  const go = () => {
    if (!playing) { col.classList.remove("playing"); return; }
    if (k >= ex.steps.length) { col.classList.remove("playing"); return done && done(); }
    if (sl) sl.value = k; if (sv) sv.textContent = `${k + 1} / ${ex.steps.length}`;   // scrub the slider as it plays
    img.src = ASSET + ex.steps[k].spec; au.src = ASSET + ex.steps[k].audio;
    au.currentTime = 0; au.play().catch(() => {});
    au.onended = () => { k++; go(); };
  };
  go();
}
function playExample(i) {            // one plate's sweep: same plate → stop; other → switch
  if (playing && playIdx === i) return stopSeq();
  stopSeq(); stopOthers(stopSeq); playing = true; playIdx = i; sweepCol(i, stopSeq);
}
function playSequence() {            // all three syllables' sweeps, in order
  if (playing && playIdx === -1) return stopSeq();
  stopSeq(); stopOthers(stopSeq); playing = true; playIdx = -1;
  const btn = document.getElementById("playAll"); btn.textContent = "■ Stop"; btn.setAttribute("aria-pressed", "true");
  const chain = i => { if (!playing || i >= exs().length) return stopSeq(); sweepCol(i, () => chain(i + 1)); };
  chain(0);
}
function stopSeq() {
  playing = false; playIdx = -1;
  document.querySelectorAll("#stimRow .plate-audio").forEach(a => { a.onended = null; a.pause(); });
  document.querySelectorAll("#stimRow .plate").forEach(c => c.classList.remove("playing"));
  const btn = document.getElementById("playAll");
  if (btn) { btn.textContent = "▶ Play all"; btn.setAttribute("aria-pressed", "false"); }
  if (M) updateStep();              // resync plates to the slider position
}

function playSource() {              // natural recording of the first example's syllable (single reused element)
  stopOthers(srcStop);
  const ex = exs()[0];
  if (ex && ex.src_audio) { srcAudio.src = ASSET + ex.src_audio; srcAudio.currentTime = 0; srcAudio.play().catch(() => {}); }
}

function onKey(e) {
  if (!M) return;
  if (/input|textarea|select|audio/i.test(e.target.tagName) && e.target.type !== "range") return;
  const nc = M.continua.length, n = nsteps();
  if (e.key === "ArrowRight") state.step = Math.min(n - 1, state.step + 1);
  else if (e.key === "ArrowLeft") state.step = Math.max(0, state.step - 1);
  else if (e.key === "ArrowDown") { stopSeq(); state.ci = (state.ci + 1) % nc; clampStep(); render(); return e.preventDefault(); }
  else if (e.key === "ArrowUp") { stopSeq(); state.ci = (state.ci - 1 + nc) % nc; clampStep(); render(); return e.preventDefault(); }
  else if (e.key === "r" || e.key === "R") { playSource(); return; }
  else return;
  e.preventDefault(); stopSeq();
  document.getElementById("scale").value = state.step; updateStep();
}

const CHART = (title, xt, yt, xrev) => ({
  title: { text: title, font: { color: AXTITLE, size: 13, family: FONT.family } },
  paper_bgcolor: PLOT_BG, plot_bgcolor: PLOT_BG, font: FONT, margin: { l: 54, r: 16, t: 34, b: 44 },
  legend: { orientation: "h", y: -0.18, font: { size: 12 } },
  xaxis: { title: xt, autorange: xrev ? "reversed" : true, gridcolor: GRID, zeroline: false, dtick: xrev ? undefined : 1 },
  yaxis: { title: yt, autorange: "reversed", gridcolor: GRID, zeroline: false }
});

function drawVowel() {
  if (typeof Plotly === "undefined") return;
  const traces = [{
    x: M.vowel_space.map(v => v.f2), y: M.vowel_space.map(v => v.f1),
    text: M.vowel_space.map(v => v.ipa), mode: "markers+text", type: "scatter",
    textposition: "top center", textfont: { color: "#9a9486", size: 13 },
    marker: { size: 7, color: "#ddd6c8" }, hoverinfo: "skip", showlegend: false
  }];
  exs().forEach((ex, i) => {
    traces.push({
      x: ex.steps.map(s => s.f2), y: ex.steps.map(s => s.f1), mode: "lines+markers", type: "scatter",
      line: { color: EXC[i], width: 1.5 }, marker: { size: 4, color: EXC[i] }, name: ex.syl,
      hovertemplate: `${ex.syl}<br>F2 %{x:.0f} · F1 %{y:.0f}<extra></extra>`
    });
    const s = ex.steps[state.step];
    traces.push({
      x: [s.f2], y: [s.f1], mode: "markers", type: "scatter", showlegend: false, hoverinfo: "skip",
      marker: { size: 13, color: EXC[i], line: { color: "#fff", width: 2 } }
    });
  });
  Plotly.react("vowelChart", traces, CHART("F1–F2 vowel space · 3 syllables (current step ●)", "F2 (Hz)", "F1 (Hz)", true),
    { displayModeBar: false, responsive: true });
}

function drawPitch() {
  if (typeof Plotly === "undefined") return;
  const x = exs()[0].steps.map((_, i) => i + 1), traces = [];
  exs().forEach((ex, i) => {
    traces.push({
      x, y: ex.steps.map(s => s.f0), mode: "lines+markers", type: "scatter",
      line: { color: EXC[i], width: 2 }, marker: { size: 5, color: EXC[i] }, name: ex.syl
    });
    traces.push({
      x: [state.step + 1], y: [ex.steps[state.step].f0], mode: "markers", type: "scatter",
      showlegend: false, hoverinfo: "skip", marker: { size: 13, color: EXC[i], line: { color: "#fff", width: 2 } }
    });
  });
  const lay = CHART("F0 along the continuum · 3 syllables (current step ●)", "continuum step", "F0 (Hz)", false);
  lay.yaxis.autorange = true;
  Plotly.react("vowelChart", traces, lay, { displayModeBar: false, responsive: true });
}

/* ============================================================================
   Other voices (CSMSC / LJ): the same studio on extra 1-h speakers — simple
   scales only (F0 · F1+F2 · breathy), full sentences, no chart. Self-contained
   per voice so it never touches the main F024 studio above. Reuses valsOf/ASSET.
   ============================================================================ */
const VGLOSS = {
  pitch: "<b>F0</b> — the whole sentence's pitch scales; the natural intonation contour is preserved.",
  f12: "<b>F1 + F2</b> scaled together — a vocal-tract-length-like shift of the entire vowel space.",
  breathy: "<b>R<sub>d</sub></b> — glottal source shape: modal → breathy (R<sub>d</sub> 0.5 → 2.5)."
};

fetch(ASSET + "voices.json").then(r => r.ok ? r.json() : null).then(v => {
  const host = document.getElementById("voiceList");
  if (v && v.voices && host) v.voices.forEach(voice => mountVoice(host, voice));
}).catch(() => {});

function mountVoice(host, V) {
  const s = { ci: 0, step: 3, playing: false, idx: -1 };
  const sec = document.createElement("section"); sec.className = "voice";
  sec.innerHTML =
    `<header class="voice-h">
       <h3>${V.name} <span class="voice-tag">${V.lang}</span></h3>
       <span class="voice-meta mono">${V.sr / 1000} kHz · ARIA ${V.model}</span>
     </header>
     <div class="studio">
       <div class="chooser"><span class="field-label">Control</span>
         <div class="pills v-pills"></div><p class="ctrl-note v-note"></p></div>
       <div class="stepper">
         <div class="stepper-head"><span class="field-label">Manipulation step</span>
           <span class="mono step-val v-stepval"></span></div>
         <input class="v-scale" type="range" min="0" step="1" aria-label="manipulation step"/>
         <div class="ends v-meta"></div></div>
       <div class="studio-actions"><button class="btn ghost sm v-play">▶ Play sweeps</button></div>
     </div>
     <div class="plate-row v-plates"></div>`;
  host.appendChild(sec);

  const $ = sel => sec.querySelector(sel);
  const cont = () => V.continua[s.ci], exs = () => cont().examples, nsteps = () => exs()[0].steps.length;
  const plates = () => [...sec.querySelectorAll(".plate")];

  const pills = $(".v-pills");
  pills.setAttribute("role", "radiogroup"); pills.setAttribute("aria-label", `${V.name} control`);
  V.continua.forEach((c, i) => {
    const b = document.createElement("button");
    b.className = "pill" + (i === 0 ? " active" : ""); b.textContent = c.label;
    b.setAttribute("role", "radio"); b.setAttribute("aria-checked", i === 0 ? "true" : "false");
    b.onclick = () => { stop();
      [...pills.children].forEach(x => { x.classList.remove("active"); x.setAttribute("aria-checked", "false"); });
      b.classList.add("active"); b.setAttribute("aria-checked", "true");
      s.ci = i; s.step = Math.min(s.step, nsteps() - 1); render(); };
    pills.appendChild(b);
  });
  $(".v-scale").addEventListener("input", e => { stop(); s.step = +e.target.value; update(); });
  $(".v-play").addEventListener("click", playAll);

  function render() {
    const c = cont();
    $(".v-note").innerHTML = VGLOSS[c.id] || c.axis || "";
    $(".v-meta").innerHTML = `<span>${c.left}</span><span class="mono axis">${c.axis}</span><span>${c.right}</span>`;
    const sl = $(".v-scale"); sl.max = nsteps() - 1; sl.value = s.step;
    const row = $(".v-plates"); row.innerHTML = "";
    exs().forEach((ex, i) => {
      const fig = document.createElement("figure"); fig.className = "plate";
      fig.innerHTML =
        `<figcaption class="plate-cap">
           <span class="plate-syl">${ex.syl}</span>
           <span class="plate-vals"></span>
           <button class="plate-play" title="play this utterance's sweep">▶ sweep</button>
         </figcaption>
         <img class="plate-spec" alt="" loading="lazy" onerror="this.classList.add('img-missing')"/>
         <audio class="plate-audio" controls preload="none"></audio>`;
      fig.querySelector(".plate-play").onclick = () => playOne(i);
      row.appendChild(fig);
    });
    update();
  }
  function update() {
    const n = nsteps();
    $(".v-stepval").textContent = `${s.step + 1} / ${n}`;
    $(".v-scale").setAttribute("aria-valuetext", `step ${s.step + 1} of ${n}`);
    plates().forEach((col, i) => {
      const ex = exs()[i], st = ex.steps[s.step];
      const img = col.querySelector(".plate-spec");
      img.classList.remove("img-missing"); img.src = ASSET + st.spec;
      img.alt = `Spectrogram of ${ex.syl} — ${cont().label}, step ${s.step + 1} of ${n}`;
      const au = col.querySelector(".plate-audio");
      au.src = ASSET + st.audio; au.setAttribute("aria-label", `Audio: ${ex.syl}, step ${s.step + 1} of ${n}`);
      col.querySelector(".plate-vals").textContent = valsOf(st);
    });
  }
  function sweep(i, done) {
    const col = plates()[i]; if (!col) return done && done();
    const ex = exs()[i], au = col.querySelector(".plate-audio"), img = col.querySelector(".plate-spec");
    col.classList.add("playing"); let k = 0;
    const go = () => {
      if (!s.playing) return col.classList.remove("playing");
      if (k >= ex.steps.length) { col.classList.remove("playing"); return done && done(); }
      img.src = ASSET + ex.steps[k].spec; au.src = ASSET + ex.steps[k].audio;
      au.currentTime = 0; au.play().catch(() => {}); au.onended = () => { k++; go(); };
    };
    go();
  }
  function playOne(i) {               // same plate → stop; other → switch
    if (s.playing && s.idx === i) return stop();
    stop(); stopOthers(stop); s.playing = true; s.idx = i; sweep(i, stop);
  }
  function playAll() {
    if (s.playing && s.idx === -1) return stop();
    stop(); stopOthers(stop); s.playing = true; s.idx = -1;
    const b = $(".v-play"); b.textContent = "■ Stop"; b.setAttribute("aria-pressed", "true");
    const chain = i => { if (!s.playing || i >= exs().length) return stop(); sweep(i, () => chain(i + 1)); };
    chain(0);
  }
  function stop() {
    s.playing = false; s.idx = -1;
    sec.querySelectorAll(".plate-audio").forEach(a => { a.onended = null; a.pause(); });
    plates().forEach(c => c.classList.remove("playing"));
    const b = $(".v-play"); b.textContent = "▶ Play sweeps"; b.setAttribute("aria-pressed", "false"); update();
  }
  STOPPERS.add(stop);

  render();
}
