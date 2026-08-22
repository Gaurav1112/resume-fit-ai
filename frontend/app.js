/* ResumeFit AI — frontend.
 * No framework, no build step, no API key. Everything sensitive stays on the
 * backend; this file only renders what the backend computed. */

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

const state = {
  analysis: null,
  generation: null,
  resumeFile: null,
  jdFile: null,
  weights: {},
  weightLabels: {},
  validationView: "ats",
};

/* ------------------------------------------------------------------ theme */
const savedTheme = localStorage.getItem("rf-theme");
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
$("themeToggle").onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("rf-theme", next);
};

/* ------------------------------------------------------------------- tabs */
function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name)
  );
  document.querySelectorAll(".panel").forEach((p) =>
    p.classList.toggle("active", p.id === `tab-${name}`)
  );
  window.scrollTo({ top: 0, behavior: "smooth" });
}
document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => !t.disabled && showTab(t.dataset.tab);
});
const enableTab = (name) => {
  const t = document.querySelector(`.tab[data-tab="${name}"]`);
  if (t) t.disabled = false;
};

/* ------------------------------------------------------------------ toast */
let toastTimer;
function toast(msg, bad = false) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.toggle("bad", bad);
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), bad ? 7000 : 3200);
}

/* ---------------------------------------------------------------- overlay */
let stepTimer;
function showOverlay(title, steps) {
  $("loaderTitle").textContent = title;
  $("loaderText").textContent = "This runs several model calls — usually 30–90 seconds.";
  const list = $("loaderSteps");
  list.innerHTML = "";
  steps.forEach((s) => list.appendChild(el("li", "", esc(s))));
  $("overlay").classList.remove("hidden");

  let i = 0;
  const tick = () => {
    [...list.children].forEach((li, idx) => {
      li.className = idx < i ? "done" : idx === i ? "active" : "";
    });
    if (i < steps.length - 1) i++;
  };
  tick();
  clearInterval(stepTimer);
  stepTimer = setInterval(tick, 4200);
}
function hideOverlay() {
  clearInterval(stepTimer);
  $("overlay").classList.add("hidden");
}

/* ------------------------------------------------------------------- http */
async function api(path, options = {}) {
  const res = await fetch(path, options);
  const type = res.headers.get("content-type") || "";
  const body = type.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    const msg =
      (body && (body.message || body.detail)) ||
      (typeof body === "string" ? body : `Request failed (${res.status})`);
    throw new Error(msg);
  }
  return body;
}

/* --------------------------------------------------------------- dropzone */
function wireDrop(dropId, inputId, chipId, key) {
  const drop = $(dropId), input = $(inputId), chip = $(chipId);
  const accept = (file) => {
    state[key] = file;
    chip.innerHTML = `<span>📄 ${esc(file.name)} · ${(file.size / 1024).toFixed(0)} KB</span>`;
    const clear = el("button", "ghost small", "Remove");
    clear.onclick = () => {
      state[key] = null;
      input.value = "";
      chip.classList.add("hidden");
    };
    chip.appendChild(clear);
    chip.classList.remove("hidden");
  };
  drop.onclick = () => input.click();
  drop.onkeydown = (e) => (e.key === "Enter" || e.key === " ") && input.click();
  input.onchange = () => input.files[0] && accept(input.files[0]);
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.remove("over");
    })
  );
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) accept(f);
  });
}
wireDrop("resumeDrop", "resumeFile", "resumeChip", "resumeFile");
wireDrop("jdDrop", "jdFile", "jdChip", "jdFile");

/* ------------------------------------------------------------------ dials */
const scoreColor = (v) =>
  v >= 90 ? "var(--good)" : v >= 80 ? "var(--accent)" : v >= 70 ? "var(--warn)" : "var(--bad)";

function dial(value, label, band) {
  const v = Math.max(0, Math.min(100, Number(value) || 0));
  const r = 24, c = 2 * Math.PI * r;
  const node = el("div", "score");
  node.innerHTML = `
    <div class="dial">
      <svg width="58" height="58">
        <circle cx="29" cy="29" r="${r}" fill="none" stroke="var(--surface-2)" stroke-width="6"/>
        <circle cx="29" cy="29" r="${r}" fill="none" stroke="${scoreColor(v)}" stroke-width="6"
                stroke-linecap="round" stroke-dasharray="${c}"
                stroke-dashoffset="${c - (v / 100) * c}"/>
      </svg>
      <div class="val">${v.toFixed(0)}</div>
    </div>
    <div class="score-meta">
      <h3>${esc(label)}</h3>
      <div class="band" style="color:${scoreColor(v)}">${esc(band || "")}</div>
    </div>`;
  return node;
}

/* ---------------------------------------------------------------- weights */
async function loadConfig() {
  try {
    const cfg = await api("/api/config");
    state.weights = { ...cfg.weights };
    state.weightLabels = cfg.weight_labels;
    renderWeights();
    const health = await api("/api/health");
    $("providerPill").textContent = `${health.provider} · ${health.model}`;
    if (!health.configured) {
      toast("No API key configured — set ANTHROPIC_API_KEY in .env", true);
    }
  } catch (e) {
    $("providerPill").textContent = "backend unreachable";
  }
}
function renderWeights() {
  const grid = $("weightGrid");
  grid.innerHTML = "";
  Object.entries(state.weights).forEach(([key, value]) => {
    const label = el("label");
    label.innerHTML = `<span>${esc(state.weightLabels[key] || key)}</span>`;
    const input = el("input");
    input.type = "number";
    input.min = 0; input.max = 1; input.step = 0.05;
    input.value = value;
    input.onchange = () => (state.weights[key] = parseFloat(input.value) || 0);
    label.appendChild(input);
    grid.appendChild(label);
  });
}
$("resetWeights").onclick = async (e) => {
  e.preventDefault();
  const cfg = await api("/api/config");
  state.weights = { ...cfg.weights };
  renderWeights();
  toast("Weights reset");
};

/* ---------------------------------------------------------------- analyze */
$("analyzeBtn").onclick = async () => {
  const fd = new FormData();
  if (state.resumeFile) fd.append("resume_file", state.resumeFile);
  fd.append("resume_text", $("resumeText").value);
  if (state.jdFile) fd.append("jd_file", state.jdFile);
  fd.append("jd_text", $("jdText").value);
  fd.append("target_market", $("market").value);
  fd.append("weights", JSON.stringify(state.weights));

  if (!state.resumeFile && $("resumeText").value.trim().length < 200) {
    return toast("Upload or paste your master resume first.", true);
  }
  if (!state.jdFile && $("jdText").value.trim().length < 80) {
    return toast("Paste the job description first.", true);
  }

  showOverlay("Analyzing", [
    "Parsing your resume into a structured profile",
    "Extracting and classifying JD requirements",
    "Building the evidence index",
    "Matching requirements against evidence",
    "Adjudicating ambiguous matches",
    "Analysing gaps and choosing positioning",
  ]);
  try {
    state.analysis = await api("/api/analyze", { method: "POST", body: fd });
    renderAnalysis(state.analysis);
    enableTab("analysis");
    showTab("analysis");
    toast("Analysis complete");
  } catch (e) {
    toast(e.message, true);
  } finally {
    hideOverlay();
  }
};

/* -------------------------------------------------------- render analysis */
function renderAnalysis(a) {
  const strip = $("analysisScores");
  strip.innerHTML = "";
  strip.appendChild(dial(a.jd_match_score, "JD match", "requirement coverage"));
  strip.appendChild(dial(a.baseline_scores.total, "Baseline fit", a.baseline_scores.band));
  const p0 = a.matrix.filter((r) => r.priority === "P0");
  const p0met = p0.filter((r) => r.score >= 0.6).length;
  strip.appendChild(
    dial(p0.length ? (100 * p0met) / p0.length : 100, "Mandatory met", `${p0met} of ${p0.length}`)
  );
  strip.appendChild(
    dial(
      Math.max(0, 100 - a.gaps.filter((g) => g.risk === "HIGH").length * 20),
      "Risk profile",
      `${a.gaps.filter((g) => g.risk === "HIGH").length} high-risk gaps`
    )
  );

  // positioning
  const pos = a.positioning;
  $("positioningBadge").textContent = pos.supported ? "Supported by evidence" : "Level mismatch";
  $("positioningBadge").className = "badge " + (pos.supported ? "ok" : "warn");
  $("positioningBox").innerHTML = `
    <div class="item ${pos.supported ? "ok" : "warn"}">
      <h4>${esc(pos.target_title)} ${pos.target_seniority ? `· ${esc(pos.target_seniority)}` : ""}</h4>
      <p>${esc(pos.identity_statement)}</p>
      ${pos.support_reasoning ? `<p style="margin-top:6px">${esc(pos.support_reasoning)}</p>` : ""}
      ${pos.differentiators?.length
        ? `<p style="margin-top:6px"><b>Differentiators:</b> ${pos.differentiators.map(esc).join(" · ")}</p>`
        : ""}
      ${pos.emphasise?.length
        ? `<p style="margin-top:4px"><b>Emphasise:</b> ${pos.emphasise.map(esc).join(", ")}</p>`
        : ""}
    </div>
    ${(a.warnings || [])
      .map((w) => `<div class="item warn"><p>${esc(w)}</p></div>`)
      .join("")}`;

  renderMatrix();
  $("matrixFilter").oninput = renderMatrix;
  $("onlyGaps").onchange = renderMatrix;

  // gaps
  const gaps = $("gapsList");
  gaps.innerHTML = "";
  if (!a.gaps.length) {
    gaps.appendChild(el("div", "item ok", "<p>No unmet requirements. Strong fit.</p>"));
  }
  a.gaps.forEach((g) => {
    const cls = g.risk === "HIGH" ? "crit" : g.risk === "MEDIUM" ? "warn" : "info";
    gaps.appendChild(
      el(
        "div",
        `item ${cls}`,
        `<div class="row between center">
           <h4>${esc(g.requirement)}</h4>
           <span class="badge ${g.risk === "HIGH" ? "bad" : g.risk === "MEDIUM" ? "warn" : "info"}">
             ${esc(g.priority)} · ${esc(g.risk)} risk</span>
         </div>
         <p>${esc(g.evidence_status)}</p>
         <p style="margin-top:5px"><b>Recommendation:</b> ${esc(g.recommendation)}</p>`
      )
    );
  });

  renderBreakdown($("baselineBreakdown"), a.baseline_scores);
  renderTrace($("analysisTrace"), a.trace);
  $("generateHint").textContent = a.usage
    ? `${a.usage.calls} model calls · ${a.usage.input_tokens.toLocaleString()} in / ${a.usage.output_tokens.toLocaleString()} out tokens`
    : "";
}

const TIER = {
  EXACT: ["ok", "Exact"],
  STRONG_SEMANTIC: ["ok", "Strong"],
  PARTIAL: ["warn", "Partial"],
  WEAK: ["warn", "Weak"],
  NONE: ["bad", "None"],
};

function renderMatrix() {
  const a = state.analysis;
  if (!a) return;
  const q = $("matrixFilter").value.toLowerCase().trim();
  const onlyGaps = $("onlyGaps").checked;
  const rows = a.matrix.filter(
    (r) =>
      (!q || r.requirement.toLowerCase().includes(q)) && (!onlyGaps || r.score < 0.6)
  );

  const table = $("matrixTable");
  table.innerHTML =
    `<thead><tr><th>Requirement</th><th>Pri</th><th>Match</th>
     <th style="width:110px">Score</th><th>Evidence</th></tr></thead>`;
  const tbody = el("tbody");
  rows.forEach((r) => {
    const [cls, label] = TIER[r.match_type] || ["bad", r.match_type];
    const pct = Math.round(r.score * 100);
    const tr = el("tr");
    tr.innerHTML = `
      <td><b>${esc(r.requirement)}</b>
        ${r.notes ? `<div class="evidence">${esc(r.notes)}</div>` : ""}</td>
      <td><span class="badge ${r.priority === "P0" ? "bad" : r.priority === "P1" ? "warn" : "info"}">${esc(r.priority)}</span></td>
      <td><span class="badge ${cls}">${esc(label)}</span></td>
      <td><b style="font-variant-numeric:tabular-nums">${pct}%</b>
        <div class="bar"><i style="width:${pct}%;background:${scoreColor(pct)}"></i></div></td>
      <td>${r.evidence?.length
        ? `<div class="evidence">${r.evidence.map((e) => `“${esc(e)}”`).join("<br/>")}
             ${r.sources?.length ? `<br/><i>${esc(r.sources.join(" · "))}</i>` : ""}</div>`
        : `<span class="evidence">—</span>`}</td>`;
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  if (!rows.length) {
    table.appendChild(el("tbody", "", `<tr><td colspan="5" class="empty">No rows match.</td></tr>`));
  }
}

function renderBreakdown(container, report) {
  container.innerHTML = "";
  report.components.forEach((c) => {
    const item = el("div", "item info");
    item.innerHTML = `
      <div class="row between center">
        <h4>${esc(c.label)}</h4>
        <span class="badge info">${c.raw.toFixed(0)}/100 · weight ${(c.weight * 100).toFixed(0)}%</span>
      </div>
      <div class="bar"><i style="width:${c.raw}%;background:${scoreColor(c.raw)}"></i></div>
      <p style="margin-top:6px">${esc(c.explanation)}</p>
      ${c.details?.length
        ? `<div class="offenders">${c.details.map((d) => `<div>${esc(d)}</div>`).join("")}</div>`
        : ""}`;
    container.appendChild(item);
  });
}

function renderTrace(container, trace) {
  container.innerHTML = "";
  (trace || []).forEach((n) => {
    const node = el("div", `node ${n.status}`);
    node.innerHTML = `<span class="dot"></span><span>${esc(n.name)}</span>
      <span class="ms">${n.status === "cached" ? "cached" : n.duration_ms + "ms"}</span>`;
    node.title = n.note + (n.error ? `\n${n.error}` : "");
    container.appendChild(node);
  });
}

/* --------------------------------------------------------------- generate */
$("generateBtn").onclick = async () => {
  if (!state.analysis) return;
  showOverlay("Generating tailored resume", [
    "Writing the tailored document",
    "Running ATS format checks",
    "Running the truthfulness gate",
    "Repairing any failures and re-validating",
    "Surfacing supported keywords",
    "Auditing every claim against your master resume",
    "Simulating the recruiter scan",
  ]);
  try {
    state.generation = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        analysis_id: state.analysis.analysis_id,
        max_repair_iterations: parseInt($("repairIters").value, 10),
        lift_rounds: parseInt($("liftRounds").value, 10),
      }),
    });
    renderGeneration(state.generation);
    enableTab("resume");
    enableTab("changes");
    showTab("resume");
    toast(
      state.generation.status === "optimized"
        ? "Resume generated — all critical checks passed"
        : "Resume generated — needs review",
      state.generation.status !== "optimized"
    );
  } catch (e) {
    toast(e.message, true);
  } finally {
    hideOverlay();
  }
};

/* ------------------------------------------------------ render generation */
function renderGeneration(g) {
  const strip = $("finalScores");
  strip.innerHTML = "";
  const comp = (k) => g.scores.components.find((c) => c.key === k)?.raw ?? 0;
  strip.appendChild(dial(g.scores.total, "Overall fit", g.scores.band));
  strip.appendChild(dial(g.ats_report_score ?? atsScore(g), "ATS readiness", "format compliance"));
  strip.appendChild(dial(comp("semantic_alignment"), "JD match", "requirement coverage"));
  strip.appendChild(dial(g.recruiter.score, "Recruiter readability", "10-second scan"));

  const banner = $("statusBanner");
  if (g.status === "optimized") {
    banner.innerHTML = `<div class="banner ok"><b>✓ Optimized.</b>
      <div>Every critical ATS and truthfulness check passed. Note: no resume is
      universally "100% ATS compatible" — different platforms parse differently.
      This document complies with the rules that are safe across all of them.</div></div>`;
  } else {
    banner.innerHTML = `<div class="banner warn"><b>⚠ Needs review.</b><div>
      One or more critical checks did not pass, so this is not marked optimized.
      <ul>${g.status_reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div></div>`;
  }

  $("resumePreview").textContent = g.plain_text;

  const r = g.recruiter;
  $("recruiterBox").innerHTML = `
    <div class="item info">
      <p><b>Who is this?</b> ${esc(r.who_is_this)}</p>
      <p><b>Level:</b> ${esc(r.what_level)}</p>
      <p><b>Specialisation:</b> ${esc(r.specialisation)}</p>
      <p><b>Relevance:</b> ${esc(r.relevance_to_role)}</p>
      ${r.technologies_visible?.length
        ? `<p style="margin-top:6px"><b>Visible tech:</b> ${r.technologies_visible.map(esc).join(", ")}</p>`
        : ""}
    </div>
    <div class="item ok"><h4>Top strengths at a glance</h4>
      <ul style="margin:4px 0 0;padding-left:18px;font-size:12.5px">
        ${(r.top_strengths || []).map((s) => `<li>${esc(s)}</li>`).join("")}</ul></div>
    <div class="item warn"><h4>Weaknesses a recruiter would notice</h4>
      <ul style="margin:4px 0 0;padding-left:18px;font-size:12.5px">
        ${(r.top_weaknesses || []).map((s) => `<li>${esc(s)}</li>`).join("")}</ul></div>`;

  renderValidation();
  document.querySelectorAll("#validationTabs .seg").forEach((b) => {
    b.onclick = () => {
      state.validationView = b.dataset.v;
      document.querySelectorAll("#validationTabs .seg").forEach((x) =>
        x.classList.toggle("active", x === b)
      );
      renderValidation();
    };
  });

  renderLoop(g.diff.loop, g.diff.lift);
  renderChanges(g);
  prefillTracker(g);
}

const atsScore = (g) => {
  const w = { critical: 3, warning: 1, info: 0.25 };
  const checks = g.ats_report.checks;
  const total = checks.reduce((s, c) => s + w[c.severity], 0);
  const earned = checks.reduce((s, c) => s + (c.passed ? w[c.severity] : 0), 0);
  return total ? (100 * earned) / total : 0;
};

function renderValidation() {
  const g = state.generation;
  const report = state.validationView === "ats" ? g.ats_report : g.truth_report;
  const box = $("validationList");
  box.innerHTML = "";
  const sorted = [...report.checks].sort(
    (a, b) => a.passed - b.passed || (a.severity === "critical" ? -1 : 1)
  );
  sorted.forEach((c) => {
    const cls = c.passed ? "ok" : c.severity === "critical" ? "crit" : c.severity;
    const item = el("div", `item ${cls}`);
    item.innerHTML = `
      <div class="row between center">
        <h4>${c.passed ? "✓" : "✕"} ${esc(c.label)}</h4>
        <span class="badge ${c.passed ? "ok" : c.severity === "critical" ? "bad" : "warn"}">
          ${esc(c.severity)}</span>
      </div>
      <p>${esc(c.detail)}</p>
      ${c.offenders?.length
        ? `<div class="offenders">${c.offenders.map((o) => `<div>${esc(o)}</div>`).join("")}</div>`
        : ""}`;
    box.appendChild(item);
  });
}

function renderLoop(loop, lift) {
  const box = $("loopBox");
  box.innerHTML = "";
  if (!loop || !loop.attempts) {
    box.innerHTML = `<p class="hint">No loop data.</p>`;
    return;
  }
  box.appendChild(
    el(
      "div",
      `banner ${loop.converged ? "ok" : "warn"}`,
      `<div><b>${loop.converged ? "Converged" : "Stopped without full convergence"}</b> ·
       ${loop.iterations} iteration${loop.iterations === 1 ? "" : "s"} ·
       stop reason: <code>${esc(loop.stop_reason)}</code></div>`
    )
  );
  loop.attempts.forEach((a) => {
    const row = el("div", "looprow");
    row.innerHTML = `<div class="n">${a.iteration}</div>
      <div style="flex:1">
        <b>Score ${a.score.toFixed(1)}</b>
        ${a.critical_failures.length
          ? `<span class="badge bad" style="margin-left:8px">${a.critical_failures.length} critical</span>`
          : `<span class="badge ok" style="margin-left:8px">critical clear</span>`}
        ${a.warnings.length
          ? `<span class="badge warn" style="margin-left:6px">${a.warnings.length} warnings</span>`
          : ""}
        ${a.critical_failures.length
          ? `<div class="evidence">${a.critical_failures.map(esc).join(" · ")}</div>`
          : ""}
        ${a.feedback_sent.length
          ? `<div class="evidence">↩ fed back: ${esc(a.feedback_sent.length)} instruction(s)</div>`
          : ""}
      </div>`;
    box.appendChild(row);
  });
  (lift || []).forEach((r) => {
    const row = el("div", "looprow");
    row.innerHTML = `<div class="n">L${r.round}</div><div style="flex:1">
      <b>Keyword lift</b> — ${esc(r.action)}
      ${r.missing?.length ? `<div class="evidence">${r.missing.map(esc).join(", ")}</div>` : ""}
      </div>`;
    box.appendChild(row);
  });
}

/* ----------------------------------------------------------- changes tab */
function renderChanges(g) {
  const list = $("changesList");
  list.innerHTML = "";
  if (!g.changes.length) list.innerHTML = `<p class="hint">No changes recorded.</p>`;
  g.changes.forEach((c) => {
    list.appendChild(
      el(
        "div",
        "item info",
        `<div class="row between center">
           <h4>${esc(c.change)}</h4><span class="chip ${esc(c.category)}">${esc(c.category)}</span>
         </div>
         <p><b>Reason:</b> ${esc(c.reason)}</p>
         ${c.source ? `<p><b>Source:</b> ${esc(c.source)}</p>` : ""}`
      )
    );
  });

  const s = g.diff.stats || {};
  $("diffStats").innerHTML = `
    <div><b>${s.kept ?? 0}</b>kept verbatim</div>
    <div><b>${s.rewritten ?? 0}</b>rewritten</div>
    <div><b>${s.added ?? 0}</b>newly composed</div>
    <div><b>${s.removed ?? 0}</b>dropped as irrelevant</div>
    <div><b>${(g.diff.reordered_skills || []).length}</b>skills promoted</div>`;

  const diff = $("diffList");
  diff.innerHTML = "";
  (g.diff.rewritten || []).forEach((d) => {
    diff.appendChild(
      el(
        "div",
        "item",
        `<div class="row between center">
           <h4>${esc(d.where)}</h4><span class="chip rewritten">rewritten · ${esc(d.similarity)} similar</span>
         </div>
         <div class="diffpair">
           <div class="before"><div class="label">Master resume</div>${esc(d.before)}</div>
           <div class="after"><div class="label">Tailored</div>${esc(d.after)}</div>
         </div>`
      )
    );
  });
  (g.diff.removed || []).forEach((d) => {
    diff.appendChild(
      el(
        "div",
        "item warn",
        `<span class="chip removed">removed</span> <b>${esc(d.where)}</b>
         <p style="margin-top:5px">${esc(d.text)}</p>`
      )
    );
  });
  (g.diff.reordered_skills || []).forEach((d) => {
    diff.appendChild(
      el(
        "div",
        "item info",
        `<span class="chip rewritten">reordered</span>
         <b>${esc(d.skill)}</b> moved from position ${d.from} to ${d.to} in Core Skills.`
      )
    );
  });
  if (!diff.children.length) diff.innerHTML = `<p class="hint">No structural changes.</p>`;
}

/* ---------------------------------------------------------------- exports */
const download = (fmt) => {
  if (!state.generation) return;
  window.location.href = `/api/export/${state.generation.version_id}.${fmt}`;
};
$("dlTxt").onclick = () => download("txt");
$("dlDocx").onclick = () => download("docx");
$("dlPdf").onclick = () => download("pdf");
$("copyBtn").onclick = async () => {
  await navigator.clipboard.writeText(state.generation.plain_text);
  toast("Plain text copied");
};

/* ---------------------------------------------------------------- tracker */
function prefillTracker(g) {
  const jd = state.analysis?.jd || {};
  $("appCompany").value = jd.company || "";
  $("appTitle").value = jd.job_title || "";
  $("appDate").value = new Date().toISOString().slice(0, 10);
}

$("saveAppBtn").onclick = async () => {
  if (!state.generation) return toast("Generate a resume first.", true);
  const g = state.generation;
  const comp = (k) => g.scores.components.find((c) => c.key === k)?.raw ?? 0;
  try {
    await api("/api/applications", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        company: $("appCompany").value,
        job_title: $("appTitle").value,
        jd_excerpt: (state.analysis?.jd?.responsibilities || []).slice(0, 2).join(" "),
        version_id: g.version_id,
        version_name: g.version_name,
        positioning: state.analysis?.positioning?.target_title || "",
        applied_on: $("appDate").value,
        ats_score: atsScore(g),
        jd_match_score: comp("semantic_alignment"),
        url: $("appUrl").value,
        status: $("appStatus").value,
        notes: $("appNotes").value,
      }),
    });
    toast("Saved to tracker");
    loadTracker();
  } catch (e) {
    toast(e.message, true);
  }
};

async function loadTracker() {
  try {
    const [analytics, versions, apps] = await Promise.all([
      api("/api/analytics/positioning"),
      api("/api/versions"),
      api("/api/applications"),
    ]);

    $("trackerNote").textContent = analytics.note;
    const pt = $("positioningTable");
    pt.innerHTML = "";
    if (!analytics.rows.length) {
      pt.innerHTML = `<p class="empty">No applications tracked yet. Save one from the
        Changes tab to start building an outcome signal.</p>`;
    } else {
      const table = el("table", "matrix");
      table.innerHTML = `<thead><tr><th>Positioning</th><th>Apps</th><th>Interviews</th>
        <th>Offers</th><th>Interview rate</th><th>Offer rate</th><th>Avg ATS</th><th></th></tr></thead>`;
      const tb = el("tbody");
      analytics.rows.forEach((r) => {
        const best = r.positioning === analytics.best_positioning;
        tb.appendChild(
          el(
            "tr",
            "",
            `<td><b>${esc(r.positioning)}</b>${best ? ' <span class="badge ok">best</span>' : ""}</td>
             <td>${r.applications}</td><td>${r.interviews}</td><td>${r.offers}</td>
             <td><b>${r.interview_rate}%</b></td><td><b>${r.offer_rate}%</b></td>
             <td>${r.avg_ats}</td>
             <td>${r.significant ? "" : '<span class="badge warn">low sample</span>'}</td>`
          )
        );
      });
      table.appendChild(tb);
      pt.appendChild(table);
    }

    const vt = $("versionsTable");
    vt.innerHTML = `<thead><tr><th>Version</th><th>Role</th><th>Company</th>
      <th>ATS</th><th>Match</th><th>Status</th><th>Created</th><th></th></tr></thead>`;
    const vb = el("tbody");
    versions.forEach((v) => {
      const tr = el("tr");
      tr.innerHTML = `<td><b>${esc(v.name)}</b></td><td>${esc(v.job_title || "")}</td>
        <td>${esc(v.company || "")}</td><td>${(v.ats_score || 0).toFixed(0)}</td>
        <td>${(v.jd_match_score || 0).toFixed(0)}</td>
        <td><span class="badge ${v.status === "optimized" ? "ok" : "warn"}">${esc(v.status)}</span></td>
        <td>${esc((v.created_at || "").slice(0, 10))}</td>
        <td><button class="ghost small" data-dl="${esc(v.id)}">PDF</button>
            <button class="ghost small" data-del="${esc(v.id)}">Delete</button></td>`;
      vb.appendChild(tr);
    });
    vt.innerHTML = vt.innerHTML;
    vt.appendChild(vb);
    if (!versions.length) {
      vt.appendChild(el("tbody", "", `<tr><td colspan="8" class="empty">No versions yet.</td></tr>`));
    }
    vt.querySelectorAll("[data-dl]").forEach(
      (b) => (b.onclick = () => (window.location.href = `/api/export/${b.dataset.dl}.pdf`))
    );
    vt.querySelectorAll("[data-del]").forEach(
      (b) =>
        (b.onclick = async () => {
          await api(`/api/versions/${b.dataset.del}`, { method: "DELETE" });
          toast("Version deleted");
          loadTracker();
        })
    );

    const at = $("appsTable");
    at.innerHTML = `<thead><tr><th>Company</th><th>Role</th><th>Positioning</th>
      <th>Applied</th><th>Status</th><th>Stage</th><th>ATS</th><th></th></tr></thead>`;
    const ab = el("tbody");
    apps.forEach((a) => {
      ab.appendChild(
        el(
          "tr",
          "",
          `<td><b>${esc(a.company || "")}</b></td><td>${esc(a.job_title || "")}</td>
           <td>${esc(a.positioning || "")}</td><td>${esc(a.applied_on || "")}</td>
           <td>${esc(a.status || "")}</td><td>${esc(a.interview_stage || "")}</td>
           <td>${(a.ats_score || 0).toFixed(0)}</td>
           <td><button class="ghost small" data-delapp="${esc(a.id)}">Delete</button></td>`
        )
      );
    });
    at.appendChild(ab);
    if (!apps.length) {
      at.appendChild(el("tbody", "", `<tr><td colspan="8" class="empty">No applications yet.</td></tr>`));
    }
    at.querySelectorAll("[data-delapp]").forEach(
      (b) =>
        (b.onclick = async () => {
          await api(`/api/applications/${b.dataset.delapp}`, { method: "DELETE" });
          loadTracker();
        })
    );
  } catch (e) {
    toast(e.message, true);
  }
}
$("refreshTracker").onclick = loadTracker;

$("purgeBtn").onclick = async () => {
  if (!confirm("Delete every stored resume, analysis, version and application? This cannot be undone."))
    return;
  await api("/api/data", { method: "DELETE" });
  toast("All local data deleted");
  loadTracker();
};

/* ------------------------------------------------------------------- init */
loadConfig();
loadTracker();
