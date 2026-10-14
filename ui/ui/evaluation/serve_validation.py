#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Interactive web UI for the offline validation results, served on localhost.

`validate_prediction_vs_dataset.py` writes, per episode, an `episode_<id>.json`
containing the three curves (recorded state, recorded action, model prediction)
for every joint. This script serves those over HTTP with a page that shows all
joints IN PARALLEL as small multiples with a synchronised crosshair: hover any
joint and every other joint jumps to the same frame, so you can read the whole
robot at one instant instead of one joint at a time.

Usage:
  cd ~/Desktop/I/grootn1.7/Isaac-GR00T && source .venv/bin/activate
  python gr00t/eval/real_robot/daksha_b1/serve_validation_ui.py \
      --data-dir /home/s1/Desktop/datasets/validation_pick_place_brown \
      --port 8123

Then open http://localhost:8123
"""

from __future__ import annotations

import argparse
import json
import socket
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Daksha B1 Validation</title>
<style>
/* ---- tokens: light is the base, dark redefines only the values ---------- */
:root{
  color-scheme: light;
  --page:#f6f6f4;
  --surface-1:#fcfcfb;
  --border:#e4e4e0;
  --grid:#ececea;
  --text-primary:#0b0b0b;
  --text-secondary:#52514e;
  --text-muted:#84837d;
  --series-gt:#2a78d6;      /* categorical slot 1 - recorded action */
  --series-pred:#eb6834;    /* categorical slot 2 - model prediction */
  --series-state:#a3a29b;   /* recessive reference line, not an identity hue */
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
  --shadow:0 1px 2px rgba(0,0,0,.05);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --page:#121211;
    --surface-1:#1a1a19;
    --border:#302f2d;
    --grid:#262624;
    --text-primary:#ffffff;
    --text-secondary:#c3c2b7;
    --text-muted:#8f8e85;
    --series-gt:#3987e5;
    --series-pred:#d95926;
    --series-state:#75746d;
    --shadow:none;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --page:#121211; --surface-1:#1a1a19; --border:#302f2d; --grid:#262624;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8f8e85;
  --series-gt:#3987e5; --series-pred:#d95926; --series-state:#75746d; --shadow:none;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:14px;line-height:1.45}
.wrap{max-width:1560px;margin:0 auto;padding:22px 20px 60px}
h1{font-size:19px;margin:0 0 3px;letter-spacing:-.01em}
.sub{color:var(--text-secondary);font-size:12.5px;margin-bottom:16px}
.sub code{background:rgba(128,128,128,.16);padding:1px 6px;border-radius:4px;font-size:12px}

/* ---- one filter row above everything it scopes ------------------------- */
.filters{position:sticky;top:0;z-index:20;background:var(--page);
  padding:10px 0 10px;border-bottom:1px solid var(--border);margin-bottom:14px;
  display:flex;flex-wrap:wrap;gap:18px;align-items:center}
.fgroup{display:flex;align-items:center;gap:7px}
.flabel{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted)}
button{font:inherit;font-size:12.5px;color:var(--text-secondary);background:var(--surface-1);
  border:1px solid var(--border);border-radius:7px;padding:5px 11px;cursor:pointer;min-height:30px}
button:hover{border-color:var(--text-muted)}
button[aria-pressed="true"]{background:var(--text-primary);color:var(--page);border-color:var(--text-primary)}
button:focus-visible{outline:2px solid var(--series-gt);outline-offset:2px}
.legend button{display:flex;align-items:center;gap:7px}
.legend button[aria-pressed="false"]{opacity:.45}
.swatch{width:14px;height:3px;border-radius:2px;display:inline-block}
.swatch.thin{height:2px}

/* ---- kpis --------------------------------------------------------------*/
.kpis{display:flex;flex-wrap:wrap;gap:26px;padding:14px 16px;background:var(--surface-1);
  border:1px solid var(--border);border-radius:10px;margin-bottom:8px;box-shadow:var(--shadow)}
.kpi .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted)}
.kpi .v{font-size:23px;font-weight:600;letter-spacing:-.02em}
.kpi .v.name{font-size:15px;font-weight:600;letter-spacing:0;padding-top:5px}
.kpi .u{font-size:12px;color:var(--text-muted);font-weight:400}

/* ---- frame readout ------------------------------------------------------*/
.framebar{display:flex;align-items:center;gap:14px;padding:9px 16px;margin-bottom:16px;
  background:var(--surface-1);border:1px solid var(--border);border-radius:10px;
  font-variant-numeric:tabular-nums;color:var(--text-secondary);font-size:12.5px}
.framebar b{color:var(--text-primary);font-size:14px}

/* ---- chart grid --------------------------------------------------------*/
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;
  padding:10px 12px 6px;box-shadow:var(--shadow);position:relative}
.card.dim{opacity:.62}
.ctitle{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:2px}
.cname{font-size:12.5px;font-weight:600;letter-spacing:-.01em}
.cname .dim-i{color:var(--text-muted);font-weight:400;margin-right:5px}
.cmeta{font-size:11px;color:var(--text-secondary);font-variant-numeric:tabular-nums;white-space:nowrap}
.badge{display:inline-flex;align-items:center;gap:4px;font-size:10.5px;font-weight:600;
  padding:1px 6px;border-radius:999px;border:1px solid currentColor;margin-left:6px}
svg{display:block;width:100%;overflow:visible;touch-action:none}
.tip{position:absolute;pointer-events:none;z-index:9;background:var(--surface-1);
  border:1px solid var(--border);border-radius:8px;padding:7px 9px;font-size:11.5px;
  box-shadow:0 4px 14px rgba(0,0,0,.14);white-space:nowrap;font-variant-numeric:tabular-nums;
  color:var(--text-secondary);opacity:0;transition:opacity .08s}
.tip .row{display:flex;align-items:center;gap:6px}
.tip .row b{color:var(--text-primary);margin-left:auto;padding-left:12px}

/* ---- table view --------------------------------------------------------*/
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums;
  background:var(--surface-1)}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}
th:nth-child(1),td:nth-child(1),th:nth-child(2),td:nth-child(2){text-align:left}
/* sticky to the scroll container (.tblwrap), not the page - a page-relative
   offset here lands the header on top of the first data row */
th{position:sticky;top:0;background:var(--surface-1);color:var(--text-secondary);
  font-size:11px;text-transform:uppercase;letter-spacing:.05em;z-index:5}
.tblwrap{border:1px solid var(--border);border-radius:10px;overflow:auto;margin-bottom:22px}
h2{font-size:14px;margin:22px 0 8px}
.note{font-size:12.5px;line-height:1.6;color:var(--text-secondary);background:var(--surface-1);
  border:1px solid var(--border);border-left:3px solid var(--warning);border-radius:8px;
  padding:11px 14px;margin:20px 0 0}
.hidden{display:none}
</style>
</head>
<body>
<div class="wrap">
  <h1>Daksha B1 &mdash; model prediction vs recorded dataset</h1>
  <div class="sub" id="sub"></div>

  <div class="filters">
    <div class="fgroup"><span class="flabel">Episode</span><span id="eps"></span></div>
    <div class="fgroup legend"><span class="flabel">Series</span>
      <button id="s-gt"    aria-pressed="true"><span class="swatch" style="background:var(--series-gt)"></span>Dataset action</button>
      <button id="s-pred"  aria-pressed="true"><span class="swatch" style="background:var(--series-pred)"></span>Model prediction</button>
      <button id="s-state" aria-pressed="true"><span class="swatch thin" style="background:var(--series-state)"></span>Robot state</button>
    </div>
    <div class="fgroup"><span class="flabel">Joints</span>
      <button id="t-static" aria-pressed="true">Show static joints</button></div>
    <div class="fgroup"><span class="flabel">View</span>
      <button id="v-charts" aria-pressed="true">Charts</button>
      <button id="v-table"  aria-pressed="false">Table</button></div>
    <div class="fgroup" style="margin-left:auto"><button id="theme">Theme</button></div>
  </div>

  <div class="kpis" id="kpis"></div>
  <div class="framebar" id="framebar"></div>

  <div id="charts"><div class="grid" id="grid"></div></div>
  <div id="table" class="hidden"></div>

  <div class="note" id="note"></div>
</div>

<script>
"use strict";
const $ = s => document.querySelector(s);
const FPS = 30;
// Never zoom a y-axis tighter than this (rad). A joint that barely moves would
// otherwise have its sub-millirad jitter magnified into a dramatic-looking error.
const MIN_SPAN = 0.12;
// Below this total travel a joint did not meaningfully move in the episode, so a
// percentage error against its range is meaningless.
const STATIC_RANGE = 0.02;

const S = {
  meta:null, ep:null, cache:{}, d:null,
  show:{gt:true, pred:true, state:true},
  showStatic:true, view:"charts", frame:null, pinned:false,
};
window.vizState = S;   // debug/automation hook: inspect or drive the page from the console

const fmt = (v,n=3) => (v>=0?" ":"") + v.toFixed(n);

/* ---------- data ---------------------------------------------------------- */
async function loadEpisode(id){
  if(!S.cache[id]){
    const r = await fetch(`episode_${id}.json`, {cache:"no-store"});
    if(!r.ok) throw new Error(`episode_${id}.json -> HTTP ${r.status}`);
    S.cache[id] = await r.json();
  }
  S.ep = id; S.d = S.cache[id];
  S.frame = Math.min(S.frame ?? 0, S.d.gt.length-1);
  renderAll();
}

function verdictOf(row){
  if(row.gt_range < STATIC_RANGE) return {label:"static", color:"var(--text-muted)"};
  if(row.nmae !== null && row.nmae < 0.10) return {label:"match", color:"var(--good)"};
  if(row.nmae !== null && row.nmae < 0.25) return {label:"close", color:"var(--warning)"};
  return {label:"check", color:"var(--critical)"};
}
const visibleJoints = () => S.d.rows
  .map((r,i)=>i)
  .filter(i => S.showStatic || S.d.rows[i].gt_range >= STATIC_RANGE);

/* ---------- chart --------------------------------------------------------- */
const PAD = {l:48, r:12, t:10, b:22};
const H = 148;
const NS = "http://www.w3.org/2000/svg";
const el = (n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};

function seriesFor(j){
  const out = [];
  if(S.show.state && S.d.state) out.push({key:"state", name:"Robot state",       col:"var(--series-state)", w:1.25, data:S.d.state.map(r=>r[j])});
  if(S.show.gt)                 out.push({key:"gt",    name:"Dataset action",    col:"var(--series-gt)",    w:2,    data:S.d.gt.map(r=>r[j])});
  if(S.show.pred)               out.push({key:"pred",  name:"Model prediction",  col:"var(--series-pred)",  w:1.75, data:S.d.pred.map(r=>r[j])});
  return out;
}

function drawChart(card, j, W){
  const host = card.querySelector(".plot");
  const n = S.d.gt.length;
  const ser = seriesFor(j);
  host.innerHTML = "";
  const svg = el("svg",{width:W, height:H, viewBox:`0 0 ${W} ${H}`, role:"img",
                        "aria-label":`${S.d.labels[j]} over ${n} frames`});

  // y-domain: always includes gt+pred so hiding a series never rescales the frame
  let lo = Infinity, hi = -Infinity;
  for(const a of [S.d.gt, S.d.pred]) for(let i=0;i<n;i++){ const v=a[i][j]; if(v<lo)lo=v; if(v>hi)hi=v; }
  if(hi-lo < MIN_SPAN){ const m=(hi+lo)/2; lo=m-MIN_SPAN/2; hi=m+MIN_SPAN/2; }
  const pad=(hi-lo)*0.10; lo-=pad; hi+=pad;

  const pw = W-PAD.l-PAD.r, ph = H-PAD.t-PAD.b;
  const X = i => PAD.l + (n<2?0:i/(n-1)*pw);
  const Y = v => PAD.t + (hi===lo?ph/2:(hi-v)/(hi-lo)*ph);

  // recessive grid: solid hairlines, one shade off the surface
  for(const t of [0,0.5,1]){
    const v = lo + t*(hi-lo), y = Y(v);
    svg.appendChild(el("line",{x1:PAD.l,x2:W-PAD.r,y1:y,y2:y,stroke:"var(--grid)","stroke-width":1}));
    const tx = el("text",{x:PAD.l-7,y:y+3.5,"text-anchor":"end","font-size":9.5,
                          fill:"var(--text-muted)","font-variant-numeric":"tabular-nums"});
    tx.textContent = v.toFixed(2); svg.appendChild(tx);
  }
  for(let k=0;k<=4;k++){
    const i = Math.round(k/4*(n-1)), x = X(i);
    const tx = el("text",{x, y:H-6, "text-anchor":k===0?"start":(k===4?"end":"middle"),
                          "font-size":9.5, fill:"var(--text-muted)","font-variant-numeric":"tabular-nums"});
    tx.textContent = i; svg.appendChild(tx);
  }

  for(const s of ser){
    let d = "";
    for(let i=0;i<n;i++) d += (i?"L":"M") + X(i).toFixed(2) + " " + Y(s.data[i]).toFixed(2);
    svg.appendChild(el("path",{d, fill:"none", stroke:s.col, "stroke-width":s.w,
                               "stroke-linejoin":"round","stroke-linecap":"round"}));
  }

  // crosshair layer, updated in place by setFrame
  const cross = el("g",{class:"cross", opacity:0});
  cross.appendChild(el("line",{class:"cl", y1:PAD.t, y2:H-PAD.b, stroke:"var(--text-muted)","stroke-width":1}));
  for(const s of ser){
    // 2px surface ring so an overlapping marker stays readable
    cross.appendChild(el("circle",{class:"cd", r:3.5, fill:s.col, stroke:"var(--surface-1)","stroke-width":2}));
  }
  svg.appendChild(cross);

  // hit area covers the whole plot, far bigger than the marks
  const hit = el("rect",{x:PAD.l,y:0,width:pw,height:H,fill:"transparent",style:"cursor:crosshair"});
  svg.appendChild(hit);
  host.appendChild(svg);

  card._geom = {X, Y, n, ser, W, pw};
  const toFrame = ev => {
    const r = svg.getBoundingClientRect();
    const rel = (ev.clientX - r.left) / (r.width) * W;   // svg user units
    return Math.max(0, Math.min(n-1, Math.round((rel-PAD.l)/pw*(n-1))));
  };
  hit.addEventListener("pointermove", ev => { if(!S.pinned){ S.hoverCard = j; setFrame(toFrame(ev)); } });
  hit.addEventListener("pointerleave", () => { if(!S.pinned){ S.hoverCard = null; renderTips(); } });
  hit.addEventListener("click", ev => { S.pinned = !S.pinned; S.hoverCard = j; setFrame(toFrame(ev)); });
}

/* ---------- crosshair sync ------------------------------------------------ */
function setFrame(i){
  S.frame = i;
  for(const card of document.querySelectorAll(".card")){
    const j = +card.dataset.j, g = card._geom; if(!g) continue;
    const cross = card.querySelector(".cross");
    cross.setAttribute("opacity", 1);
    const x = g.X(i);
    cross.querySelector(".cl").setAttribute("x1", x);
    cross.querySelector(".cl").setAttribute("x2", x);
    const dots = cross.querySelectorAll(".cd");
    g.ser.forEach((s,k)=>{ dots[k].setAttribute("cx", x); dots[k].setAttribute("cy", g.Y(s.data[i])); });
  }
  renderTips(); renderFrameBar();
  if(S.view === "table") renderFrameTable();
}

function renderTips(){
  for(const card of document.querySelectorAll(".card")){
    const tip = card.querySelector(".tip"), j = +card.dataset.j;
    if(S.hoverCard !== j || S.frame === null){ tip.style.opacity = 0; continue; }
    const g = card._geom;
    tip.innerHTML = `<div style="color:var(--text-muted);margin-bottom:3px">frame ${S.frame}
      &middot; ${(S.frame/FPS).toFixed(2)} s</div>` +
      g.ser.map(s=>`<div class="row"><span class="swatch" style="background:${s.col}"></span>${s.name}
        <b>${fmt(s.data[S.frame])}</b></div>`).join("");
    tip.style.opacity = 1;
    const x = g.X(S.frame) / g.W * card.querySelector(".plot").clientWidth;
    const left = Math.min(Math.max(6, x + 12), card.clientWidth - tip.offsetWidth - 6);
    tip.style.left = left + "px";
    tip.style.top  = "34px";
  }
}

/* ---------- render -------------------------------------------------------- */
function renderKpis(){
  const rows = S.d.rows, n = S.d.gt.length;
  let se=0, sa=0, c=0;
  for(let i=0;i<n;i++) for(let j=0;j<rows.length;j++){
    const e = S.d.gt[i][j]-S.d.pred[i][j]; sa += Math.abs(e); se += e*e; c++;
  }
  const moving = rows.filter(r=>r.gt_range>=STATIC_RANGE).length;
  const worst = rows.filter(r=>r.gt_range>=STATIC_RANGE)
                    .sort((a,b)=>(b.nmae??-1)-(a.nmae??-1))[0];
  const kpi = (k,v,u="",cls="") => `<div class="kpi"><div class="k">${k}</div>
    <div class="v ${cls}">${v}<span class="u">${u}</span></div></div>`;
  $("#kpis").innerHTML =
    kpi("Overall MAE", (sa/c).toFixed(4), " rad") +
    kpi("Overall RMSE", Math.sqrt(se/c).toFixed(4), " rad") +
    kpi("Frames", n, ` @${FPS}fps`) +
    kpi("Moving joints", `${moving}<span class="u">/${rows.length}</span>`) +
    kpi("Worst moving joint", worst ? worst.joint.split(":")[1] : "—",
        worst ? ` ${(worst.nmae*100).toFixed(0)}% of travel` : "", "name");
}

function renderGrid(){
  const grid = $("#grid"); grid.innerHTML = "";
  for(const j of visibleJoints()){
    const r = S.d.rows[j], v = verdictOf(r);
    const card = document.createElement("div");
    card.className = "card" + (r.gt_range < STATIC_RANGE ? " dim" : "");
    card.dataset.j = j;
    card.innerHTML = `
      <div class="ctitle">
        <div class="cname"><span class="dim-i">${j}</span>${S.d.labels[j].replace(":"," · ")}
          <span class="badge" style="color:${v.color}">${v.label}</span></div>
      </div>
      <div class="cmeta">MAE ${r.mae.toFixed(4)} rad &middot; travel ${r.gt_range.toFixed(3)} rad
        ${r.gt_range>=STATIC_RANGE ? "&middot; "+(r.nmae*100).toFixed(0)+"% of travel" : ""}</div>
      <div class="plot"></div>
      <div class="tip"></div>`;
    grid.appendChild(card);
  }
  // Measure once, after every card is in the DOM: the grid columns are equal, and
  // measuring per card would read widths from before the page scrollbar appeared,
  // so early charts would end up on a slightly different scale than later ones.
  const W = Math.max(260, grid.querySelector(".plot").clientWidth);
  for(const card of grid.querySelectorAll(".card")) drawChart(card, +card.dataset.j, W);
  if(S.frame !== null) setFrame(S.frame);
}

function renderFrameBar(){
  const i = S.frame ?? 0;
  $("#framebar").innerHTML =
    `<span>Frame <b>${i}</b> / ${S.d.gt.length-1}</span>
     <span>t = <b>${(i/FPS).toFixed(2)}</b> s</span>
     <span style="color:var(--text-muted)">${S.pinned
        ? "pinned — click a chart to unpin"
        : "hover any joint to move every crosshair together · ←/→ to step · click to pin"}</span>`;
}

function renderFrameTable(){
  const i = S.frame ?? 0;
  const body = S.d.rows.map((r,j)=>{
    const st = S.d.state ? fmt(S.d.state[i][j]) : "—";
    const e = S.d.gt[i][j]-S.d.pred[i][j];
    return `<tr><td>${j}</td><td>${r.joint}</td><td>${st}</td>
      <td>${fmt(S.d.gt[i][j])}</td><td>${fmt(S.d.pred[i][j])}</td>
      <td style="color:${Math.abs(e)>0.05?"var(--critical)":"var(--text-secondary)"}">${fmt(e,4)}</td></tr>`;
  }).join("");
  $("#frametbl").innerHTML =
    `<tr><th>dim</th><th>joint</th><th>state</th><th>dataset action</th>
      <th>model prediction</th><th>error</th></tr>` + body;
  $("#frametitle").textContent = `Values at frame ${i} (t = ${(i/FPS).toFixed(2)} s)`;
}

function renderTable(){
  // Rank by error relative to travel, but keep static joints out of the ranking:
  // a joint that moved 0.0015 rad can post a big percentage on pure noise.
  const rank = r => r.gt_range < STATIC_RANGE ? -1 : (r.nmae ?? -1);
  const rows = [...S.d.rows].sort((a,b)=>rank(b)-rank(a));
  const body = rows.map(r=>{
    const v = verdictOf(r);
    return `<tr><td>${r.idx}</td><td>${r.joint}</td>
      <td>${r.mae.toFixed(4)}</td><td>${r.rmse.toFixed(4)}</td>
      <td>${r.gt_range.toFixed(4)}</td>
      <td>${r.gt_range<STATIC_RANGE?"—":(r.nmae*100).toFixed(1)+"%"}</td>
      <td style="color:${v.color};font-weight:600">${v.label}</td></tr>`;
  }).join("");
  $("#table").innerHTML =
    `<h2>Per-joint error (worst first)</h2>
     <div class="tblwrap"><table>
       <tr><th>dim</th><th>joint</th><th>MAE (rad)</th><th>RMSE (rad)</th>
           <th>travel (rad)</th><th>MAE / travel</th><th>verdict</th></tr>${body}
     </table></div>
     <h2 id="frametitle"></h2>
     <div class="tblwrap"><table id="frametbl"></table></div>`;
  renderFrameTable();
}

function renderAll(){
  $("#sub").innerHTML =
    `Checkpoint <code>${S.meta.model}</code> &middot; dataset <code>${S.meta.dataset}</code>
     &middot; embodiment <code>${S.meta.embodiment}</code> &middot; action horizon
     <code>${S.d.action_horizon}</code> &middot; ${S.d.rows.length} joints, model output order`;
  renderKpis(); renderFrameBar();
  if(S.view === "charts"){ $("#charts").classList.remove("hidden"); $("#table").classList.add("hidden"); renderGrid(); }
  else { $("#charts").classList.add("hidden"); $("#table").classList.remove("hidden"); renderTable(); }
}

/* ---------- controls ------------------------------------------------------ */
function toggle(btn, on){ btn.setAttribute("aria-pressed", on ? "true" : "false"); }

function wire(){
  const bind = (sel, key) => $(sel).addEventListener("click", () => {
    // keep at least one series on screen
    const others = Object.keys(S.show).filter(k=>k!==key).some(k=>S.show[k]);
    if(S.show[key] && !others) return;
    S.show[key] = !S.show[key];
    toggle($(sel), S.show[key]); renderGrid();
  });
  bind("#s-gt","gt"); bind("#s-pred","pred"); bind("#s-state","state");

  $("#t-static").addEventListener("click", () => {
    S.showStatic = !S.showStatic; toggle($("#t-static"), S.showStatic); renderGrid();
  });
  $("#v-charts").addEventListener("click", () => setView("charts"));
  $("#v-table").addEventListener("click", () => setView("table"));
  $("#theme").addEventListener("click", () => {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const cur = document.documentElement.getAttribute("data-theme") || (dark?"dark":"light");
    document.documentElement.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
    if(S.view === "charts") renderGrid();
  });
  addEventListener("keydown", ev => {
    if(S.frame === null) return;
    const step = ev.shiftKey ? 10 : 1;
    if(ev.key === "ArrowRight"){ setFrame(Math.min(S.d.gt.length-1, S.frame+step)); ev.preventDefault(); }
    if(ev.key === "ArrowLeft"){  setFrame(Math.max(0, S.frame-step)); ev.preventDefault(); }
  });
  let t; addEventListener("resize", () => { clearTimeout(t); t = setTimeout(()=>{ if(S.view==="charts") renderGrid(); }, 150); });
}

function setView(v){
  S.view = v; toggle($("#v-charts"), v==="charts"); toggle($("#v-table"), v==="table"); renderAll();
}

/* ---------- boot ---------------------------------------------------------- */
(async function(){
  wire();
  S.meta = await (await fetch("ui_meta.json", {cache:"no-store"})).json();
  $("#eps").innerHTML = S.meta.episode_ids
    .map(id=>`<button data-ep="${id}" aria-pressed="false">${id}</button>`).join(" ");
  $("#eps").addEventListener("click", async ev => {
    const b = ev.target.closest("button[data-ep]"); if(!b) return;
    for(const x of $("#eps").querySelectorAll("button")) toggle(x, x===b);
    await loadEpisode(+b.dataset.ep);
  });
  const first = $("#eps").querySelector("button");
  toggle(first, true);
  $("#note").innerHTML =
    `<b>How to read this.</b> Blue is the action the human demonstrator recorded; orange is what
     the model predicts from the same recorded camera frames and joint state; grey is the measured
     joint position. Overlapping blue and orange means the model learned that joint. Every chart
     shares one crosshair, so hovering one joint reads the whole robot at that instant.
     Y-axes never zoom tighter than ${MIN_SPAN} rad, so a joint that barely moves looks like it
     barely moves. Joints with under ${STATIC_RANGE} rad of travel are marked <em>static</em> &mdash;
     they are motionless in the recording, so a percentage error against their travel is meaningless.
     <b>This validates the model only</b> &mdash; the robot can still move wrongly if the ROS2 bridge
     maps joints in a different order, which is a separate check.`;
  await loadEpisode(S.meta.episode_ids[0]);
  setFrame(0);
})();
</script>
</body>
</html>
"""


def parse_args():
    p = argparse.ArgumentParser(description="Serve the Daksha B1 validation UI on localhost")
    p.add_argument(
        "--data-dir",
        default="/home/s1/Desktop/datasets/validation_pick_place_brown",
        help="Directory written by validate_prediction_vs_dataset.py (holds meta.json + episode_*.json).",
    )
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to reach it from another machine.")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    meta_file = data_dir / "meta.json"
    if not meta_file.exists():
        raise SystemExit(
            f"{meta_file} not found. Run validate_prediction_vs_dataset.py first "
            f"(it writes meta.json and episode_<id>.json into its --out-dir)."
        )
    meta = json.loads(meta_file.read_text())
    # Trust the directory, not meta.json: each validation run overwrites meta.json
    # with only the episodes it just did, so a second run would otherwise hide the
    # dumps from the first. The episode files themselves are the source of truth.
    meta["episode_ids"] = sorted(
        int(p.stem.split("_")[1]) for p in data_dir.glob("episode_*.json")
    )
    if not meta["episode_ids"]:
        raise SystemExit(f"no episode_*.json dumps in {data_dir}")
    (data_dir / "ui_meta.json").write_text(json.dumps(meta, indent=1))

    # Regenerate the page every start so a UI edit never needs a manual copy.
    (data_dir / "index.html").write_text(INDEX_HTML)

    handler = partial(SimpleHTTPRequestHandler, directory=str(data_dir))
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as e:
        raise SystemExit(f"cannot bind {args.host}:{args.port} ({e}). Is another server already up?")
    httpd.allow_reuse_address = True

    print(f"serving {data_dir}")
    print(f"episodes: {meta['episode_ids']}")
    print(f"\n  open  http://localhost:{args.port}\n")
    if args.host == "0.0.0.0":
        print(f"  LAN   http://{socket.gethostbyname(socket.gethostname())}:{args.port}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
