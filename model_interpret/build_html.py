"""Assemble the XGBoost explainer HTML (standalone + artifact body) from data + images."""
import json, os

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/manit/Desktop/fun_projects/tsa"
data = json.load(open(f"{OUT}/html_data.json"))
imgs = json.load(open(f"{OUT}/html_imgs.json"))

DATA_JS = json.dumps(data)
IMG_SINGLE = imgs["single_tree.png"]
IMG_IMP = imgs["feature_importance.png"]
IMG_SHAP = imgs["shap_summary.png"]
IMG_T0 = imgs["ensemble_tree_0.png"]
IMG_T1 = imgs["ensemble_tree_1.png"]

STYLE = r"""
<style>
:root{
  --paper:#eef1f3; --panel:#ffffff; --ink:#121a22; --muted:#5b6b78;
  --line:#d4dbe0; --accent:#0f766e; --accent-soft:#0f766e22;
  --pos:#2f9e6e; --neg:#c2410c;
  --shadow:0 1px 2px rgba(16,30,40,.06),0 8px 28px rgba(16,30,40,.07);
  --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root{
    --paper:#0c1116; --panel:#141c24; --ink:#e7edf2; --muted:#8ea0ad;
    --line:#243039; --accent:#4cc9bf; --accent-soft:#4cc9bf1f;
    --pos:#43c08a; --neg:#f0875a;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 34px rgba(0,0,0,.45);
  }
}
:root[data-theme="dark"]{
  --paper:#0c1116; --panel:#141c24; --ink:#e7edf2; --muted:#8ea0ad;
  --line:#243039; --accent:#4cc9bf; --accent-soft:#4cc9bf1f;
  --pos:#43c08a; --neg:#f0875a;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 34px rgba(0,0,0,.45);
}
:root[data-theme="light"]{
  --paper:#eef1f3; --panel:#ffffff; --ink:#121a22; --muted:#5b6b78;
  --line:#d4dbe0; --accent:#0f766e; --accent-soft:#0f766e22;
  --pos:#2f9e6e; --neg:#c2410c;
  --shadow:0 1px 2px rgba(16,30,40,.06),0 8px 28px rgba(16,30,40,.07);
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  line-height:1.6;-webkit-font-smoothing:antialiased;}
.wrap{max-width:860px;margin:0 auto;padding:0 24px;}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums;}
h1,h2,h3{text-wrap:balance;line-height:1.15;margin:0;}
section{padding:54px 0;border-top:1px solid var(--line);}
section:first-of-type{border-top:none;}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.18em;text-transform:uppercase;
  color:var(--accent);margin-bottom:14px;}
p{color:var(--ink);max-width:62ch;}
.muted{color:var(--muted);}

/* hero */
.hero{padding:76px 0 60px;}
.hero h1{font-size:clamp(32px,6vw,52px);font-weight:800;letter-spacing:-.02em;}
.hero .lede{font-size:clamp(17px,2.4vw,21px);color:var(--muted);margin-top:20px;max-width:58ch;}
.hero .lede b{color:var(--ink);font-weight:600;}
.flow{display:flex;align-items:stretch;gap:10px;flex-wrap:wrap;margin-top:34px;}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;
  box-shadow:var(--shadow);min-width:132px;}
.chip .k{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);}
.chip .v{font-family:var(--mono);font-size:22px;font-weight:600;margin-top:4px;}
.chip.accent{border-color:var(--accent);} .chip.accent .v{color:var(--accent);}
.arrow{align-self:center;color:var(--muted);font-family:var(--mono);font-size:20px;}

/* generic panel */
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  box-shadow:var(--shadow);padding:22px;}
.imgbox{margin-top:22px;border:1px solid var(--line);border-radius:12px;background:var(--panel);
  padding:14px;overflow-x:auto;box-shadow:var(--shadow);}
.imgbox img{display:block;max-width:100%;height:auto;margin:0 auto;}
.cap{font-size:13px;color:var(--muted);margin-top:10px;font-family:var(--mono);}

/* equation */
.eq{font-family:var(--mono);font-size:15px;background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:20px;overflow-x:auto;box-shadow:var(--shadow);margin-top:22px;line-height:2;}
.eq .lbl{color:var(--muted);} .eq .pos{color:var(--pos);} .eq .neg{color:var(--neg);}
.eq .tot{color:var(--accent);font-weight:700;}
.eq hr{border:none;border-top:1px dashed var(--line);margin:8px 0;}

/* interactive */
.lab{margin-top:26px;background:var(--panel);border:1px solid var(--line);border-radius:14px;
  box-shadow:var(--shadow);padding:22px;}
.readouts{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px;}
.readouts .cell{text-align:left;}
.readouts .k{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
.readouts .v{font-family:var(--mono);font-size:19px;font-weight:600;margin-top:3px;}
.v.pos{color:var(--pos)} .v.neg{color:var(--neg)} .v.acc{color:var(--accent)}
canvas{width:100%;height:300px;display:block;}
.controls{display:flex;align-items:center;gap:14px;margin-top:16px;flex-wrap:wrap;}
input[type=range]{flex:1;min-width:180px;accent-color:var(--accent);height:4px;}
button{font-family:var(--mono);font-size:13px;cursor:pointer;border:1px solid var(--accent);
  background:var(--accent);color:#fff;border-radius:8px;padding:9px 16px;font-weight:600;}
button:hover{filter:brightness(1.08)} button:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
button.ghost{background:transparent;color:var(--accent);}
.tcount{font-family:var(--mono);font-size:13px;color:var(--muted);min-width:96px;}
@media (prefers-color-scheme:dark){.readouts .v.pos{color:var(--pos)}}

/* spec grid */
.specs{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:22px;}
.spec{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:var(--shadow);}
.spec .k{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
.spec .v{font-family:var(--mono);font-size:24px;font-weight:600;margin-top:6px;}
.spec .s{font-size:12px;color:var(--muted);margin-top:4px;}

.foot{padding:40px 0 70px;color:var(--muted);font-family:var(--mono);font-size:12.5px;}
.theme-btn{position:fixed;top:16px;right:16px;z-index:10;width:38px;height:38px;border-radius:10px;
  border:1px solid var(--line);background:var(--panel);color:var(--ink);font-size:16px;padding:0;}
@media(max-width:560px){.readouts{grid-template-columns:repeat(2,1fr)}}
</style>
"""

BODY = r"""
<button class="theme-btn" id="themeBtn" aria-label="Toggle theme">◐</button>

<div class="wrap">
  <header class="hero">
    <div class="eyebrow">TSA volume model · gradient boosting</div>
    <h1>A forecast is just hundreds of tiny corrections, added up.</h1>
    <p class="lede">Your XGBoost model doesn't make one clever guess. It starts at the
      <b>average</b> and then stacks <b>200 small trees</b>, each one nudging the number a little
      closer to the truth. Here's the whole machine, end to end — using one real day from the test set
      (<span class="mono">__DATE__</span>).</p>
    <div class="flow">
      <div class="chip"><div class="k">Start · average</div><div class="v mono">__BASE__</div></div>
      <div class="arrow">+</div>
      <div class="chip"><div class="k">200 tree nudges</div><div class="v mono">__SUMC__</div></div>
      <div class="arrow">=</div>
      <div class="chip accent"><div class="k">Prediction</div><div class="v mono">__FINAL__</div></div>
      <div class="arrow">vs</div>
      <div class="chip"><div class="k">Actual</div><div class="v mono">__ACTUAL__</div></div>
    </div>
  </header>

  <section>
    <div class="eyebrow">01 · The unit</div>
    <h2>What a leaf node actually is</h2>
    <p>Every tree ends in <b>leaves</b>. A leaf is <b>not</b> the prediction — it's a single number, in
    volume units, that gets <b>added</b> to the running total. A row answers the yes/no questions down the
    tree, lands in exactly one leaf, and collects that leaf's value.</p>
    <p>Below is one tree on its own (trained at full strength so the leaves are easy to read). The baseline
    is the training mean; each leaf is a <b>deviation</b> from it. Land in the far-right leaf and you add
    <span class="mono">+527,687</span>; land in the far-left and you add <span class="mono">−913,999</span>.</p>
    <div class="imgbox"><img src="__IMG_SINGLE__" alt="A single depth-3 decision tree"></div>
    <p class="cap">one tree · root split on lag365_residual_anchor · leaves = volume added to the baseline</p>
  </section>

  <section>
    <div class="eyebrow">02 · The rule</div>
    <h2>Trees don't vote. They add.</h2>
    <p>This is the one idea that makes gradient boosting click. The trees are not averaged and they don't
    take a majority vote. Each tree predicts the <b>leftover error</b> of all the trees before it, and every
    leaf it produces is <b>summed</b> into the total:</p>
    <div class="eq">
      <span class="lbl">prediction =</span> <span class="tot">__BASE__</span>  <span class="lbl">(the average)</span><br>
      <span class="lbl">&nbsp;&nbsp;+ tree&nbsp;&nbsp;0 →</span> <span class="pos">+10,100</span><br>
      <span class="lbl">&nbsp;&nbsp;+ tree&nbsp;&nbsp;1 →</span> <span class="pos">&nbsp;+5,933</span><br>
      <span class="lbl">&nbsp;&nbsp;+ tree&nbsp;&nbsp;2 →</span> <span class="pos">+10,975</span><br>
      <span class="lbl">&nbsp;&nbsp;+ tree&nbsp;&nbsp;3 →</span> <span class="neg">&nbsp;−2,334</span>  <span class="lbl">← pulls back down</span><br>
      <span class="lbl">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋮&nbsp;&nbsp;&nbsp;&nbsp;(200 trees)</span><br>
      <hr>
      <span class="lbl">&nbsp;&nbsp;= </span><span class="tot">__FINAL__</span>  <span class="lbl">← final prediction</span>
    </div>
    <p style="margin-top:20px">Two things make each tree <b>deliberately weak</b>: they're shallow
    (<span class="mono">max_depth 4</span>) and every leaf is shrunk by the
    <b>learning rate</b> (<span class="mono">0.05</span>) before being added. So no single tree matters much —
    the model improves in hundreds of careful little steps. That's exactly what keeps it from overfitting.</p>
  </section>

  <section>
    <div class="eyebrow">03 · Watch it converge</div>
    <h2>Add the trees one at a time</h2>
    <p>Start at the average (the flat dashed line) and add trees left to right. Watch the prediction (teal)
    climb toward the actual value (the target line) and then settle. The first handful of trees do most of
    the work; the rest fine-tune.</p>
    <div class="lab">
      <div class="readouts">
        <div class="cell"><div class="k">Trees used</div><div class="v acc" id="rTrees">0</div></div>
        <div class="cell"><div class="k">This tree's nudge</div><div class="v" id="rNudge">—</div></div>
        <div class="cell"><div class="k">Prediction</div><div class="v" id="rPred">__BASE__</div></div>
        <div class="cell"><div class="k">Gap to actual</div><div class="v" id="rGap">—</div></div>
      </div>
      <canvas id="chart" width="1600" height="600"></canvas>
      <div class="controls">
        <button id="playBtn">▶ Play</button>
        <button class="ghost" id="resetBtn">Reset</button>
        <input type="range" id="slider" min="0" max="200" value="0">
        <span class="tcount mono" id="sliderLbl">0 / 200 trees</span>
      </div>
    </div>
  </section>

  <section>
    <div class="eyebrow">04 · What drives it</div>
    <h2>Which features the 200 trees lean on</h2>
    <p>Across every split in every tree, these features earned the most "gain" (error reduced). It's an
    <b>anchor-driven</b> model: last year's same-period level sets the ballpark, and recent volume plus
    holiday effects do the correcting — not raw calendar fields.</p>
    <div class="imgbox"><img src="__IMG_IMP__" alt="Feature importance by gain"></div>
    <p class="cap">total gain per feature · top 15</p>
    <div class="imgbox" style="margin-top:18px"><img src="__IMG_SHAP__" alt="SHAP summary"></div>
    <p class="cap">SHAP · average magnitude of each feature's push on a prediction (title text is a cosmetic
    leftover; the ranking is correct)</p>
  </section>

  <section>
    <div class="eyebrow">05 · The machine, in numbers</div>
    <h2>Model spec</h2>
    <div class="specs">
      <div class="spec"><div class="k">Trees</div><div class="v">200</div><div class="s">added in sequence</div></div>
      <div class="spec"><div class="k">Max depth</div><div class="v">4</div><div class="s">≤16 leaves each</div></div>
      <div class="spec"><div class="k">Learning rate</div><div class="v">0.05</div><div class="s">shrink per tree</div></div>
      <div class="spec"><div class="k">Total leaves</div><div class="v">__LEAVES__</div><div class="s">across all trees</div></div>
      <div class="spec"><div class="k">Test MAE</div><div class="v">__MAE__</div><div class="s">last 90 days</div></div>
      <div class="spec"><div class="k">Test MAPE</div><div class="v">__MAPE__%</div><div class="s">avg % error</div></div>
    </div>
    <p class="muted" style="margin-top:22px;font-size:14px">This is a standalone illustrative model
    (200 trees, depth 4) trained on <span class="mono">master_features.csv</span> to make the mechanics
    visible — it is not your production AutoGluon ensemble, though it leans on the same anchor features.</p>
  </section>

  <div class="foot wrap" style="padding-left:0;padding-right:0">
    base_score + Σ(leaf × 0.05) = prediction · generated from model_interpret/interpret_xgb.py
  </div>
</div>

<script>
const DATA = __DATA_JS__;
function fmt(n){return Math.round(n).toLocaleString('en-US');}

/* ---- theme toggle ---- */
const root=document.documentElement, tb=document.getElementById('themeBtn');
tb.addEventListener('click',()=>{
  const cur=root.getAttribute('data-theme')
    || (matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
  root.setAttribute('data-theme', cur==='dark'?'light':'dark');
  draw(state);
});
function css(v){return getComputedStyle(root).getPropertyValue(v).trim();}

/* ---- chart ---- */
const cv=document.getElementById('chart'), ctx=cv.getContext('2d');
const N=DATA.n_trees, base=DATA.base, actual=DATA.actual, run=DATA.running, contribs=DATA.contribs;
let allVals=run.concat([base,actual]);
let lo=Math.min(...allVals), hi=Math.max(...allVals);
let pad=(hi-lo)*0.12; lo-=pad; hi+=pad;
let state=0;

function resize(){
  const r=cv.getBoundingClientRect(), dpr=Math.min(devicePixelRatio||1,2);
  cv.width=r.width*dpr; cv.height=r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  return r;
}
function draw(k){
  const r=resize(), W=r.width, H=r.height;
  const mL=70,mR=16,mT=16,mB=28, pw=W-mL-mR, ph=H-mT-mB;
  ctx.clearRect(0,0,W,H);
  const X=i=>mL+(N<=1?0:(i/(N))*pw);
  const Y=v=>mT+ph-((v-lo)/(hi-lo))*ph;
  const ink=css('--ink'), muted=css('--muted'), line=css('--line'),
        acc=css('--accent'), pos=css('--pos');
  // grid + y labels
  ctx.font='11px ui-monospace,monospace'; ctx.textAlign='right'; ctx.textBaseline='middle';
  for(let g=0;g<=4;g++){
    const v=lo+(hi-lo)*g/4, y=Y(v);
    ctx.strokeStyle=line; ctx.lineWidth=1; ctx.beginPath();
    ctx.moveTo(mL,y); ctx.lineTo(W-mR,y); ctx.stroke();
    ctx.fillStyle=muted; ctx.fillText((v/1e6).toFixed(2)+'M', mL-8, y);
  }
  // actual target line
  ctx.strokeStyle=ink; ctx.globalAlpha=.55; ctx.setLineDash([5,4]); ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(mL,Y(actual)); ctx.lineTo(W-mR,Y(actual)); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha=1;
  ctx.fillStyle=ink; ctx.textAlign='left'; ctx.fillText('actual', mL+6, Y(actual)-10);
  // base dashed
  ctx.strokeStyle=muted; ctx.setLineDash([2,4]); ctx.beginPath();
  ctx.moveTo(mL,Y(base)); ctx.lineTo(W-mR,Y(base)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle=muted; ctx.fillText('average (start)', mL+6, Y(base)+14);
  // running prediction polyline up to k
  ctx.strokeStyle=acc; ctx.lineWidth=2.2; ctx.beginPath();
  ctx.moveTo(X(0),Y(base));
  for(let i=0;i<k;i++){ ctx.lineTo(X(i+1),Y(run[i])); }
  ctx.stroke();
  // area fill
  ctx.lineTo(X(k),Y(lo)); ctx.lineTo(X(0),Y(lo)); ctx.closePath();
  ctx.fillStyle=acc+'1f'; ctx.fill();
  // endpoint dot
  if(k>0){const ex=X(k),ey=Y(run[k-1]);
    ctx.fillStyle=acc; ctx.beginPath(); ctx.arc(ex,ey,4,0,7); ctx.fill();
    ctx.strokeStyle=css('--panel'); ctx.lineWidth=2; ctx.stroke();}
}
function setReadouts(k){
  document.getElementById('rTrees').textContent=k;
  document.getElementById('sliderLbl').textContent=k+' / '+N+' trees';
  const pred = k>0?run[k-1]:base;
  document.getElementById('rPred').textContent=fmt(pred);
  const gap=actual-pred;
  const gEl=document.getElementById('rGap');
  gEl.textContent=(gap>=0?'+':'−')+fmt(Math.abs(gap));
  gEl.className='v '+(Math.abs(gap)<60000?'acc':'');
  const nEl=document.getElementById('rNudge');
  if(k>0){const c=contribs[k-1];
    nEl.textContent=(c>=0?'+':'−')+fmt(Math.abs(c));
    nEl.className='v '+(c>=0?'pos':'neg');
  } else {nEl.textContent='—'; nEl.className='v';}
}
function render(k){state=k; document.getElementById('slider').value=k; draw(k); setReadouts(k);}

const slider=document.getElementById('slider');
slider.max=N;
slider.addEventListener('input',e=>{stopPlay();render(+e.target.value);});
let timer=null;
function stopPlay(){if(timer){clearInterval(timer);timer=null;document.getElementById('playBtn').textContent='▶ Play';}}
document.getElementById('playBtn').addEventListener('click',()=>{
  if(timer){stopPlay();return;}
  if(state>=N) render(0);
  document.getElementById('playBtn').textContent='❚❚ Pause';
  timer=setInterval(()=>{ let nx=state+2; if(nx>=N){nx=N; render(nx); stopPlay();} else render(nx); },45);
});
document.getElementById('resetBtn').addEventListener('click',()=>{stopPlay();render(0);});
addEventListener('resize',()=>draw(state));
render(0);
</script>
"""

repl = {
 "__DATE__": data["example_date"],
 "__BASE__": f'{round(data["base"]):,}',
 "__SUMC__": f'+{round(sum(data["contribs"])):,}',
 "__FINAL__": f'{round(data["final"]):,}',
 "__ACTUAL__": f'{round(data["actual"]):,}',
 "__LEAVES__": f'{data["total_leaves"]:,}',
 "__MAE__": f'{round(data["mae"]):,}',
 "__MAPE__": f'{data["mape"]:.2f}',
}
body = BODY
for k,v in repl.items():
    body = body.replace(k,v)
body = body.replace("__IMG_SINGLE__",IMG_SINGLE).replace("__IMG_IMP__",IMG_IMP)\
           .replace("__IMG_SHAP__",IMG_SHAP).replace("__DATA_JS__",DATA_JS)

inner = STYLE + body

# artifact body file (no doctype/head/body — harness wraps it)
open(f"{OUT}/artifact_body.html","w").write(inner)

# standalone local file (openable by double-click)
standalone = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
  "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
  "<title>How your XGBoost model works</title></head><body>"
  + inner + "</body></html>")
open(f"{ROOT}/model_interpret/xgboost_explainer.html","w").write(standalone)

print("wrote artifact_body.html (", len(inner), "bytes )")
print("wrote model_interpret/xgboost_explainer.html")
