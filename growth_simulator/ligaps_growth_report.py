"""Interactive offline HTML report for LiGAPS-Beef growth optimizer outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _json_safe_value(value: Any) -> Any:
    """Convert pandas/numpy scalars to JSON-safe Python values."""
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return value


LIVE_CHART_JS = r'''
(function(){
const COLORS=['#2563eb','#dc2626','#16a34a','#9333ea','#ea580c','#0891b2','#4f46e5','#be123c','#65a30d','#7c3aed','#0f766e','#b45309'];
function n(v){v=Number(v);return Number.isFinite(v)?v:null} function fmt(v){if(v===null||v===undefined)return '';v=Number(v);return Number.isFinite(v)?(Math.abs(v)>=1000?v.toLocaleString(undefined,{maximumFractionDigits:3}):Number(v.toPrecision(6)).toString()):String(v)}
function step(x){if(!Number.isFinite(x)||x<=0)return 1;let p=Math.pow(10,Math.floor(Math.log10(x))),q=x/p;return (q<=1?1:q<=2?2:q<=5?5:10)*p}
class RChart{constructor(canvas,cfg){this.canvas=canvas;this.ctx=canvas.getContext('2d');this.labels=(cfg.data&&cfg.data.labels)||[];this.datasets=((cfg.data&&cfg.data.datasets)||[]).map((d,i)=>({label:d.label||('series '+(i+1)),data:(d.data||[]).map(n),borderColor:d.borderColor||COLORS[i%COLORS.length],hidden:false}));this.s=0;this.e=Math.max(0,this.labels.length-1);this.hover=null;this.drag=false;this.legend=[];this.tip=this.mkTip();this.bind();this.draw()}
mkTip(){let t=this.canvas.parentElement.querySelector('.chart-tooltip');if(!t){t=document.createElement('div');t.className='chart-tooltip';this.canvas.parentElement.appendChild(t)}return t}
bind(){window.addEventListener('resize',()=>this.draw());this.canvas.addEventListener('mousemove',e=>this.move(e));this.canvas.addEventListener('mouseleave',()=>{this.hover=null;this.tip.style.display='none';this.draw()});this.canvas.addEventListener('click',e=>this.click(e));this.canvas.addEventListener('wheel',e=>this.wheel(e),{passive:false});this.canvas.addEventListener('mousedown',e=>this.down(e));window.addEventListener('mouseup',()=>{this.drag=false;this.canvas.style.cursor='crosshair'});window.addEventListener('mousemove',e=>this.pan(e));let b=this.canvas.parentElement.querySelector('[data-reset-chart]');if(b)b.addEventListener('click',()=>{this.s=0;this.e=Math.max(0,this.labels.length-1);this.draw()})}
bounds(){let r=this.canvas.getBoundingClientRect(),d=window.devicePixelRatio||1,W=Math.max(420,Math.floor(r.width*d)),H=Math.max(280,Math.floor(r.height*d));if(this.canvas.width!==W||this.canvas.height!==H){this.canvas.width=W;this.canvas.height=H}this.ctx.setTransform(d,0,0,d,0,0);W/=d;H/=d;return{W,H,L:66,R:24,T:52,B:64,PW:W-90,PH:H-116}}
range(){let a=Math.max(0,Math.floor(this.s)),b=Math.min(this.labels.length-1,Math.ceil(this.e));return [a,Math.max(a,b)]}
vr(a,b){let vals=[];this.datasets.forEach(ds=>{if(ds.hidden)return;for(let i=a;i<=b;i++){let v=ds.data[i];if(v!==null)vals.push(v)}});let mn=vals.length?Math.min(...vals):0,mx=vals.length?Math.max(...vals):1;if(mn===mx){mn-=1;mx+=1}let p=(mx-mn)*.08;mn-=p;mx+=p;let st=step((mx-mn)/5);return [Math.floor(mn/st)*st,Math.ceil(mx/st)*st,st]}
x(i,b){return b.L+((i-this.s)/Math.max(1e-9,this.e-this.s))*b.PW} y(v,b,mn,mx){return b.T+((mx-v)/Math.max(1e-9,mx-mn))*b.PH} idx(x,b){return Math.round(this.s+((x-b.L)/b.PW)*(this.e-this.s))}
move(ev){if(this.drag)return;let b=this.bounds(),r=this.canvas.getBoundingClientRect(),x=ev.clientX-r.left,y=ev.clientY-r.top;if(x<b.L||x>b.L+b.PW||y<b.T||y>b.T+b.PH){this.hover=null;this.tip.style.display='none';this.draw();return}this.hover=Math.max(0,Math.min(this.labels.length-1,this.idx(x,b)));let rows=this.datasets.filter(ds=>!ds.hidden).map(ds=>`<div><span style="color:${ds.borderColor};font-weight:700">●</span> ${ds.label}: <b>${fmt(ds.data[this.hover])}</b></div>`).join('');this.tip.innerHTML=`<div><b>day:</b> ${this.labels[this.hover]}</div>${rows}`;let pr=this.canvas.parentElement.getBoundingClientRect();this.tip.style.display='block';this.tip.style.left=Math.min(pr.width-250,Math.max(8,ev.clientX-pr.left+14))+'px';this.tip.style.top=Math.max(8,ev.clientY-pr.top+14)+'px';this.draw()}
click(ev){let r=this.canvas.getBoundingClientRect(),x=ev.clientX-r.left,y=ev.clientY-r.top;for(let h of this.legend){if(x>=h.x&&x<=h.x+h.w&&y>=h.y&&y<=h.y+h.h){this.datasets[h.i].hidden=!this.datasets[h.i].hidden;this.draw();return}}}
wheel(ev){ev.preventDefault();if(this.labels.length<2)return;let b=this.bounds(),r=this.canvas.getBoundingClientRect(),x=ev.clientX-r.left,anchor=this.s+((x-b.L)/b.PW)*(this.e-this.s),f=ev.deltaY<0?.82:1.22,span=Math.max(8,(this.e-this.s)*f),ns=anchor-(anchor-this.s)*f,ne=ns+span;if(ns<0){ne-=ns;ns=0}if(ne>this.labels.length-1){ns-=ne-(this.labels.length-1);ne=this.labels.length-1}this.s=Math.max(0,ns);this.e=Math.min(this.labels.length-1,ne);this.draw()}
down(ev){let b=this.bounds(),r=this.canvas.getBoundingClientRect(),x=ev.clientX-r.left,y=ev.clientY-r.top;if(x<b.L||x>b.L+b.PW||y<b.T||y>b.T+b.PH)return;this.drag=true;this.dx=x;this.ds=this.s;this.de=this.e;this.canvas.style.cursor='grabbing'}
pan(ev){if(!this.drag)return;let b=this.bounds(),r=this.canvas.getBoundingClientRect(),x=ev.clientX-r.left,shift=-(x-this.dx)/b.PW*(this.de-this.ds),ns=this.ds+shift,ne=this.de+shift;if(ns<0){ne-=ns;ns=0}if(ne>this.labels.length-1){ns-=ne-(this.labels.length-1);ne=this.labels.length-1}this.s=ns;this.e=ne;this.draw()}
draw(){let ctx=this.ctx,b=this.bounds();ctx.clearRect(0,0,b.W,b.H);ctx.font='12px Arial';let [a,z]=this.range(),[mn,mx,st]=this.vr(a,z);ctx.strokeStyle='#e5e7eb';ctx.fillStyle='#475569';ctx.textAlign='right';ctx.textBaseline='middle';for(let yv=mn;yv<=mx+st*.5;yv+=st){let yy=this.y(yv,b,mn,mx);ctx.beginPath();ctx.moveTo(b.L,yy);ctx.lineTo(b.L+b.PW,yy);ctx.stroke();ctx.fillText(fmt(yv),b.L-8,yy)}ctx.strokeStyle='#111827';ctx.beginPath();ctx.moveTo(b.L,b.T);ctx.lineTo(b.L,b.T+b.PH);ctx.lineTo(b.L+b.PW,b.T+b.PH);ctx.stroke();ctx.textAlign='center';ctx.textBaseline='top';let tc=Math.min(8,z-a+1);for(let t=0;t<tc;t++){let i=Math.round(a+t*(z-a)/Math.max(1,tc-1));ctx.fillText(String(this.labels[i]),this.x(i,b),b.T+b.PH+11)}this.datasets.forEach(ds=>{if(ds.hidden)return;ctx.strokeStyle=ds.borderColor;ctx.lineWidth=2;ctx.beginPath();let on=false;for(let i=a;i<=z;i++){let v=ds.data[i];if(v===null){on=false;continue}let xx=this.x(i,b),yy=this.y(v,b,mn,mx);if(!on){ctx.moveTo(xx,yy);on=true}else ctx.lineTo(xx,yy)}ctx.stroke();if(this.hover!==null&&this.hover>=a&&this.hover<=z){let v=ds.data[this.hover];if(v!==null){ctx.fillStyle=ds.borderColor;ctx.beginPath();ctx.arc(this.x(this.hover,b),this.y(v,b,mn,mx),3.5,0,Math.PI*2);ctx.fill()}}});if(this.hover!==null&&this.hover>=a&&this.hover<=z){let hx=this.x(this.hover,b);ctx.strokeStyle='#64748b';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(hx,b.T);ctx.lineTo(hx,b.T+b.PH);ctx.stroke();ctx.setLineDash([])}this.legend=[];let lx=b.L,ly=18;ctx.textAlign='left';ctx.textBaseline='middle';this.datasets.forEach((ds,i)=>{let tw=ctx.measureText(ds.label).width,w=Math.min(220,tw+30);if(lx+w>b.W-b.R){lx=b.L;ly+=20}ctx.globalAlpha=ds.hidden?.35:1;ctx.fillStyle=ds.borderColor;ctx.fillRect(lx,ly-5,10,10);ctx.fillStyle='#111827';ctx.fillText(ds.label,lx+15,ly);ctx.globalAlpha=1;this.legend.push({x:lx,y:ly-9,w,h:18,i});lx+=w+12});ctx.textAlign='right';ctx.fillStyle='#64748b';ctx.fillText(`view: ${a+1}-${z+1} / ${this.labels.length}`,b.W-b.R,b.H-18)}}
window.Chart=function(ctx,cfg){return new RChart(ctx.canvas?ctx.canvas:ctx,cfg)};
})();
'''

HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>__TITLE__</title>
<style>
body{margin:0;font-family:Arial,Helvetica,sans-serif;background:#f8fafc;color:#0f172a}header{background:#0f172a;color:white;padding:24px 32px}header h1{margin:0 0 6px 0;font-size:24px}header p{margin:0;color:#cbd5e1}main{padding:24px 32px 48px;max-width:1480px;margin:0 auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:18px}.card{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px;box-shadow:0 1px 2px rgba(15,23,42,.06)}.card h2{margin:0 0 6px 0;font-size:18px}.chart-card{position:relative}.chart-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px}.chart-actions button{border:1px solid #cbd5e1;background:#f8fafc;border-radius:8px;padding:6px 10px;cursor:pointer;color:#0f172a}.chart-actions button:hover{background:#e2e8f0}canvas{width:100%;height:380px;cursor:crosshair;display:block}.chart-tooltip{display:none;position:absolute;pointer-events:none;z-index:5;min-width:210px;max-width:280px;background:rgba(15,23,42,.94);color:white;border-radius:10px;padding:10px 12px;font-size:12px;line-height:1.45;box-shadow:0 10px 22px rgba(15,23,42,.25)}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:18px}.pill{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:12px 14px}.pill .k{color:#64748b;font-size:12px}.pill .v{font-weight:700;margin-top:4px}.table-toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:10px 0 12px}.table-toolbar input{width:min(420px,100%);border:1px solid #cbd5e1;border-radius:10px;padding:9px 11px}.table-wrap{overflow:auto;max-height:760px;border:1px solid #e2e8f0;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:12px;background:white}th,td{border-bottom:1px solid #e2e8f0;padding:7px 9px;text-align:right;white-space:nowrap}th{position:sticky;top:0;background:#f1f5f9;z-index:2;color:#334155}th:first-child,td:first-child{text-align:left}.last-panel{margin-top:20px;background:#0f172a;color:white;border-radius:16px;padding:20px}.last-panel h2{margin-top:0}.last-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}.last-item{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);border-radius:12px;padding:11px}.last-item .k{color:#cbd5e1;font-size:12px;overflow-wrap:anywhere}.last-item .v{font-weight:700;margin-top:5px;overflow-wrap:anywhere}.note{color:#64748b;font-size:13px;line-height:1.5}
</style><script>__LIVE_CHART_JS__</script></head><body>
<header><h1>__TITLE__</h1><p>Self-contained offline HTML report. All graph panels are live JavaScript canvas charts: hover for values, click legend labels to hide/show, use mouse wheel to zoom, drag to pan, and Reset to restore the full horizon.</p></header>
<main><section class="meta" id="meta"></section><section class="grid">
<div class="card chart-card"><div class="chart-head"><div><h2>Body weight trajectory</h2><p class="note">TBW and genetic potential TBW.</p></div><div class="chart-actions"><button data-reset-chart>Reset</button></div></div><canvas id="chart_tbw"></canvas></div>
<div class="card chart-card"><div class="chart-head"><div><h2>Feed intake and available feeds</h2><p class="note">Daily intake and available dry-matter feeds.</p></div><div class="chart-actions"><button data-reset-chart>Reset</button></div></div><canvas id="chart_feed"></canvas></div>
<div class="card chart-card"><div class="chart-head"><div><h2>Climate temperature inputs</h2><p class="note">Minimum and maximum daily temperature.</p></div><div class="chart-actions"><button data-reset-chart>Reset</button></div></div><canvas id="chart_climate_temp"></canvas></div>
<div class="card chart-card"><div class="chart-head"><div><h2>Radiation, wind, vapor pressure, rain</h2><p class="note">Daily growth-optimizer climate proxies.</p></div><div class="chart-actions"><button data-reset-chart>Reset</button></div></div><canvas id="chart_climate_other"></canvas></div>
<div class="card chart-card"><div class="chart-head"><div><h2>Energy and heat outputs</h2><p class="note">Metabolisable energy uptake and heat production.</p></div><div class="chart-actions"><button data-reset-chart>Reset</button></div></div><canvas id="chart_energy"></canvas></div>
<div class="card chart-card"><div class="chart-head"><div><h2>Beef production</h2><p class="note">Daily simulated deboned-carcass mass exported from muscle tissue + intramuscular fat + miscellaneous fat; original slaughter-event output is preserved as beef_event_kg.</p></div><div class="chart-actions"><button data-reset-chart>Reset</button></div></div><canvas id="chart_beef"></canvas></div>
<div class="card chart-card"><div class="chart-head"><div><h2>Defining and limiting factors for growth</h2><p class="note">R-style factor rows: protein=0.5, energy=1.5, digestion cap.=2.5, cold stress=3.5, heat stress=4.5, genotype=5.5. Inactive/missing values are exported and plotted as 0.</p></div><div class="chart-actions"><button data-reset-chart>Reset</button></div></div><canvas id="chart_factors"></canvas></div>
</section><section class="card" style="margin-top:20px"><h2>Complete daily output table</h2><p class="note">All columns from the generated CSV are shown below for the full observed horizon.</p><div class="table-toolbar"><input id="table_filter" placeholder="Filter rows across all columns..."/><span id="table_count" class="note"></span></div><div class="table-wrap"><table id="daily_table"></table></div></section>
<section class="last-panel"><h2>Last-day output panel</h2><p class="note" style="color:#cbd5e1">Every output field from the final simulated day is listed here.</p><div class="last-grid" id="last_panel"></div></section></main>
<script>
const DATA=__DATA_JSON__; const COLUMNS=__COLUMNS_JSON__; const LAST=__LAST_JSON__; const SUMMARY=__SUMMARY_JSON__;
function num(x){const n=Number(x);return Number.isFinite(n)?n:null}function fmt(x){if(x===null||x===undefined)return '';const n=Number(x);if(Number.isFinite(n))return Math.abs(n)>=1000?n.toLocaleString(undefined,{maximumFractionDigits:3}):Number(n.toPrecision(6)).toString();return String(x)}
function labels(){return DATA.map((r,i)=>r.fattening_day??r.doy??(i+1))}function series(c){return DATA.map(r=>num(r[c]))}function ds(label,col,color){return{label:label,data:series(col),borderColor:color,fill:false}}function makeChart(id,datasets){return new Chart(document.getElementById(id),{type:'line',data:{labels:labels(),datasets:datasets},options:{responsive:true,animation:false}})}function available(cols){return cols.filter(c=>COLUMNS.includes(c))}
const meta=document.getElementById('meta');const metaItems=[['rows',DATA.length],['case_id',LAST.case_id??SUMMARY.case_id],['scale',LAST.scale??SUMMARY.scale],['breed',LAST.breed??SUMMARY.breed],['diet',LAST.diet??SUMMARY.diet],['last_fattening_day',LAST.fattening_day],['last_doy',LAST.doy]];meta.innerHTML=metaItems.map(([k,v])=>`<div class="pill"><div class="k">${k}</div><div class="v">${fmt(v)}</div></div>`).join('');
makeChart('chart_tbw',available(['tbw_kg','genetic_potential_tbw_kg']).map((c,i)=>ds(c,c,['#2563eb','#dc2626'][i])));makeChart('chart_feed',available(['feed_intake_kg_dm_day','feed1_available_kg_dm_day','feed2_available_kg_dm_day','feed3_available_kg_dm_day','feed4_available_kg_dm_day']).map((c,i)=>ds(c,c,['#16a34a','#9333ea','#ea580c','#0891b2','#4f46e5'][i])));makeChart('chart_climate_temp',available(['mint','maxt']).map((c,i)=>ds(c,c,['#0ea5e9','#ef4444'][i])));makeChart('chart_climate_other',available(['rad','vpr','wind','rain','aha','okta']).map((c,i)=>ds(c,c,['#f59e0b','#06b6d4','#84cc16','#3b82f6','#a855f7','#64748b'][i])));makeChart('chart_energy',available(['me_uptake_mj_day','heat_production']).map((c,i)=>ds(c,c,['#14b8a6','#f97316'][i])));makeChart('chart_beef',available(['beef_production_kg']).map((c,i)=>ds(c,c,['#be123c'][i])));makeChart('chart_factors',available(['protein','energy','digestion_cap','cold_stress','heat_stress','genotype']).map((c,i)=>ds(c.replace('_',' '),c,['#7c3aed','#eab308','#22c55e','#0ea5e9','#ef4444','#111827'][i])));
const table=document.getElementById('daily_table'),count=document.getElementById('table_count');function renderTable(rows){const head='<thead><tr>'+COLUMNS.map(c=>`<th>${c}</th>`).join('')+'</tr></thead>';const body='<tbody>'+rows.map(r=>'<tr>'+COLUMNS.map(c=>`<td>${fmt(r[c])}</td>`).join('')+'</tr>').join('')+'</tbody>';table.innerHTML=head+body;count.textContent=`${rows.length} / ${DATA.length} rows`}renderTable(DATA);document.getElementById('table_filter').addEventListener('input',e=>{const q=String(e.target.value||'').toLowerCase().trim();if(!q){renderTable(DATA);return}renderTable(DATA.filter(r=>COLUMNS.some(c=>String(r[c]??'').toLowerCase().includes(q))))});
document.getElementById('last_panel').innerHTML=COLUMNS.map(c=>`<div class="last-item"><div class="k">${c}</div><div class="v">${fmt(LAST[c])}</div></div>`).join('');
</script></body></html>
'''


def generate_growth_html_report(
    daily: pd.DataFrame,
    summary: pd.DataFrame | dict | None,
    output_dir: str | Path,
    *,
    case_id: int = 1,
    title: str | None = None,
) -> Path:
    """Generate a self-contained interactive HTML report in ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"growth_optimizer_report_case_{int(case_id)}.html"

    df = daily.copy()
    factor_columns = ["genotype", "heat_stress", "cold_stress", "digestion_cap", "energy", "protein"]
    for _factor_col in factor_columns:
        if _factor_col in df.columns:
            df[_factor_col] = pd.to_numeric(df[_factor_col], errors="coerce").fillna(0.0)
    columns = [str(c) for c in df.columns]
    records = [{str(col): _json_safe_value(row[col]) for col in df.columns} for _, row in df.iterrows()]
    last = {str(col): _json_safe_value(df.iloc[-1][col]) for col in df.columns} if len(df) else {}
    if isinstance(summary, pd.DataFrame):
        summary_obj = {str(col): _json_safe_value(summary.iloc[0][col]) for col in summary.columns} if len(summary) else {}
    elif isinstance(summary, dict):
        summary_obj = {str(k): _json_safe_value(v) for k, v in summary.items()}
    else:
        summary_obj = {}

    report_title = title or f"LiGAPS-Beef Growth Optimizer Report - Case {int(case_id)}"
    html = HTML_TEMPLATE
    replacements = {
        "__TITLE__": report_title,
        "__LIVE_CHART_JS__": LIVE_CHART_JS,
        "__DATA_JSON__": json.dumps(records, ensure_ascii=False),
        "__COLUMNS_JSON__": json.dumps(columns, ensure_ascii=False),
        "__LAST_JSON__": json.dumps(last, ensure_ascii=False),
        "__SUMMARY_JSON__": json.dumps(summary_obj, ensure_ascii=False),
    }
    for key, value in replacements.items():
        html = html.replace(key, value)
    report_path.write_text(html, encoding="utf-8")
    return report_path
