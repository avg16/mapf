const $ = id => document.getElementById(id);
const canvas = $('world'), ctx = canvas.getContext('2d');
const descriptions = {
  hybrid:'The learned gate chooses shield or CS-PIBT. LaCAM takes over after 16 steps without a new best distance.',
  shield:'The trained policy proposes moves. Conflicting robots wait. Safe moves can still lead to permanent jams.',
  pibt:'Distance-based priority inheritance and backtracking coordinates every step, without the neural policy.',
  cspibt:'The trained policy guides candidate ordering; PIBT coordinates every simultaneous move.',
  gate:'The learned gate selects shield or CS-PIBT. No LaCAM fallback: see whether robots can finish.',
  lacam:'Search over full joint configurations using plain PIBT to generate successors. Replay the complete found plan.'
};
let data=null, frame=0, selected=null, playing=false, last=0, busy=false;
let comparisonRows=[], comparisonSeed=null, comparisonCount=null;
const modes=['shield','pibt','cspibt','gate','hybrid','lacam'];
const names={shield:'Policy + shield',pibt:'Plain PIBT',cspibt:'CS-PIBT always',gate:'Learned gate',hybrid:'Learned gate + LaCAM',lacam:'Full LaCAM'};
const timeLabel = seconds => seconds < 1 ? `${(seconds*1000).toFixed(1)} ms` : `${seconds.toFixed(3)} s`;
function setBusy(value){busy=value;for(const id of ['generate','newmap','compare','mode','seed','count'])$(id).disabled=value;for(const button of document.querySelectorAll('.replay-button'))button.disabled=value;}
const colour = i => `hsl(${(i*137.508+145)%360} 53% 43%)`;
function pause(){playing=false;$('play').textContent='▶';}
function update(){
  if(!data)return;
  const f=data.frames[frame];
  $('timeline').value=frame;$('clock').textContent=`${frame} / ${data.frames.length-1}`;
  $('arrived').textContent=`${f.arrived} / ${data.count}`;$('remaining').textContent=f.remaining;
  $('search-calls').textContent=f.search_calls;$('search-total').textContent=`of ${data.result.n_cspibt} full-run PIBT / CS-PIBT calls`;
  $('lacam-calls').textContent=f.lacam_calls;$('lacam-total').textContent=`of ${data.result.n_lacam} full-run LaCAM calls`;
  $('compute-time').textContent=timeLabel(f.compute_seconds);$('total-time').textContent=timeLabel(data.result.runtime);
  $('quality').textContent=data.result.subopt===null?'Unsolved':`${data.result.subopt.toFixed(2)}×`;
  $('flowtime').textContent=data.result.success?`Full-run flowtime ${data.result.solution_flowtime} · makespan ${data.result.solution_makespan}`:'Quality is not reported for truncated runs';
  $('nodes').textContent=f.lacam_nodes;$('safety').textContent=`of ${data.result.lacam_nodes} full-run nodes · 0 collisions verified`;
  $('score').textContent=f.q===null?'—':f.q.toFixed(3);$('score-bar').style.width=`${(f.q??0)*100}%`;
  $('threshold').textContent=f.q===null?'No gate decision for this move':`Search threshold ${data.threshold.toFixed(3)}`;
  $('tier').textContent=f.method;
  const ended=frame===data.frames.length-1;
  $('run-status').textContent=ended?(data.result.success?'All robots reached their goals':(data.mode==='lacam'?'Search budget exhausted · no plan':'Step limit reached · robots remain')):(playing?'Simulation playing':'Paused · explore the grid');
  if(selected!==null){const p=f.positions[selected],g=data.goals[selected];$('agent-info').textContent=`Robot ${selected+1} · cell (${Math.floor(p/32)}, ${p%32}) · goal (${Math.floor(g/32)}, ${g%32})${p===g?' · arrived':''}`;}
  const vals=data.frames.slice(0,frame+1).map(f=>f.remaining), max=Math.max(data.frames[0].remaining,1),end=Math.max(data.frames.length-1,1);
  const pts=vals.map((v,i)=>`${i/end*196+2},${70-v/max*65}`).join(' ');
  $('chart').innerHTML=`<line x1="0" y1="71" x2="200" y2="71" stroke="#dce3dc"/><polyline points="${pts}" fill="none" stroke="#34815b" stroke-width="2"/>`;
  draw();
}
function draw(){
  const size=canvas.width,cell=size/32;
  ctx.clearRect(0,0,size,size);ctx.fillStyle='#f2f5ed';ctx.fillRect(0,0,size,size);
  if(!data)return;
  for(let r=0;r<32;r++)for(let c=0;c<32;c++){
    if(!data.free[r][c]){ctx.fillStyle='#c7d0bf';ctx.fillRect(c*cell+1,r*cell+1,cell-2,cell-2);}
  }
  ctx.strokeStyle='#e4e9df';ctx.lineWidth=.7;
  ctx.beginPath();for(let i=0;i<=32;i++){ctx.moveTo(i*cell,0);ctx.lineTo(i*cell,size);ctx.moveTo(0,i*cell);ctx.lineTo(size,i*cell);}ctx.stroke();
  const positions=data.frames[frame].positions;
  if(selected!==null&&$('fov').checked){
    const p=positions[selected],r=Math.floor(p/32),c=p%32;
    ctx.save();ctx.fillStyle='#368b6916';ctx.strokeStyle='#368b6980';ctx.setLineDash([5,4]);ctx.fillRect((c-5)*cell,(r-5)*cell,11*cell,11*cell);ctx.strokeRect((c-5)*cell,(r-5)*cell,11*cell,11*cell);ctx.restore();
  }
  if($('trails').checked){
    positions.forEach((_,i)=>{if(selected!==null&&selected!==i)return;ctx.beginPath();ctx.strokeStyle=colour(i);ctx.globalAlpha=.25;ctx.lineWidth=selected===i?4:2;for(let t=Math.max(0,frame-12);t<=frame;t++){const p=data.frames[t].positions[i],x=(p%32+.5)*cell,y=(Math.floor(p/32)+.5)*cell;if(t===Math.max(0,frame-12))ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();});ctx.globalAlpha=1;
  }
  if($('goals').checked)data.goals.forEach((g,i)=>{ctx.globalAlpha=selected!==null&&selected!==i?.2:.65;ctx.strokeStyle=colour(i);ctx.lineWidth=selected===i?3:1.5;const x=(g%32+.5)*cell,y=(Math.floor(g/32)+.5)*cell;ctx.strokeRect(x-cell*.24,y-cell*.24,cell*.48,cell*.48);});
  ctx.globalAlpha=1;
  positions.forEach((p,i)=>{const x=(p%32+.5)*cell,y=(Math.floor(p/32)+.5)*cell;ctx.globalAlpha=selected!==null&&selected!==i?.4:1;ctx.beginPath();ctx.arc(x,y,cell*.32,0,Math.PI*2);ctx.fillStyle=colour(i);ctx.fill();ctx.lineWidth=1.5;ctx.strokeStyle='#fff';ctx.stroke();if(p===data.goals[i]){ctx.fillStyle='white';ctx.beginPath();ctx.arc(x,y,cell*.10,0,Math.PI*2);ctx.fill();}if(selected===i){ctx.beginPath();ctx.arc(x,y,cell*.46,0,Math.PI*2);ctx.lineWidth=2;ctx.strokeStyle='#21352d';ctx.stroke();}});ctx.globalAlpha=1;
}
async function generate(){
  if(busy)return;
  const seed=Number($('seed').value),count=Number($('count').value),mode=$('mode').value;
  if(!Number.isInteger(seed)||seed<0||seed>4294967295){alert('Enter a map seed between 0 and 4294967295.');return;}
  setBusy(true);pause();$('loading').hidden=false;$('loading').textContent='Computing a real trajectory…';
  try{
    const response=await fetch('/api/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed,count,mode})});
    const body=await response.json();if(!response.ok)throw Error(body.error||'Simulation failed');
    data=body;frame=0;selected=null;$('agent-info').textContent='Select any robot on the grid.';$('timeline').max=data.frames.length-1;
    $('summary').textContent=`Full run: ${data.result.success?'solved':'not solved'} · ${data.frames.length-1} moves · ${data.result.n_cspibt} PIBT / CS-PIBT calls · ${data.result.n_lacam} LaCAM calls · ${data.result.runtime.toFixed(2)} s compute. Playback speed is independent of computation.`;
    $('loading').hidden=true;update();
  }catch(error){$('loading').textContent=`Could not generate simulation: ${error.message}. Try another map or controller.`;$('run-status').textContent='Simulation unavailable';}
  finally{setBusy(false);}
}
function addComparisonRow(row){
  const r=row.result, tr=document.createElement('tr');
  const cells=[names[row.mode],`${row.solved_runs} / ${row.repeats}`,`${r.arrived} / ${row.count}`,
    `${(row.runtime_median*1000).toFixed(1)} [${(row.runtime_min*1000).toFixed(1)}–${(row.runtime_max*1000).toFixed(1)}]`,
    r.n_cspibt,r.n_lacam,r.lacam_nodes,`${(r.search_rate*100).toFixed(1)}%`,r.solution_flowtime??'—',
    r.subopt===null?'—':`${r.subopt.toFixed(2)}×`,r.solution_makespan??'—'];
  cells.forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.appendChild(td);});
  if(!row.stable_outcomes){tr.cells[0].textContent+=' *';tr.title='Outcomes varied across repeats; path and call metrics show the first run.';}
  const td=document.createElement('td'),button=document.createElement('button');button.textContent='View ↗';button.className='secondary replay-button';button.disabled=busy;
  button.onclick=()=>{if(busy)return;$('seed').value=row.seed;$('count').value=row.count;$('mode').value=row.mode;$('description').textContent=descriptions[row.mode];generate();$('world').scrollIntoView({behavior:'smooth',block:'center'});};
  td.appendChild(button);tr.appendChild(td);$('comparison-body').appendChild(tr);
}
async function compare(){
  if(busy)return;
  const seed=Number($('seed').value),count=Number($('count').value);
  if(!Number.isInteger(seed)||seed<0||seed>4294967295){alert('Enter a valid map seed.');return;}
  pause();setBusy(true);comparisonRows=[];comparisonSeed=seed;comparisonCount=count;$('export').disabled=true;$('comparison-body').replaceChildren();
  $('comparison-section').scrollIntoView({behavior:'smooth',block:'start'});
  try{
    for(const [index,mode] of modes.entries()){
      $('comparison-status').textContent=`Seed ${seed} · ${count} robots · benchmarking ${names[mode]} (${index+1}/6), three fresh runs…`;
      const response=await fetch('/api/benchmark',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed,count,mode})});
      const row=await response.json();if(!response.ok)throw Error(row.error||'Benchmark failed');
      comparisonRows.push(row);addComparisonRow(row);
    }
    const hybrid=comparisonRows.find(r=>r.mode==='hybrid'),always=comparisonRows.find(r=>r.mode==='cspibt');
    const saved=always.result.n_cspibt>0?100*(1-hybrid.result.n_cspibt/always.result.n_cspibt):null;
    $('comparison-status').textContent=`Seed ${seed} · ${count} robots · completed 18 fresh runs. ${saved===null?'':`Hybrid used ${Math.abs(saved).toFixed(1)}% ${saved>=0?'fewer':'more'} external CS-PIBT calls than always-search on this instance. `}Compare success, quality and runtime together; full LaCAM calls include additional internal search.${comparisonRows.some(r=>!r.stable_outcomes)?' * Outcomes varied across repeats; non-time metrics show the first repeat.':''}`;
  }catch(error){$('comparison-status').textContent=`Benchmark stopped: ${error.message}. Completed rows are retained; retry to run all controllers.`;}
  finally{setBusy(false);$('export').disabled=comparisonRows.length===0;}
}
function exportComparison(){
  const header=['seed','robots','controller','repeats','solved_runs','stable_outcomes','runtime_median_ms','runtime_min_ms','runtime_max_ms','arrived_first_run','one_step_search_calls','lacam_calls','lacam_nodes','search_calls_per_replay_step','replay_steps','solution_flowtime','solution_makespan','distance_lower_bound','quality_ratio','verified_collisions'];
  const rows=comparisonRows.map(row=>{const r=row.result;return [row.seed,row.count,row.mode,row.repeats,row.solved_runs,row.stable_outcomes,row.runtime_median*1000,row.runtime_min*1000,row.runtime_max*1000,r.arrived,r.n_cspibt,r.n_lacam,r.lacam_nodes,r.search_rate,r.steps,r.solution_flowtime,r.solution_makespan,r.sum_lb,r.subopt,r.collisions];});
  const csv=[header,...rows].map(row=>row.map(v=>v??'').join(',')).join('\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=`mapf-comparison-${comparisonSeed}-${comparisonCount}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$('compare').onclick=compare;$('export').onclick=exportComparison;
$('generate').onclick=generate;
$('newmap').onclick=()=>{$('seed').value=Math.floor(Math.random()*1000000);generate();};
$('mode').onchange=()=>{$('description').textContent=descriptions[$('mode').value];};
$('play').onclick=()=>{if(!data||busy)return;if(playing)pause();else{if(frame===data.frames.length-1)frame=0;playing=true;last=performance.now();$('play').textContent='Ⅱ';}update();};
$('restart').onclick=()=>{pause();frame=0;update();};
$('step').onclick=()=>{if(!data)return;pause();frame=Math.min(frame+1,data.frames.length-1);update();};
$('timeline').oninput=()=>{pause();frame=Number($('timeline').value);update();};
for(const id of ['fov','trails','goals'])$(id).onchange=draw;
canvas.onclick=e=>{if(!data)return;const rect=canvas.getBoundingClientRect(),c=Math.floor((e.clientX-rect.left)/rect.width*32),r=Math.floor((e.clientY-rect.top)/rect.height*32);const i=data.frames[frame].positions.indexOf(r*32+c);selected=i<0?null:i;if(selected===null)$('agent-info').textContent='Select any robot on the grid.';update();};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON','TEXTAREA'].includes(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click();}else if(e.code==='ArrowRight'){$('step').click();}});
function tick(now){if(playing&&data&&now-last>=1000/Number($('speed').value)){last=now;frame=Math.min(frame+1,data.frames.length-1);if(frame===data.frames.length-1)pause();update();}requestAnimationFrame(tick);}
$('description').textContent=descriptions[$('mode').value];requestAnimationFrame(tick);generate();
