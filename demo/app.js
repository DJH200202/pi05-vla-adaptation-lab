'use strict';
const D=window.DEMO_DATA, colors={Base:'#9aa5b5',A:'#9b76d3',B:'#e19d38',C:'#128b89',D:'#366bd3'};
const names={Base:'原始基础权重',A:'仅动作专家 LoRA',B:'语言模块 + 动作专家 LoRA',C:'双 LoRA + 视觉 / 外部投影等模块',D:'语言模块 LoRA + 视觉 / 投影 / 动作专家原权重'};
const el=id=>document.getElementById(id);
el('scope-table').innerHTML=D.groups.map(g=>`<tr><td><span class="badge" style="background:${colors[g]}">${g}</span></td><td>${names[g]}</td><td>${g==='Base'?'0':(D.training[g].parameters.trainable/1e6).toFixed(2)+' M'}</td><td><b>${(D.evaluation[g].success_rate*100).toFixed(1)}%</b> <small>(${D.evaluation[g].successes}/500)</small></td></tr>`).join('');
el('loss-table').innerHTML=Object.entries(D.training).map(([g,t])=>`<tr><td>${g}</td><td>${t.final_loss.toFixed(6)}</td><td>${t.last100_mean.toFixed(6)}</td><td>${t.last300_mean.toFixed(6)}</td></tr>`).join('');
el('task-list').innerHTML=D.tasks.map(t=>`<li value="${t.id}">${t.prompt}</li>`).join('');
const visible=new Set(['A','B','C','D']);let chosen=3000;
for(const g of visible){const b=document.createElement('button');b.textContent=g;b.style.setProperty('--g',colors[g]);b.className='active';b.setAttribute('aria-pressed','true');b.onclick=()=>{if(visible.has(g))visible.delete(g);else visible.add(g);b.classList.toggle('active',visible.has(g));b.setAttribute('aria-pressed',String(visible.has(g)));draw();};el('group-toggles').append(b);}
const canvas=el('loss-chart');
function draw(){const box=canvas.getBoundingClientRect(),ratio=devicePixelRatio||1,w=box.width,h=box.height;canvas.width=w*ratio;canvas.height=h*ratio;const c=canvas.getContext('2d');c.scale(ratio,ratio);const p={l:52,r:15,t:15,b:38},iw=w-p.l-p.r,ih=h-p.t-p.b,top=.13;const x=s=>p.l+(s-1)/2999*iw,y=v=>p.t+(top-v)/top*ih;c.clearRect(0,0,w,h);c.font='11px -apple-system, sans-serif';c.fillStyle='#627186';c.strokeStyle='#e5e9ed';c.lineWidth=1;
for(let v=0;v<=.12001;v+=.02){c.beginPath();c.moveTo(p.l,y(v));c.lineTo(w-p.r,y(v));c.stroke();c.fillText(v.toFixed(2),7,y(v)+4);}for(let s=0;s<=3000;s+=500){c.fillText(String(s),x(Math.max(s,1))-12,h-12);}c.fillText('loss',7,10);
function line(arr,color,alpha,width){c.strokeStyle=color;c.globalAlpha=alpha;c.lineWidth=width;c.beginPath();arr.forEach((v,i)=>{if(i===0)c.moveTo(x(i+1),y(v));else c.lineTo(x(i+1),y(v));});c.stroke();c.globalAlpha=1;}
for(const g of visible){if(el('raw-toggle').checked)line(D.training[g].loss,colors[g],.2,.7);if(el('ema-toggle').checked)line(D.training[g].loss_ema,colors[g],1,1.8);}
c.strokeStyle='#8492a3';c.setLineDash([3,4]);c.beginPath();c.moveTo(x(chosen),p.t);c.lineTo(x(chosen),h-p.b);c.stroke();c.setLineDash([]);for(const g of visible){c.fillStyle=colors[g];c.beginPath();c.arc(x(chosen),y(D.training[g][el('ema-toggle').checked?'loss_ema':'loss'][chosen-1]),3.5,0,2*Math.PI);c.fill();}
el('step-output').value=chosen;el('step-slider').value=chosen;el('chart-readout').innerHTML=`<span>Step <b>${chosen}</b> · 原始 loss</span>`+[...visible].map(g=>`<span style="color:${colors[g]}">${g} <strong>${D.training[g].loss[chosen-1].toFixed(6)}</strong></span>`).join('');}
canvas.addEventListener('pointermove',e=>{const r=canvas.getBoundingClientRect();chosen=Math.max(1,Math.min(3000,Math.round(1+(e.clientX-r.left-52)/(r.width-67)*2999)));draw();});
el('step-slider').oninput=e=>{chosen=+e.target.value;draw();};el('raw-toggle').onchange=draw;el('ema-toggle').onchange=draw;new ResizeObserver(draw).observe(canvas.parentElement);
for(const step of D.training.C.checkpoints){const b=document.createElement('button');b.textContent=step;b.onclick=()=>{chosen=step;draw();};el('checkpoint-buttons').append(b);}
function figures(prefix,items){const tabs=el(prefix+'-tabs');for(const [i,[file,title,caption]] of items.entries()){const b=document.createElement('button');b.textContent=title;b.onclick=()=>{tabs.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');el(prefix+'-figure').src=`assets/charts/${file}.png`;el(prefix+'-figure').alt=title;el(prefix+'-caption').textContent=caption;el(prefix+'-download').href=`assets/charts/${file}.svg`;};tabs.append(b);if(i===0)b.click();}}
figures('training',[
 ['loss-raw','原始 loss + 平滑','四组共享坐标范围；淡线为全部原始记录，实线为 EMA α=0.05。'],
 ['loss-comparison','四组趋势对照','完整 1–3000 步，未裁切早期高 loss。'],
 ['optimization','学习率与梯度','记录中的学习率和梯度范数。梯度图使用 EMA 平滑。'],
 ['loss-and-time','末段 loss 与耗时','耗时排除日志标记的编译步 1、3，仅为当次观测，不是受控硬件性能测试。']]);
figures('result',[
 ['success-rates','总体成功率','每组 500 回合，全部 errors=0、退出码=0。'],
 ['task-heatmap','逐任务热力图','每格 50 回合；任务 5 是 C 与 D 差异较大的任务。'],
 ['parameter-tradeoff','参数与效果','C 466.96M，D 872.77M；参数数量不等同于显存或耗时。'],
 ['paired-outcomes','配对成败','对照同一初态的结果，而不只比较两个平均分。'],
 ['recipe-comparison','450 / 500 配方','两轮训练示教和归一化均不同，不能将变化单独归因于数据数量。']]);
let activeCase=D.cases[0],selected=['B','C','D'];
function stop(){document.querySelectorAll('#players video').forEach(v=>v.pause());}
function renderPlayers(){stop();const container=el('players');container.innerHTML='';const available=Object.keys(activeCase.records);selected=selected.map(g=>available.includes(g)?g:'C');for(let slot=0;slot<3;slot++){const g=selected[slot],r=activeCase.records[g],card=document.createElement('article');card.className='player';card.innerHTML=`<div class="player-head"><select aria-label="第 ${slot+1} 列模型">${available.map(k=>`<option value="${k}" ${k===g?'selected':''}>${k} · ${names[k]}</option>`).join('')}</select></div><video controls playsinline muted preload="metadata" poster="media/${activeCase.id}/${g}-start.png" src="media/${activeCase.id}/${g}.mp4"></video><div class="player-foot"><span class="${r.success?'success':'failure'}">${r.success?'成功':'超时未成功'}</span><span>${r.steps} 控制步 · ${r.policy_calls} 次预测</span></div>`;card.querySelector('select').onchange=e=>{selected[slot]=e.target.value;renderPlayers();};card.querySelector('video').playbackRate=+el('speed').value;container.append(card);}el('play-status').textContent='已加载同一初态的三个策略记录，可切换模型并同时播放。';}
function setCase(c,b){activeCase=c;el('case-tabs').querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');el('case-title').textContent=c.title;el('case-prompt').textContent=c.prompt;el('case-key').textContent=`TASK ${String(c.task_id).padStart(2,'0')} / INIT ${String(c.initial_state_id).padStart(2,'0')}`;el('case-note').textContent='初态哈希一致 · 同一控制步对应同一播放时间 · 成功后停止的片段保留终帧 · 本页不做未经人工确认的失败机制归因。';renderPlayers();}
for(const [i,c] of D.cases.entries()){const b=document.createElement('button');b.textContent=c.title;b.onclick=()=>setCase(c,b);el('case-tabs').append(b);if(i===0)b.click();}
el('play-all').onclick=async()=>{const vs=[...document.querySelectorAll('#players video')],t=Math.min(...vs.map(v=>v.currentTime));vs.forEach(v=>v.currentTime=Math.min(t,Number.isFinite(v.duration)?v.duration:t));const results=await Promise.allSettled(vs.map(v=>v.play()));el('play-status').textContent=results.some(r=>r.status==='rejected')?'部分视频未能播放，请检查媒体文件或使用播放器的播放按钮。':'正在播放配对录像。';};el('pause-all').onclick=()=>{stop();el('play-status').textContent='全部暂停。';};el('restart-all').onclick=()=>{stop();document.querySelectorAll('#players video').forEach(v=>v.currentTime=0);el('play-status').textContent='已回到同一初始画面。';};el('speed').onchange=e=>document.querySelectorAll('#players video').forEach(v=>v.playbackRate=+e.target.value);
