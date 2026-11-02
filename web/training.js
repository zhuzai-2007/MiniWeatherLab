(function () {
  'use strict';
  window.initTraining = function initTraining(root) {
    if (!root || root.dataset.trainingReady) return;
    root.dataset.trainingReady = 'true';
    root.innerHTML = `<div class="training-screen">
      <div class="training-topline"><div><span class="training-index">03 / 训练过程</span><h2>看见模型学习</h2></div><span>真实训练记录与固定样本</span></div>
      <section class="training-chart-panel" aria-label="训练损失曲线"><div class="training-chart-heading"><strong>训练损失曲线</strong><div class="training-key"><span class="a">━ A 验证</span><span class="a faint">┄ A 训练</span><span class="b">━ B 验证</span><span class="b faint">┄ B 训练</span><span class="training-scale-label">MSE · 对数刻度</span></div></div><svg class="training-loss" viewBox="0 0 1200 250" preserveAspectRatio="none" role="img" aria-label="A 和 B 训练与验证损失随 Epoch 变化"></svg></section>
      <div class="training-controls"><button class="training-play">▶ 播放训练过程</button><label class="training-speed-label">速度 <select class="training-speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><label class="training-timeline"><span>训练进度</span><input class="training-slider" type="range" min="0" max="18" step="0.01" value="0" aria-label="训练 Epoch"></label><strong class="training-epoch">Epoch 0</strong></div>
      <div class="training-snap-title"><strong>固定样本的预测演化</strong><span class="training-checkpoint">加载真实 checkpoint…</span></div>
      <div class="training-fields"><figure><figcaption><b>真实未来</b><span>左移 · 固定参考</span></figcaption><canvas class="training-truth" width="288" height="288" aria-label="固定样本实际未来"></canvas><small class="training-truth-min"></small></figure><figure><figcaption><b>模型 A</b><span>无方向 · 趋近双峰平均场</span></figcaption><canvas class="training-a" width="288" height="288" aria-label="模型 A 的真实 checkpoint 预测"></canvas><small class="training-a-min"></small></figure><figure><figcaption><b>模型 B</b><span>有方向 · 恢复单侧冷中心</span></figcaption><canvas class="training-b" width="288" height="288" aria-label="模型 B 的真实 checkpoint 预测"></canvas><small class="training-b-min"></small></figure></div>
      <div class="training-foot"><span>相同 64×64 局部窗口 · 色标 −14 ～ 0 K（非线性）· 正值显示为白色</span><span>固定测试集：A 0.624 K ≈ Oracle 0.622 K；A 与 Oracle 场 RMSE 0.049 K</span></div><div class="training-error" hidden></div>
    </div>`;
    const q = s => root.querySelector(s);
    let replay, aLog, bLog, epoch = 0, playing = false, last = 0, animation = 0, lastSnapshot = -1, viewX = 0, viewY = 0;
    const svg = q('.training-loss'), slider = q('.training-slider'), button = q('.training-play');
    const colors = { a: '#d47b61', b: '#2c8978' };
    function chart() {
      if (!aLog || !bLog) return;
      const L=74, R=1176, T=12, B=214, max=18, yMin=-4, yMax=0;
      const x = e => L + e / max * (R - L);
      const y = v => B - (Math.log10(Math.max(0.0001,v)) - yMin) / (yMax-yMin) * (B-T);
      const paths = [[aLog,'train_loss',colors.a,true],[aLog,'val_loss',colors.a,false],[bLog,'train_loss',colors.b,true],[bLog,'val_loss',colors.b,false]];
      let out = `<defs><clipPath id="training-progress-clip"><rect x="${L}" y="0" width="${Math.max(0,x(epoch)-L)}" height="250"/></clipPath></defs>`;
      [-4,-3,-2,-1,0].forEach(k => {const yy=y(10**k);out += `<line x1="${L}" y1="${yy}" x2="${R}" y2="${yy}" stroke="#e6ebe7"/><text x="${L-10}" y="${yy+4}" text-anchor="end" fill="#718079" font-size="14">${10**k}</text>`;});
      [0,3,6,9,12,15,18].forEach(e=>{out+=`<text x="${x(e)}" y="239" text-anchor="middle" fill="#718079" font-size="14">${e}</text>`;});
      paths.forEach(([data,key,color,dash])=>{const points=data.map(d=>`${x(d.epoch)},${y(d[key])}`).join(' ');const dashes=dash?'stroke-dasharray="7 6"':'';out+=`<polyline points="${points}" fill="none" stroke="${color}" opacity=".14" stroke-width="2" ${dashes}/><polyline points="${points}" fill="none" stroke="${color}" opacity="${dash?.72:1}" stroke-width="${dash?2.4:3.5}" ${dashes} clip-path="url(#training-progress-clip)"/>`;});
      out+=`<line x1="${x(epoch)}" y1="${T}" x2="${x(epoch)}" y2="${B}" stroke="#8e9c94" stroke-width="1"/><text x="${R}" y="249" text-anchor="end" fill="#718079" font-size="13">Epoch</text>`;svg.innerHTML=out;
    }
    function heat(canvas, field) {
      const n=field.length, size=64, off=document.createElement('canvas'); off.width=off.height=size;
      const c=off.getContext('2d'), im=c.createImageData(size,size); let min=Infinity;
      for(let j=0;j<n;j++)for(let i=0;i<n;i++)min=Math.min(min,field[j][i]);
      for(let j=0;j<size;j++)for(let i=0;i<size;i++){
        const v=field[viewY+j][viewX+i];
        // One shared field-of-view and fixed color scale across all snapshots.
        const rgb=window.temperatureColor(v);
        const p=(j*size+i)*4; for(let k=0;k<3;k++)im.data[p+k]=rgb[k]; im.data[p+3]=255;
      }
      c.putImageData(im,0,0);const ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=true;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(off,0,0,canvas.width,canvas.height);
      return min;
    }
    function render() {
      slider.value=epoch.toFixed(2);q('.training-epoch').textContent=`Epoch ${Number(epoch.toFixed(1))}`;chart();
      const stage=replay.stages.reduce((best,s)=>Math.abs(s.epoch-epoch)<Math.abs(best.epoch-epoch)?s:best,replay.stages[0]);
      q('.training-checkpoint').textContent=`训练记录已加载 · 真实快照：Epoch ${stage.epoch}（最近保存）`;
      if(stage.epoch!==lastSnapshot){lastSnapshot=stage.epoch;const mA=heat(q('.training-a'),stage.a),mB=heat(q('.training-b'),stage.b_left);q('.training-a-min').textContent=`全场最低 ${mA.toFixed(2)} K`;q('.training-b-min').textContent=`全场最低 ${mB.toFixed(2)} K`;}
    }
    function stop(){playing=false;cancelAnimationFrame(animation);button.textContent=epoch>=18?'↻ 重播训练过程':'▶ 继续播放';}
    function tick(now){if(!playing)return;const delta=Math.min(0.1,(now-last)/1000);last=now;epoch=Math.min(18,epoch+delta*18/15*Number(q('.training-speed').value));render();if(epoch>=18){stop();return;}animation=requestAnimationFrame(tick);}
    button.onclick=()=>{if(!replay)return;if(playing){stop();return;}if(epoch>=18)epoch=0;playing=true;last=performance.now();button.textContent='Ⅱ 暂停';animation=requestAnimationFrame(tick);};
    slider.oninput=()=>{if(!replay)return;if(playing)stop();epoch=Number(slider.value);render();};
    Promise.all(['data/replay.json','data/a.json','data/b.json'].map(url=>fetch(url).then(r=>{if(!r.ok)throw new Error(`${url}: HTTP ${r.status}`);return r.json();}))).then(data=>{
      [replay,aLog,bLog]=data;let peak=Infinity,px=0,py=0;replay.initial.forEach((row,y)=>row.forEach((v,x)=>{if(v<peak){peak=v;px=x;py=y;}}));viewX=Math.max(0,Math.min(replay.size-64,px-32));viewY=Math.max(0,Math.min(replay.size-64,py-32));const m=heat(q('.training-truth'),replay.left);q('.training-truth-min').textContent=`全场最低 ${m.toFixed(2)} K`;q('.training-checkpoint').textContent='训练记录已加载';render();
    }).catch(err=>{console.error(err);q('.training-error').hidden=false;q('.training-error').textContent='训练记录加载失败，请通过本地 HTTP 服务器打开并检查 web/data。';q('.training-checkpoint').textContent='训练数据不可用';button.disabled=true;slider.disabled=true;});
  };
})();

