(function () {
  'use strict';
  window.initPrinciple = function initPrinciple(root) {
    if (!root || root.dataset.principleReady) return;
    root.dataset.principleReady = 'true';
    root.innerHTML = `
      <div class="principle-shell">
        <section class="principle-panel principle-flow" aria-label="实验流程">
          <div class="principle-head"><div><span class="principle-kicker">从数据到结果</span><h2>一个初始场，如何变成预测？</h2></div><span class="principle-caption">理想化二维平流实验</span></div>
          <div class="flow-list">
            <article class="flow-step"><span class="flow-index">01</span><div class="flow-copy"><h3>生成初始温度场</h3><p>在 96 × 96 网格上生成一个低温异常。</p></div><div class="flow-art mini-field" aria-hidden="true"><span></span></div></article>
            <div class="flow-link" aria-hidden="true">↓</div>
            <article class="flow-step"><span class="flow-index">02</span><div class="flow-copy"><h3>形成两种可能未来</h3><p>异常向左或向右平移 12 格，各有 50% 概率。</p></div><div class="flow-art two-futures" aria-hidden="true"><span>50%</span><span>50%</span></div></article>
            <div class="flow-link" aria-hidden="true">↓</div>
            <article class="flow-step"><span class="flow-index">03</span><div class="flow-copy"><h3>给模型不同的信息</h3><p>A 只看二维温度场；B 额外获得方向条件。</p></div><div class="flow-art input-pills" aria-hidden="true"><span><b>A</b> 初始场</span><span><b>B</b> 初始场 + 方向</span></div></article>
            <div class="flow-link" aria-hidden="true">↓</div>
            <article class="flow-step"><span class="flow-index">04</span><div class="flow-copy"><h3>CNN 提取空间结构</h3><p>不同卷积核响应不同局部模式；多层组合成空间特征。</p></div><div class="flow-art flow-kernel" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div></article>
            <div class="flow-link" aria-hidden="true">↓</div>
            <article class="flow-step"><span class="flow-index">05</span><div class="flow-copy"><h3>输出未来场，用 MSE 学习</h3><p>特征经训练映射为未来场；B 能随方向改变预测。</p></div><div class="flow-art loss-mark" aria-hidden="true"><span>预测</span><b>↔</b><span>真实</span><strong>MSE</strong></div></article>
          </div>
          <p class="principle-foot">MSE 比较预测与真实未来；A / B 同结构，各 766,433 参数，B 的方向条件为 −1 或 +1。</p>
        </section>
        <section class="principle-panel principle-conv" aria-label="卷积计算示意">
          <div class="principle-head"><div><span class="principle-kicker">放大第 04 步</span><h2>卷积如何提取局部结构？</h2></div></div>
          <p class="principle-intro">不同卷积核对低温中心、左右变化等局部模式产生不同响应；多层组合后可表达更大范围的结构。</p>
          <div class="principle-grids"><div class="grid-block"><span>输入温度场</span><div class="p-input-grid" aria-label="8 乘 8 输入示意网格"></div></div><div class="grid-arrow" aria-hidden="true">→</div><div class="grid-block"><span>输出特征图</span><div class="p-output-grid" aria-label="6 乘 6 输出特征示意网格"></div></div></div>
          <div class="p-kernel-choice" aria-label="选择示意卷积核"><span>比较同一局部区域</span><button type="button" data-kernel="center" class="selected" aria-pressed="true">低温响应</button><button type="button" data-kernel="gradient" aria-pressed="false">左右变化</button><span class="p-comparison"></span></div><div class="principle-formula">Y<sub>o,x,y</sub> = ∑<sub>c,i,j</sub> W<sub>o,c,i,j</sub> X<sub>c,x+i,y+j</sub> + b<sub>o</sub></div>
          <div class="calculation"><div class="calc-block"><span>局部输入</span><div class="p-patch-grid"></div></div><b class="calc-operator">×</b><div class="calc-block"><span>卷积核</span><div class="p-kernel-grid"></div></div><b class="calc-operator">=</b><div class="calc-block"><span>逐项乘积</span><div class="p-product-grid"></div></div><div class="calc-result"><span>求和 + 偏置 0</span><strong class="p-sum">0</strong></div></div>
          <div class="principle-controls"><button class="p-play">▶ 播放</button><button class="p-step">单步</button><button class="p-reset">重置</button><label>速度 <select class="p-speed"><option value="1200">慢</option><option value="700" selected>正常</option><option value="350">快</option></select></label><span class="p-position"></span></div>
          <p class="principle-note">示意核并非训练权重。CNN 逐层提取空间模式，通过训练学习它们与未来温度场的关系。</p>
        </section>
      </div>`;
    const q = selector => root.querySelector(selector);
    const input = Array.from({length:8}, (_, y) => Array.from({length:8}, (_, x) => -Math.round(4 * Math.exp(-((x-3.5)**2 + (y-3.5)**2)/4))));
    const kernels = {center:[[0,1,0],[1,2,1],[0,1,0]],gradient:[[-1,0,1],[-2,0,2],[-1,0,1]]};
    let selectedKernel = 'center', step = 14, timer = null;
    const value = (y,x,which=selectedKernel) => kernels[which].reduce((sum,row,i) => sum + row.reduce((part,k,j) => part + k*input[y+i][x+j],0),0);
    function render() {
      const row = Math.floor(step / 6), col = step % 6, kernel=kernels[selectedKernel];
      q('.p-input-grid').innerHTML = input.flatMap((line,y) => line.map((n,x) => `<span class="${y>=row&&y<row+3&&x>=col&&x<col+3?'in-window':''}" style="--cold:${Math.abs(n)/4}">${n}</span>`)).join('');
      q('.p-output-grid').innerHTML = Array.from({length:36},(_,i) => `<span class="${i===step?'current':''}">${i<=step?value(Math.floor(i/6),i%6):'·'}</span>`).join('');
      const patch = [], products = [];
      for (let y=0;y<3;y++) for(let x=0;x<3;x++) { patch.push(input[row+y][col+x]); products.push(input[row+y][col+x]*kernel[y][x]); }
      q('.p-patch-grid').innerHTML = patch.map(n=>`<span>${n}</span>`).join('');
      q('.p-kernel-grid').innerHTML = kernel.flat().map(n=>`<span>${n}</span>`).join('');
      q('.p-product-grid').innerHTML = products.map(n=>`<span>${n}</span>`).join('');
      q('.p-sum').textContent = value(row,col);
      q('.p-comparison').textContent = `低温响应 ${value(row,col,'center')} · 左右变化 ${value(row,col,'gradient')}`;
      root.querySelectorAll('[data-kernel]').forEach(button=>{const on=button.dataset.kernel===selectedKernel;button.classList.toggle('selected',on);button.setAttribute('aria-pressed',String(on));});
      q('.p-position').textContent = `位置 ${step+1} / 36`;
    }
    function stop() { clearInterval(timer); timer = null; q('.p-play').textContent = step===35 ? '↻ 重播' : '▶ 播放'; }
    root.querySelectorAll('[data-kernel]').forEach(button=>{button.onclick=()=>{stop();selectedKernel=button.dataset.kernel;render();};});
    q('.p-play').onclick = () => { if(timer){stop();return;} if(step===35)step=0; render(); q('.p-play').textContent='Ⅱ 暂停'; timer=setInterval(()=>{step++;render();if(step===35)stop();}, Number(q('.p-speed').value)); };
    q('.p-step').onclick = () => {stop();step=Math.min(35,step+1);render();};
    q('.p-reset').onclick = () => {stop();step=0;render();};
    q('.p-speed').onchange = () => { if(timer) { stop(); q('.p-play').click(); } };
    render();
  };
})();
