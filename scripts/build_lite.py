"""Build the Lite and offline pages from local sources and saved predictions.

Usage: python scripts/build_lite.py
No network or machine-learning libraries are required for this build.
"""
from pathlib import Path
import json
import re

root = Path(__file__).resolve().parents[1]
read = lambda p: (root / p).read_text(encoding='utf8')
template = read('web/index.html')
data = json.loads(read('scripts/lite/data.json'))
if data['size'] != 96 or data['scale'] != 1000 or len(data['samples']) < 3:
    raise ValueError('Unexpected fixed-sample data format')

template = re.sub(r'<link rel="stylesheet".*?</head>', '</head>', template, count=1)
template = template.replace('<head>', '<head><link rel="icon" href="data:,">', 1)
template = re.sub(r'<a class="github".*?</a>', '', template, count=1)
template = re.sub(r'<a class="version-link".*?</a>', '', template)
template = re.sub(r'\s*<button id="load-models".*?</button>', '', template, count=1)
template = re.sub(r'<details class="parameters">.*?</details>',
                  '<label class="lite-sample">固定样本 <select id="sample" aria-label="固定样本">'
                  + ''.join(f'<option value="{i}">{s["label"]}</option>' for i,s in enumerate(data['samples']))
                  + '</select></label><p class="hint">CNN 结果为已训练模型的预计算输出；不能任意调整初始场。</p>', template, count=1)
if template.count('id="primary-layer"') != 1 or template.count('id="sample"') != 1:
    raise ValueError('Full page layout no longer matches the Lite build')
template = template.replace('96 × 96 · 同色系云图，轮廓区分图层 · 非物理相加', '96 × 96 · 固定色标 −14～0 K · 轮廓区分图层')

css = '\n'.join(read('web/'+name) for name in ('style.css','principle.css','training.css'))
css += '\n.lite-sample,.lite-primary{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:8px 0;color:#64706f;font-size:12px}.lite-sample select,.lite-primary select{max-width:70%;border:1px solid #e3e7e3;border-radius:5px;background:#fff;color:#222a2b;padding:4px}.lite-version{font-size:11px;color:#64706f;margin-left:6px}'
template = template.replace('</head>', '<style>'+css+'</style></head>', 1)

training = read('web/training.js')
old = "Promise.all(['data/replay.json','data/a.json','data/b.json'].map(url=>fetch(url).then(r=>{if(!r.ok)throw new Error(`${url}: HTTP ${r.status}`);return r.json();})))"
if old not in training:
    raise ValueError('Training data loader differs from the checked-in Lite source')
training = training.replace(old, 'Promise.resolve(window.LITE_DATA.training)')
if 'window.temperatureColor(v)' not in training:
    raise ValueError('Training source does not use the common physical color scale')

decode = r'''(function(){
const raw=__DATA__;
function field(s){const b=atob(s),d=new DataView(new ArrayBuffer(b.length));for(let i=0;i<b.length;i++)d.setUint8(i,b.charCodeAt(i));return Float32Array.from({length:b.length/2},(_,i)=>d.getInt16(i*2,true)/raw.scale)}
function grid(a){return Array.from({length:raw.size},(_,i)=>Array.from(a.slice(i*raw.size,(i+1)*raw.size)))}
const samples=raw.samples.map(s=>({label:s.label,initial:field(s.initial),a:field(s.a),b_left:field(s.b_left),b_right:field(s.b_right)}));
const t=raw.training,replay={size:raw.size,initial:grid(field(t.initial)),left:grid(field(t.left)),stages:t.stages.map(s=>({epoch:s.epoch,a:grid(field(s.a)),b_left:grid(field(s.b_left))}))};
window.LITE_DATA={samples,training:[replay,...t.logs]};
})();'''.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')))
scripts = [s.rstrip() for s in [decode, read('web/temperature-color.js'), read('web/principle.js'), training, read('scripts/lite/app.js')]]
for script in scripts:
    if '</script' in script.lower():
        raise ValueError('Unsafe closing script tag in inline source')
html = template.replace('</body>', ''.join('<script>'+s+'</script>' for s in scripts)+'</body>', 1)
for name, label in [('lite.html','轻量版 · 预计算结果'),('standalone.html','离线版 · 预计算结果')]:
    output = html.replace('完整在线版 · 实时模型按需加载',label,1)
    if label not in output:
        raise ValueError('Full version label is missing')
    if name == 'lite.html':
        output = output.replace('<div class="top-actions">',
            '<div class="top-actions"><a class="version-link" href="full.html">完整版</a>'
            '<a class="version-link" href="standalone.html" download>下载离线版</a>', 1)
    (root/'web'/name).write_text(output,encoding='utf8',newline='\n')
    print(f'{name}: {len(output.encode("utf8")):,} bytes')
