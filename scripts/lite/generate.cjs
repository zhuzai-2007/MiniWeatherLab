// Rebuild the fixed examples from the shipped trained ONNX models, without training.
// Run from the repository root: node scripts/lite/generate.cjs
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const ort = require('../../web/vendor/onnxruntime/ort.min.js');
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = pathToFileURL(path.resolve('web/vendor/onnxruntime') + path.sep).href;
const N = 96;
let maxQuantError = 0;
function pack(field) {
  const data = Buffer.alloc(field.length * 2);
  field.forEach((v, i) => {
    const q = Math.round(v * 1000);
    if (q < -32768 || q > 32767) throw new Error('Field exceeds int16 range');
    maxQuantError = Math.max(maxQuantError, Math.abs(q / 1000 - v));
    data.writeInt16LE(q, i * 2);
  });
  return data.toString('base64');
}
function gaussian(amp, cx, cy, sigma) {
  return Float32Array.from({length:N*N}, (_, i) => {
    const x = i % N, y = Math.floor(i / N);
    const dx = Math.min(Math.abs(x-cx), N-Math.abs(x-cx));
    const dy = Math.min(Math.abs(y-cy), N-Math.abs(y-cy));
    return -amp * Math.exp(-(dx*dx + dy*dy)/(2*sigma*sigma));
  });
}
async function run(session, field, direction) {
  const input = new Float32Array(N*N*2); input.set(field); input.fill(direction, N*N);
  return (await session.run({input:new ort.Tensor('float32',input,[1,2,N,N])})).output.data;
}
async function main() {
  const replay = JSON.parse(fs.readFileSync('web/data/replay.json','utf8'));
  const logs = ['a','b'].map(k => JSON.parse(fs.readFileSync(`web/data/${k}.json`,'utf8')));
  const sessions = await Promise.all(['a','b'].map(k => ort.InferenceSession.create(fs.readFileSync(`web/models/${k}.onnx`),{executionProviders:['wasm']})));
  const examples = [
    ['验证集固定样本', Float32Array.from(replay.initial.flat())],
    ['较弱且窄的异常', gaussian(7.5,35,40,4)],
    ['较强且宽的异常', gaussian(13,62,55,5.5)]
  ];
  let defaultError = 0;
  const samples = [];
  for (const [label,original] of examples) {
    // Run the saved model on exactly the input values the lightweight page
    // will decode, so the displayed prediction corresponds to its input.
    const initial = Float32Array.from(original, v => Math.round(v * 1000) / 1000);
    const [a,bl,br] = await Promise.all([run(sessions[0],initial,0),run(sessions[1],initial,-1),run(sessions[1],initial,1)]);
    if (!samples.length) {
      const last = replay.stages[replay.stages.length-1];
      for (const [actual,recorded] of [[a,last.a],[bl,last.b_left],[br,last.b_right]]) {
        recorded.flat().forEach((v,i)=>{defaultError=Math.max(defaultError,Math.abs(actual[i]-v));});
      }
      // Replay inputs were saved to four decimal places; ONNX runs on that rounded input.
      if (defaultError > 0.003) throw new Error(`Default sample differs from checkpoint record: ${defaultError} K`);
    }
    samples.push({label,initial:pack(initial),a:pack(a),b_left:pack(bl),b_right:pack(br)});
  }
  const training = {left:pack(replay.left.flat()), initial:pack(replay.initial.flat()),
    stages:replay.stages.map(s=>({epoch:s.epoch,a:pack(s.a.flat()),b_left:pack(s.b_left.flat())})), logs};
  fs.writeFileSync('scripts/lite/data.json',JSON.stringify({scale:1000,size:N,samples,training}));
  console.log(`Saved ${samples.length} samples, max quantization error ${maxQuantError.toFixed(7)} K; default vs saved final snapshot ${defaultError.toFixed(7)} K`);
}
main().catch(e=>{console.error(e);process.exitCode=1});
