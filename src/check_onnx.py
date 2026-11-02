"""Check exported ONNX outputs against the corresponding trained PyTorch weights."""
from pathlib import Path
import json
import numpy as np
import torch
import onnxruntime as ort
from models import ForecastNet
from data import fields,N
root=Path(__file__).resolve().parent.parent
x=fields(3,99)
report={}
for name in ('a','b'):
    checkpoint=torch.load(root/f'results/checkpoints/{name}_final.pt',map_location='cpu',weights_only=True)
    model=ForecastNet(checkpoint['width']); model.load_state_dict(checkpoint['state_dict']); model.eval()
    ort_session=ort.InferenceSession(str(root/f'web/models/{name}.onnx'),providers=['CPUExecutionProvider'])
    errors=[]
    for d in (-1,1):
        for field in x:
            value=np.stack((field,np.full((N,N),d if name=='b' else 0,dtype=np.float32)))[None]
            with torch.no_grad(): expected=model(torch.from_numpy(value)).numpy()
            actual=ort_session.run(None,{'input':value})[0]
            errors.append(float(np.max(np.abs(expected-actual))))
    report[name]={'max_abs_error':max(errors),'unit':'K','passed':max(errors)<1e-3}
(root/'results/metrics/onnx_parity.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
assert all(v['passed'] for v in report.values())
