"""Train two CNNs on paired, disjoint initial fields. Run from repository root."""
import argparse,json,time,os,sys
from pathlib import Path
import numpy as np
import torch
from data import fields,branches,N
from models import ForecastNet

ROOT=Path(__file__).resolve().parent.parent

def save_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,ensure_ascii=False),encoding='utf8')

def inp(x,d):
    return torch.stack((x,torch.ones_like(x)*d),1)

def aslist(x): return np.round(x,4).tolist()

def scores(pred,true):
    e=pred-true; p=pred.reshape(len(pred),-1); t=true.reshape(len(true),-1)
    pm=p.min(1); tm=t.min(1); pp=np.argmin(p,axis=1); tt=np.argmin(t,axis=1)
    px=pp%N; py=pp//N; tx=tt%N; ty=tt//N
    dx=np.minimum(abs(px-tx),N-abs(px-tx)); dy=np.minimum(abs(py-ty),N-abs(py-ty))
    hit=(pred<=-8); obs=(true<=-8); tp=int((hit&obs).sum()); fp=int((hit&~obs).sum()); fn=int((~hit&obs).sum())
    return dict(mse=float(np.mean(e**2)),rmse=float(np.sqrt(np.mean(e**2))),peak_retention=float(np.mean(pm/tm)),peak_bias=float(np.mean(pm-tm)),location_error=float(np.mean(np.sqrt(dx**2+dy**2))),precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,false_positive=fp,false_negative=fn)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=int,default=2048); p.add_argument('--val',type=int,default=256); p.add_argument('--test',type=int,default=256); p.add_argument('--epochs',type=int,default=24); p.add_argument('--batch',type=int,default=64); p.add_argument('--width',type=int,default=32); p.add_argument('--benchmark',action='store_true'); args=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable. Install CUDA-enabled PyTorch for GPU training.')
    torch.manual_seed(2026); np.random.seed(2026); torch.backends.cudnn.benchmark=True; device='cuda'
    data={k:fields(n,s) for k,n,s in [('train',args.train,11),('validation',args.val,22),('test',args.test,33)]}
    tensors={k:torch.from_numpy(v).to(device) for k,v in data.items()}
    models={k:ForecastNet(args.width).to(device) for k in ('a','b')}
    opt={k:torch.optim.AdamW(m.parameters(),lr=.001) for k,m in models.items()}; scaler=torch.amp.GradScaler('cuda')
    logs={'a':[],'b':[]}; step=0; start=time.time(); replay=[]
    sample=tensors['validation'][:1]; left,right=branches(data['validation'][:1]); mean=(left+right)/2
    snapshots=set([0,1,max(2,args.epochs//3),max(3,2*args.epochs//3),args.epochs])
    def snapshot(epoch):
        with torch.no_grad():
            a=models['a'](inp(sample,0)).float().cpu().numpy()[0,0]
            bl=models['b'](inp(sample,-1)).float().cpu().numpy()[0,0]
            br=models['b'](inp(sample,1)).float().cpu().numpy()[0,0]
        replay.append(dict(epoch=epoch,a=aslist(a),b_left=aslist(bl),b_right=aslist(br)))
        for key,m in models.items(): torch.save({'epoch':epoch,'width':args.width,'state_dict':m.state_dict()},ROOT/f'results/checkpoints/{key}_epoch{epoch}.pt')
    if not args.benchmark: snapshot(0)
    for epoch in range(1,args.epochs+1):
        for m in models.values(): m.train()
        order=torch.randperm(args.train,device=device); total={'a':0.,'b':0.}; count=0
        for idx in order.split(args.batch):
            x=tensors['train'][idx]; y=torch.cat((torch.roll(x,-12,-1),torch.roll(x,12,-1))); xx=torch.cat((x,x)); dirs=torch.cat((-torch.ones_like(x),torch.ones_like(x)))
            for key,m in models.items():
                opt[key].zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    pred=m(torch.stack((xx,dirs if key=='b' else torch.zeros_like(xx)),1)); loss=torch.nn.functional.mse_loss(pred.float(),y[:,None])
                scaler.scale(loss).backward(); scaler.unscale_(opt[key]); torch.nn.utils.clip_grad_norm_(m.parameters(),10); scaler.step(opt[key]); total[key]+=loss.item()*len(xx)
            scaler.update(); count+=len(xx); step+=1
            if args.benchmark and step>=10: break
        for m in models.values(): m.eval()
        x=tensors['validation']; y=torch.cat((torch.roll(x,-12,-1),torch.roll(x,12,-1))); xx=torch.cat((x,x)); dirs=torch.cat((-torch.ones_like(x),torch.ones_like(x)))
        with torch.no_grad():
            for key,m in models.items():
                vals=[]
                for j in range(0,len(xx),args.batch):
                    v=m(torch.stack((xx[j:j+args.batch],dirs[j:j+args.batch] if key=='b' else torch.zeros_like(xx[j:j+args.batch])),1)).float(); vals.append(((v-y[j:j+args.batch,None])**2).mean().item())
                logs[key].append(dict(epoch=epoch,step=step,train_loss=total[key]/count,val_loss=float(np.mean(vals)),lr=opt[key].param_groups[0]['lr'],seconds=time.time()-start))
        print(f'epoch {epoch} {time.time()-start:.1f}s A={logs["a"][-1]["val_loss"]:.4f} B={logs["b"][-1]["val_loss"]:.4f}',flush=True)
        if not args.benchmark and epoch in snapshots: snapshot(epoch)
        if args.benchmark: return
    metrics={}
    for split,x0 in data.items():
        x=tensors[split]; l,r=branches(x0); truth=np.concatenate((l,r)); oracle=np.concatenate(((l+r)/2,(l+r)/2)); blind=np.concatenate((l,l)); predictions={}
        with torch.no_grad():
            for key,m in models.items():
                chunks=[]
                for j in range(0,len(x),args.batch):
                    xx=x[j:j+args.batch]; aa=m(inp(xx,0 if key=='a' else -1)).cpu().numpy()[:,0]; bb=aa if key=='a' else m(inp(xx,1)).cpu().numpy()[:,0]; chunks.append((aa,bb))
                predictions[key]=np.concatenate((np.concatenate([v[0] for v in chunks]),np.concatenate([v[1] for v in chunks])))
        metrics[split]={k:scores(v,truth) for k,v in dict(a=predictions['a'],b=predictions['b'],oracle=oracle,blind=blind).items()}
        metrics[split]['a']['oracle_rmse']=float(np.sqrt(np.mean((predictions['a']-oracle)**2)))
    metrics['metadata']={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'seconds':time.time()-start,'train_initial_fields':args.train,'validation_initial_fields':args.val,'test_initial_fields':args.test,'epochs':args.epochs,'batch':args.batch,'width':args.width,'parameters':sum(p.numel() for p in models['a'].parameters()),'seed':2026,'shift':12}
    replay_data=dict(size=N,initial=aslist(data['validation'][0]),left=aslist(left[0]),right=aslist(right[0]),mean=aslist(mean[0]),stages=replay)
    for dest in ('results/metrics','web/data'):
        save_json(ROOT/dest/'metrics.json',metrics); save_json(ROOT/dest/'replay.json',replay_data)
        for key in logs: save_json(ROOT/dest/f'{key}.json',logs[key])
    for key in logs: save_json(ROOT/'results/logs'/f'{key}.json',logs[key])
    for key,m in models.items():
        torch.save({'epoch':args.epochs,'width':args.width,'state_dict':m.state_dict()},ROOT/f'results/checkpoints/{key}_final.pt')
        try: torch.onnx.export(m,inp(sample,0),str(ROOT/f'web/models/{key}.onnx'),input_names=['input'],output_names=['output'],opset_version=17,dynamo=False)
        except Exception as ex: print('ONNX export failed:',ex,flush=True)
    print(json.dumps(metrics['metadata'],indent=2),flush=True)
if __name__=='__main__': main()
