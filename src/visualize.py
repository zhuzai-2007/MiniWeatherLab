"""Make figures from recorded training logs and predictions; run after training."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#f5f8fc', 'axes.facecolor': '#f5f8fc',
                     'axes.unicode_minus': False, 'svg.fonttype': 'none'})

def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(OUT / (name + '.svg'), facecolor=fig.get_facecolor())
    plt.close(fig)

def field(ax, arr, title):
    im = ax.imshow(np.asarray(arr), origin='lower', cmap='Blues_r', vmin=-14, vmax=0, interpolation='nearest')
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    return im

def render():
    r = load('results/metrics/replay.json')
    metrics = load('results/metrics/metrics.json')
    logs = {k: load(f'results/logs/{k}.json') for k in ('a', 'b')}
    stages = r['stages']; final = stages[-1]
    initial, left, right, mean = (np.array(r[k]) for k in ('initial', 'left', 'right', 'mean'))
    a, b = np.array(final['a']), np.array(final['b_right'])

    fig, ax = plt.subplots(figsize=(12.8, 7.2), layout='constrained')
    for k, c in [('a', '#2868b2'), ('b', '#e1743b')]:
        rows = logs[k]
        ax.plot([x['epoch'] for x in rows], [x['train_loss'] for x in rows], color=c, label=f'{k.upper()} 训练')
        ax.plot([x['epoch'] for x in rows], [x['val_loss'] for x in rows], color=c, linestyle='--', label=f'{k.upper()} 验证')
    ax.axhline(metrics['validation']['oracle']['mse'], color='#627387', linestyle=':', label='验证集 Oracle Mean')
    ax.set(title='训练动态  /  Recorded training', xlabel='Epoch', ylabel='MSE (K²)')
    ax.grid(alpha=.2); ax.legend(ncol=2, frameon=False)
    save(fig, '01_training_dynamics')

    fig, axs = plt.subplots(2, min(5, len(stages)), figsize=(16, 7.2), layout='constrained', squeeze=False)
    indices = np.linspace(0, len(stages)-1, axs.shape[1]).round().astype(int)
    for j, idx in enumerate(indices):
        s = stages[idx]
        field(axs[0,j], s['a'], f"A · epoch {s['epoch']}")
        im = field(axs[1,j], s['b_right'], f"B → · epoch {s['epoch']}")
    fig.suptitle('学习预测  /  真实 checkpoint 回放', fontsize=20)
    fig.colorbar(im, ax=axs, label="温度异常 T′ (K)", shrink=.8)
    save(fig, '02_learning_to_forecast')

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for ax, arr, title in zip(axs.flat, [initial,right,a,b,mean,left],
          ['初始场','真实未来 →','模型 A · 无方向','模型 B · 已知方向 →','Oracle Mean · 理论平均','另一可能未来 ←']):
        im=field(ax, arr, title)
    fig.suptitle('预测比较  /  统一色标', fontsize=20)
    fig.colorbar(im, ax=axs, label="温度异常 T′ (K)", shrink=.8)
    save(fig, '03_forecast_comparison')

    y, x = np.unravel_index(np.argmin(right), right.shape)
    fig, ax = plt.subplots(figsize=(12.8, 7.2), layout='constrained')
    for arr, label, color, ls in [(right,'真实未来 →','#122e54','-'),(a,'A · 无方向','#3588c7','-'),(b,'B · 已知方向','#e1743b','-'),(mean,'Oracle Mean','#627387','--')]:
        ax.plot(arr[y,:], label=label, color=color, linestyle=ls, linewidth=2.5)
    ax.axhline(-8, color='#b45165', linestyle=':', label='−8 K 阈值')
    ax.set(title=f'温度横截面  /  真值极小值所在行 y = {y}', xlabel='网格 x', ylabel="温度异常 T′ (K)")
    ax.grid(alpha=.2); ax.legend(frameon=False, ncol=2)
    save(fig, '04_cross_section')

    test = metrics['test']; names=['blind','oracle','a','b']
    labels=['盲猜方向','Oracle Mean','模型 A','模型 B']
    fig, axs = plt.subplots(1, 2, figsize=(14, 7.2), layout='constrained')
    colors=['#8d9cad','#7ca8cb','#2868b2','#e1743b']
    for ax, key, title, unit in [(axs[0],'rmse','平均误差','RMSE (K)'),(axs[1],'peak_retention','极值幅度保持','比例')]:
        vals=[test[n][key] for n in names]
        ax.bar(labels, vals, color=colors); ax.set_title(title); ax.set_ylabel(unit); ax.grid(axis='y', alpha=.2)
        ax.bar_label(ax.containers[0], fmt='%.3f', padding=4)
    fig.suptitle('平均精度与极值保持  /  测试集\nB 利用方向信息；A、Oracle Mean 与盲猜不具备该信息', fontsize=16)
    save(fig, '05_accuracy_vs_extremes')

    fig = plt.figure(figsize=(16,9), layout='constrained')
    gs = fig.add_gridspec(2,4)
    panels=[]
    for i,(arr,title) in enumerate([(initial,'① 一个初始状态'),(left,'② 未来 ←  /  50%'),(right,'② 未来 →  /  50%'),(a,'③ A：单值预测')]):
        panel=fig.add_subplot(gs[0,i]); panels.append(panel); im=field(panel,arr,title)
    ax=fig.add_subplot(gs[1,:2])
    for arr,label,c in [(right,'真实未来 →','#122e54'),(a,'模型 A','#3588c7'),(b,'模型 B','#e1743b')]: ax.plot(arr[y,:],label=label,color=c,lw=2.5)
    ax.axhline(-8, color='#b45165', linestyle=':'); ax.grid(alpha=.2); ax.legend(frameon=False); ax.set_title('沿真实冷中心的剖面'); ax.set_ylabel("T′ (K)")
    panel=fig.add_subplot(gs[1,2]); panels.append(panel); field(panel,b,'④ B：方向已知 →')
    panel=fig.add_subplot(gs[1,3]); panels.append(panel); field(panel,mean,'理论条件平均')
    fig.colorbar(im, ax=panels, label="T′ (K)", shrink=.6)
    fig.suptitle('MiniWeatherLab  |  平均更准，为何极值难测？',fontsize=23,weight='bold')
    save(fig,'06_overview_16x9')
    print(f'Figures saved to {OUT}')

if __name__ == '__main__':
    render()
