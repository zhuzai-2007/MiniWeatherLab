# 实验与复现

## 数据与模型

`src/data.py` 用固定随机种子生成 96 × 96 周期网格上的高斯低温异常，左右各平移 12 格得到等概率分支。训练、验证、测试分别使用 1024、128、128 个独立生成的初始场及其两个分支，种子分别为 11、22、33。数据为合成数据，不含真实用户数据。

`src/models.py` 定义两个模型共用的编码器—解码器 CNN，输入为初始场和方向通道；A 的方向通道为零，B 为 −1 或 +1。每个模型有 766,433 个参数。`src/train.py` 使用 PyTorch、AdamW、MSE 和 CUDA 混合精度训练；本次记录的随机种子为 2026，batch 为 32，训练 18 epochs。详情及独立测试集统计保存在 `results/metrics/metrics.json`。

本次环境为 NVIDIA GeForce RTX 5070 Ti Laptop GPU、PyTorch 2.11.0+cu128、CUDA 12.8，记录的训练及评估时间约 94.7 秒；时间仅供参考，会随硬件及环境变化。峰值保持率是每个样本的预测最低温异常与真实最低温异常之比的平均，测试集 RMSE 则按全部网格汇总。

## 安装与运行

使用 Python 3 创建并激活虚拟环境，安装与本机硬件匹配的 PyTorch，然后在项目根目录运行：

```sh
python -m pip install -r requirements.txt
python src/check_data.py
```

查看网页不需要这些 Python 依赖。训练脚本要求可用的 CUDA GPU；训练会覆盖现有的模型、指标和回放文件，建议先在单独的项目副本中运行。复现本次参数：

```sh
python src/train.py --train 1024 --val 128 --test 128 --epochs 18 --batch 32 --width 32
```

浮点运算、PyTorch/CUDA 版本和硬件差异可能导致结果不完全相同。`python src/check_onnx.py` 比较最终 PyTorch 权重与 ONNX 输出，并会重写 `results/metrics/onnx_parity.json`。`python src/visualize.py` 根据保存的日志与回放生成图表；中文字体的显示取决于本地环境。

`results/checkpoints/*_final.pt` 是最终 PyTorch 权重；训练中间权重默认保留在本地、不纳入公开清单。网页实际使用 `web/models/`、`web/data/` 和 `web/vendor/` 中的文件，回放数据包含各保存阶段的预测，不需要中间权重即可播放。图表由项目代码和实验数据生成；网页截图来自本项目界面。
