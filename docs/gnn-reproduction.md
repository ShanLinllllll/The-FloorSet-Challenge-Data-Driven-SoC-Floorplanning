# GNN 训练与复核

在仓库根目录完成首页的环境、官方源码和公开验证集准备后，以下命令沿用 PowerShell 变量 `$python`。CPU 可以加载推理权重，本项目的训练与公开验证使用 CUDA。

## 从头训练全量模型

官方训练集包含 9,000 个来源文件、1,008,000 条布局。先用固定种子重建开发集与 Holdout 各 10 个来源文件，再将其余 8,980 个文件编码为训练缓存。保留的 100 条重复训练副本与各自原记录的七个张量均相同，包括参考布局。源数据、图缓存及环境建议预留约 200 GB 磁盘空间。

```powershell
& $python scripts/prepare_official_data.py training
& $python -I -B -X utf8 scripts/prepare_gnn_pilot.py
& $python -I -B -X utf8 scripts/prepare_gnn_full_dataset.py
& $python -I -B -X utf8 scripts/verify_gnn_full_dataset.py --reencode-sources 30
& $python -I -B -X utf8 scripts/train_gnn_pilot_v3.py --cache outputs/full_dataset/data --output outputs/retrain_seed17 --seed 17 --epochs 50 --batch-size 64 --patience 51 --lr 0.0003
```

`best.pt` 保存开发集选出的权重，`last.pt` 保存断点续训状态。中断后用同一命令追加 `--resume`。

## 运行回归测试

```powershell
& $python -I -B -X utf8 -m unittest discover -s tests -v
```
