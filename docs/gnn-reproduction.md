# GNN 训练与复核

在仓库根目录完成首页的环境、官方源码和公开验证集准备后，以下命令沿用 PowerShell 变量 `$python`。CPU 可以加载推理权重，记录中的训练和 2 秒对比使用 CUDA。

## 从头训练全量模型

官方训练集包含 9,000 个来源文件、1,008,000 条布局。先用固定种子重建开发集与 Holdout 各 10 个来源文件，再将其余 8,980 个文件编码为训练缓存。保留的 100 条重复训练副本与各自原记录的七个张量均相同，包括参考布局。源数据、图缓存及环境建议预留约 200 GB 磁盘空间。

```powershell
& $python scripts/prepare_official_data.py training
& $python -I -B -X utf8 scripts/prepare_gnn_pilot.py
& $python -I -B -X utf8 scripts/prepare_gnn_full_dataset.py
& $python -I -B -X utf8 scripts/verify_gnn_full_dataset.py --reencode-sources 30
& $python -I -B -X utf8 scripts/train_gnn_pilot_v3.py --cache outputs/full_dataset/data --output outputs/retrain_seed17 --seed 17 --epochs 50 --batch-size 64 --patience 51 --lr 0.0003
```

`best.pt` 保存开发集选出的权重，`last.pt` 保存断点续训状态。中断后用同一命令追加 `--resume`。可向 `scripts/compare_gnn_pilot.py` 传入 `--checkpoint outputs/retrain_seed17/best.pt` 验证自己的模型。

## 重评分新旧 GNN 的 200 份布局

配对实验保存了完整布局、原始指标和每例耗时。复制输入到新目录，复核脚本将重新读取 100 个公开案例，调用固定版官方评估器，并生成新的 `summary.json` 和 `cases.csv`。

```powershell
$source = 'results/gnn/ep47-vs-ep48'
$review = Join-Path 'outputs' ('rescore-' + [guid]::NewGuid())
New-Item -ItemType Directory -Force -Path $review | Out-Null
Copy-Item "$source/config.json","$source/layouts.jsonl" -Destination $review
& $python -I -B -X utf8 "$source/rescore.py" $review
& $python -I -B -X utf8 -m unittest discover -s tests -v
```

配对记录中的 checkpoint 路径对应原始实验目录。发布版的默认路径由 [gnn_default.json](../config/gnn_default.json) 指定，权重以 SHA-256 标识。
