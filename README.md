# FloorSet Challenge · SoC 布局求解

给定模块面积、连接图、固定端口和布局约束，求解器为每个模块返回 `(x, y, width, height)`。GNN 根据连接图预测初始布局，再通过合法化、线性规划压缩和模拟退火得到最终布局。模型与使用启发式初始化的基线共用同一套后处理。

**默认 GNN 是 FloorSet-Lite 全量训练的第 47 轮模型。** 100 个公开验证案例直接对跑时，相比旧的第 48 轮模型，最终加权 cost 从 **3.321051 降至 3.263515**，降低 **1.73%**。两者均为 100/100 硬约束可行。[权重](models/floorset-lite-gnn-ep47/best.pt) · [逐例对比](results/gnn/ep47-vs-ep48/cases.csv)

## 运行预训练模型

以下命令在仓库根目录执行。验证环境为 Windows、Python 3.12、PyTorch 2.7.1 和 RTX 4060 Laptop GPU。复核公开案例只需官方代码和验证集。

```powershell
py -3.12 -m venv .venv
$python = (Resolve-Path .venv/Scripts/python.exe).Path
& $python -m pip install -r requirements-step1-lock.txt
& $python scripts/prepare_official_data.py source
& $python scripts/prepare_official_data.py validation

# 加载默认权重并显示实际轮次、设备和校验值
& $python -c "from gnn_optimizer import MyOptimizer; s = MyOptimizer(); print(s.checkpoint_epoch, s.device, s.checkpoint_sha256)"

# 在 21、60、120 个模块的案例上比较 GNN 与启发式基线
& $python -I -B -X utf8 scripts/compare_gnn_pilot.py --cases 0,39,99 --seconds 2 --seed 42
```

把 `--cases 0,39,99` 改为 `--cases all` 即可运行完整 100 例。比较结果写入 `outputs/comparisons/` 下的新目录。接入自己的数据时，调用 `gnn_optimizer.MyOptimizer().solve(...)`，输入字段与官方接口一致。[求解器入口](gnn_optimizer.py)

## 模型和训练

| 项目 | 第 47 轮模型 |
|---|---|
| 架构 | 4 层图神经网络，隐藏维度 128，431,303 个参数 |
| 训练数据 | 8,980 个来源文件，1,005,760 条样本，含 1,005,660 个唯一输入 |
| 训练配置 | 种子 17，batch size 64，初始学习率 0.0003，完成 50 轮 |
| 开发集 | 1,120 条样本，第 47 轮选择损失最低，为 0.074771 |
| Holdout | 1,120 条样本，选择损失 0.064747，模块对相对位置准确率 84.38% |
| 权重 SHA-256 | `15672107b9ee22683f5b76795a26d849f61e32010b34d703a702ea3b5a0cb344` |

默认加载器同时核对配置、权重文件和训练完成记录。原始训练参数、逐轮指标和 Holdout 指标随权重发布。[默认配置](config/gnn_default.json) · [训练配置](models/floorset-lite-gnn-ep47/config.json) · [学习曲线](models/floorset-lite-gnn-ep47/history.json) · [训练结果](models/floorset-lite-gnn-ep47/result.json)

## 公开验证结果

以下两次实验均使用求解种子 42、每例 2 秒目标预算和本地评分 `RuntimeFactor=1`。编码与推理计入预算。耗时截止会影响退火步数，复跑时最终分数可能略有变化。

| 配对实验 | 对照加权 cost | 全量 GNN 加权 cost | 降幅 | 全量 GNN 胜 / 负 |
|---|---:|---:|---:|---:|
| 第 48 轮子集 GNN 与第 47 轮全量 GNN | 3.321051 | 3.263515 | 1.73% | 54 / 46 |
| 启发式初始化与第 47 轮全量 GNN | 3.728719 | 3.258741 | 12.60% | 65 / 35 |

直接比较新旧模型时，两者各先运行 50 次。全量模型在合法化后的加权 cost 低 2.46%，LP 压缩后低 2.85%，退火后低 1.73%。200 份保存的最终布局通过官方评估器重评分。第 48 轮使用 11,200 个训练样本和 batch size 8，两次训练同时改变了数据量和批大小。[新旧模型结果](results/gnn/ep47-vs-ep48/summary.json) · [启发式基线对比](results/gnn/ep47-vs-baseline/verification.json)

## 项目结构

| 路径 | 内容 |
|---|---|
| [models/](models/) · [config/](config/) | 发布权重、训练记录和默认模型指针 |
| [gnn_pilot/](gnn_pilot/) · [gnn_optimizer.py](gnn_optimizer.py) | 图特征、网络与推理接口 |
| [baseline_solver/](baseline_solver/) · [baseline_optimizer.py](baseline_optimizer.py) | 两种初始化共用的合法化、LP 和退火求解器 |
| [scripts/](scripts/) · [tests/](tests/) | 官方数据准备、训练、对比与回归测试 |
| [results/gnn/](results/gnn/) | GNN 实验的逐例结果、配对布局和复核记录 |
| [my_optimizer.py](my_optimizer.py) · [传统求解器说明](README-classic.md) | 原独立求解器、约 3 秒预算的历史结果及提交包使用方法 |

完整训练和保存布局的复核步骤见 [GNN 复现说明](docs/gnn-reproduction.md)。题目与评估器来自 [IntelLabs/FloorSet](https://github.com/IntelLabs/FloorSet/tree/aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa)，数据使用 [固定版 FloorSet-Lite](https://huggingface.co/datasets/IntelLabs/FloorSet/tree/7dc7c1bec07bcae583a274f5ed95e5be399cb8db)。
