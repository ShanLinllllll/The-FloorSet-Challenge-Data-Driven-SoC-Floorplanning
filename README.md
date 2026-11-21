# FloorSet · SoC 布局求解器

给定芯片模块的面积、连接关系、端口和布局约束，求解器为每个模块生成一个矩形。预训练 GNN 预测初始布局，随后通过合法化、线性规划和模拟退火生成最终布局。仓库已包含推理权重，默认入口会自动加载并校验权重。

## 跑通第一个布局

在 Windows PowerShell 中执行以下命令。已克隆仓库的队友可直接从创建 Python 3.12 环境开始。

```powershell
git clone https://github.com/ShanLinllllll/The-FloorSet-Challenge-Data-Driven-SoC-Floorplanning.git
cd The-FloorSet-Challenge-Data-Driven-SoC-Floorplanning

py -3.12 -m venv .venv
$python = (Resolve-Path .venv/Scripts/python.exe).Path
& $python -m pip install -r requirements-step1-lock.txt
& $python scripts/prepare_official_data.py source
& $python scripts/prepare_official_data.py validation
& $python -I -B -X utf8 examples/solve_layout.py --case 0
```

案例 0 有 **21 个模块**。输出是一条 JSON 记录，`block_count` 为 21，`positions` 按模块编号给出 21 组 `[x, y, width, height]`，其中 `x, y` 是矩形左下角坐标。把 `--case 0` 改为 0–99 中的其他编号，即可求解对应公开案例。默认使用可用的 CUDA 设备，也支持 CPU 推理。[可运行示例](examples/solve_layout.py)

## 接入自己的数据

从仓库根目录导入 `MyOptimizer`。批量求解时可创建一个实例，依次传入多个设计。

```python
from gnn_optimizer import MyOptimizer

solver = MyOptimizer()
positions = solver.solve(
    block_count=block_count,
    area_targets=area_targets,
    b2b_connectivity=b2b_connectivity,
    p2b_connectivity=p2b_connectivity,
    pins_pos=pins_pos,
    constraints=constraints,
    target_positions=target_positions,
)
```

输入沿用 FloorSet 官方接口。每条模块连接或端口连接包含两个索引和一个权重。`target_positions` 的每行对应一个模块：预放置模块填写完整矩形，固定形状模块填写宽高，其余未知字段填 `-1`。示例中的 `solver_inputs` 可直接把官方验证样本转换为上述参数。[输入转换](scripts/floorset_io.py) · [求解器接口](gnn_optimizer.py)

## 训练与验证成果

| 项目 | 结果 |
|---|---|
| 模型 | 4 层 GNN，隐藏维度 128，431,303 个参数 |
| 训练数据 | FloorSet-Lite 全量训练划分，8,980 个来源文件，1,005,760 条样本 |
| 开发集 | 1,120 条样本，选择损失 0.074771 |
| 独立留出集 | 1,120 条样本，选择损失 0.064747，模块对相对位置准确率 **84.38%** |
| 公开布局验证 | 21–120 个模块的 100 个案例，**100/100 满足硬约束** |
| 布局质量与耗时 | 加权 cost **3.258741**，平均 **1.951 秒/例**，其中编码与推理平均 **13.73 ms/例** |

布局结果来自 RTX 4060 Laptop GPU 上的验收运行，求解种子为 42，每例目标预算 2 秒，编码与推理计入预算。本地评分使用官方评估器，运行时间因子固定为 1。保存布局重新评分后，100 例的成本和可行性均与运行记录一致。[默认模型配置](config/gnn_default.json) · [训练与复现](docs/gnn-reproduction.md)
