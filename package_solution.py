"""Build and inspect a small handoff archive; exclude datasets and the venv."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    result_dir = ROOT / "results"
    report = json.loads((result_dir / "validation.json").read_text(encoding="utf-8"))
    verification = json.loads((result_dir / "verification.json").read_text(encoding="utf-8"))
    solver = ROOT / "FloorSet" / "iccad2026contest" / "my_optimizer.py"
    assert report["cases"] == report["feasible"] == verification["verified_cases"] == 100
    assert all(v == 0 for v in verification["hard_violations"].values())
    assert hashlib.sha256(solver.read_bytes()).hexdigest() == report["optimizer_sha256"]
    old = json.loads((result_dir / "validation_before_boundary.json").read_text(encoding="utf-8"))
    rows = report["results"]
    times = [r["runtime_seconds"] for r in rows]
    improvement = (1 - report["total_score"] / old["total_score"]) * 100
    readme = f'''# FloorSet 求解器与验证结果

本项目实现题目要求的 `solve()` 接口，返回按模块编号排列的
`(x, y, width, height)` 列表。所有评分均由未修改的官方评测器计算。
这是一份可以运行和继续优化的竞赛解法；尚未进行隐藏测试或提交到官方平台。

## 验证结果

验证日期：2026-10-08。官方仓库提交：`{report["upstream_commit"]}`。

| 项目 | 结果 |
|---|---:|
| 公开验证案例 | 100，覆盖 21–120 个模块 |
| 可行案例 | 100 / 100 |
| 本地加权成本，越低越好 | {report["total_score"]:.6f} |
| 平均求解时间 | {report["mean_runtime_seconds"]:.3f} 秒 |
| 最短 / 最长求解时间 | {min(times):.3f} / {max(times):.3f} 秒 |
| 重叠违规 | {verification["hard_violations"]["overlap"]} |
| 面积违规 | {verification["hard_violations"]["area"]} |
| 固定尺寸、预放置违规 | {verification["hard_violations"]["dimension"]} |
| MIB 违规 | {verification["soft_violations"]["mib"]} |

加入边界优先布局前的完整验证成本为 {old["total_score"]:.6f}，
最终版本降低约 {improvement:.1f}%。该比较的运行时间预算均为每案例约 3 秒。
原始官方模板只测试了 21 和 120 模块两例：两例均不可行，成本均为 10。
模板在 120 模块案例上耗时约 279 秒。没有对模板运行完整 100 例。

本地评测把 RuntimeFactor 固定为 1，不能据此声称获得正式比赛排名。
质量仍与参考解存在差距，仍有分组和边界软约束违规；详细数据见 JSON。
保存的 100 个布局已重新加载，由官方评测器逐一复核成本和全部硬约束。

## 算法

1. 将固定模块的宽高和预放置模块的坐标、宽高设为不可修改值。
   普通软模块保持目标面积；可兼容的 MIB 成员共用宽高。
2. 使用连接图的加权拉普拉斯方程，结合固定端口和预放置模块，计算初始位置。
   再用连续优化减少连线距离和模块重叠。
3. 使用矩形边缘候选点构造严格无重叠布局。另构造一份边界优先布局：
   端口范围提供初始外框估计，带边界要求的预放置模块提供确定的外框边；
   先处理角、边模块，再安排内部模块。
   外框只用于产生候选解，并非新增硬约束。
4. 在线性规划中保留模块分离关系、已有分组接触和已满足的边界条件，
   优化中心之间的加权曼哈顿距离并压缩外包矩形。
5. 做无重叠局部移动和面积保持的形状调整；比较候选并保留最好的一份。
   最终再次检查全部硬约束。

算法仅使用 `solve()` 的参数，不读取参考布局、数据生成器或数据文件。
参考解由官方评测器用于计算评分，以及提取规则允许提供的固定模块信息。
内部搜索使用由输入估计的质量尺度，最终成绩使用官方参考值重新计算。
坐标采用二进制网格，避免贴边时浮点误差产生微小空隙。

默认 `FLOORSET_SECONDS=3`。这是搜索预算，最后一次操作与最终检查可能让
实际时间略超过预算。可改为 9 等值投入更多搜索，速度因素也会影响正式评分。
计时搜索在不同机器或负载下可能选择不同候选，结果并非逐位确定。

## 当前工作区直接运行

在本 README 所在目录执行：

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 test_optimizer.py
.\\.venv\\Scripts\\python.exe -X utf8 benchmark.py
.\\.venv\\Scripts\\python.exe -X utf8 make_report.py
```

使用官方命令：

```powershell
$solverPython = (Resolve-Path '.venv\\Scripts\\python.exe').Path
Push-Location FloorSet\\iccad2026contest
& $solverPython -X utf8 iccad2026_evaluate.py --validate my_optimizer.py
& $solverPython -X utf8 iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
Pop-Location
```

`benchmark.py` 同样调用官方 ContestEvaluator，额外保存每道题的布局、
代码 SHA256、运行时间和结果。`make_report.py` 可独立复核保存的布局。
`test_optimizer.py` 包含 8 项行为测试，检查不可移动模块、分组接触、MIB、
面积、多个障碍物、缺失固定信息以及本身不可行的输入。

## 压缩包在新目录中使用

压缩包顶层的 `my_optimizer.py` 是实际提交文件。先准备官方仓库和环境，
再复制到官方竞赛目录：

```powershell
git clone https://github.com/IntelLabs/FloorSet.git FloorSet
git -C FloorSet checkout {report["upstream_commit"]}
Copy-Item my_optimizer.py FloorSet\\iccad2026contest\\my_optimizer.py
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-solver.txt
.\\.venv\\Scripts\\python.exe -X utf8 test_optimizer.py
.\\.venv\\Scripts\\python.exe -X utf8 benchmark.py
```

为复现本报告，应检出上述官方提交；当前官方仓库中已经附带公开验证集。
`environment.txt` 记录本机实际依赖版本。运行需要 SciPy，官方原始依赖文件
没有列出它，因此请使用本包的 `requirements-solver.txt`。

可以在评测前设置预算与线程数量：

```powershell
$env:FLOORSET_SECONDS='3'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
```

## 文件

- `my_optimizer.py`：可提交的求解器。
- `results/validation.json`：全部 100 个验证布局及官方成本。
- `results/verification.json`：保存布局的独立复核结果。
- `results/validation_overview.png`：质量、软约束和运行时间图。
- `results/layout_120.png`、`.svg`：120 模块案例的布局。
- `results/validation_before_boundary.json`：边界优化前的完整结果。
- `results/template_smoke.json`：原始模板的两个对照案例。
- `submission_manifest.json`：每个打包文件的 SHA256。

官方规则与工具来源：
https://github.com/IntelLabs/FloorSet/tree/main/iccad2026contest
'''
    readme_path = ROOT / "README_solution.md"
    readme_path.write_text(readme, encoding="utf-8")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    environment = ROOT / "environment.txt"
    environment.write_text(f"# Python {sys.version.split()[0]} on {sys.platform}\n" + freeze, encoding="utf-8")
    files = {"my_optimizer.py": solver, "README_solution.md": readme_path,
             "requirements-solver.txt": ROOT / "requirements-solver.txt",
             "environment.txt": environment}
    for name in ("benchmark.py", "test_optimizer.py", "make_report.py", "package_solution.py"):
        files[name] = ROOT / name
    for name in ("validation.json", "verification.json", "validation_overview.png",
                 "layout_120.png", "layout_120.svg", "validation_before_boundary.json", "template_smoke.json"):
        files["results/" + name] = result_dir / name
    manifest = {"date": "2026-10-08", "upstream_commit": report["upstream_commit"],
                "local_score": report["total_score"], "feasible_cases": report["feasible"],
                "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}}
    manifest_path = ROOT / "submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files["submission_manifest.json"] = manifest_path
    archive = ROOT / "floorset_solution.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for name, path in files.items():
            out.write(path, name)
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
        assert len(check.namelist()) == len(files)
        for name, digest in manifest["sha256"].items():
            assert hashlib.sha256(check.read(name)).hexdigest() == digest
        compile(check.read("my_optimizer.py").decode("utf-8"), "my_optimizer.py", "exec")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "files": len(files), "local_score": report["total_score"],
                      "improvement_percent": improvement}, ensure_ascii=False))


if __name__ == "__main__":
    main()
