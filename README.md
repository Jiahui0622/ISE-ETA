# ISE548 第 1 组：数据与代码

小组作业用到的原始数据和处理数据的脚本，方便大家看懂数据是怎么一步步分析出来的。

## 文件夹

| 文件夹 | 内容 |
|---|---|
| `data/` | PHMSA 配气管网事故原始文件（1,592 条记录、468 个字段），ZIP 原件、解压后的 TXT 和官方字段说明 PDF |
| `src/` | 分析脚本，见下表 |
| `results/` | 脚本算出的结果：事件树统计、两个神经网络的交叉验证记录和预测、最终模型权重（`.pt`） |
| `figures/` | 由结果画出的 6 张图，`en` 英文、`zh` 中文 |

## 脚本

| 脚本 | 作用 |
|---|---|
| `src/eta_pipeline.py` | 事件树：从原始记录筛出 1,271 起事故，分到六条路径，算各分支概率和置信区间，再按八类事故原因分别统计 |
| `src/check_raw_data.py` | 用另一套代码从原始数据重算事件树的计数，核对上一个脚本的结果 |
| `src/mlp_pytorch.py` | 神经网络：训练两个网络。网络一判断事故会不会造成死亡或住院，网络二判断泄漏会不会点燃 |
| `src/verify_mlp_inference.py` | 复核两个神经网络保存的结果 |
| `src/generate_bilingual_figures.py` | 把结果画成图 |
| `src/results_reader.py` | 读取 `results/` 里的结果，供画图用 |

## 方法

事件树：每起事故依次看三个问题，点燃了没有、爆炸了没有、有没有人死亡或住院，落到六条路径中的一条。分支概率 = 走这条分支的事故数 / 到达这个节点的事故数，路径概率是沿途分支概率相乘。

神经网络：两个结构相同的小网络，输入 → 16 → 8 → 1。用 10 折交叉验证重复 5 次检验效果，并和“只看事故原因”、逻辑回归比较。

## 运行

需要 Python 3.13 左右的版本，先装依赖：

```bash
pip install -r requirements.txt
```

在本文件夹打开终端，核对结果（不改任何文件）：

```bash
python -B src/check_raw_data.py
python -B src/verify_mlp_inference.py
```

从头重算：

```bash
python -B src/eta_pipeline.py
python -B src/mlp_pytorch.py
python -B src/generate_bilingual_figures.py
```

随机种子固定，重算结果和 `results/` 里保存的相同。重算会覆盖 `results/` 和 `figures/`，想保留原样先复制一份。
