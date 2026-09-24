# Paper 3 Python 分析工作流

这份说明写给第一次接触 Python 的使用者。整个项目已经整理成类似 Stata
`.do` 文件的顺序：编号文件负责具体分析，`main.py` 负责按顺序运行全部代码。

## 先记住三件事

1. 要运行全部分析：在 PyCharm 中打开 `workflow/main.py`，点击绿色 Run 按钮。
2. 要修改变量、样本或回归：修改 `prep/` 或 `analyses/` 中对应的编号文件。
3. 要修改数据、结果或日志的位置：修改 `workflow/config.py`。

平时不需要修改 `__init__.py`、`_deps.py` 和 `direct_run.py`。

## 文件结构

```text
paper3_python/
└── workflow/
    ├── main.py                    # 全部分析的总入口
    ├── config.py                  # 路径和运行模式设置
    ├── pipeline.py                # 安排各步骤的运行顺序
    ├── direct_run.py              # 让每个文件可在 PyCharm 直接 Run
    ├── _deps.py                   # 集中导入常用 Python 库
    ├── __init__.py                # 把 workflow 声明为 Python 包
    ├── requirements.txt           # 本项目需要的第三方库
    ├── prep/
    │   ├── 01_data_prep.py        # 清洗原始数据、生成基础变量
    │   ├── 01_nonfarm_outcomes.py # 生成非农就业结果变量
    │   ├── 05_friction_index.py   # 构造 friction index
    │   └── 01_household_panel.py  # 构造家庭面板数据
    ├── analyses/
    │   ├── 02_farm_main.py        # 农业劳动主回归
    │   ├── 03_nonfarm_main.py     # 非农结果主回归
    │   ├── 04_decomposition.py    # 分解分析
    │   ├── 06_event_study.py      # 事件研究
    │   ├── 07_placebo.py          # 安慰剂检验
    │   └── 08_bootstrap.py        # Bootstrap 推断
    └── output/
        └── README.md              # 输出文件说明
```

## 点击 Run 后发生什么

```text
main.py
  ↓
direct_run.py：准备 Python 导入路径
  ↓
config.py：读取数据路径、输出路径和运行模式
  ↓
pipeline.py：创建 Paper3Pipeline，并依次安排任务
  ↓
数据准备：01_data_prep → 01_nonfarm_outcomes
          → 05_friction_index → 01_household_panel
  ↓
实证分析：02_farm_main → 06_event_study → 08_bootstrap
          → 04_decomposition → 03_nonfarm_main → 07_placebo
  ↓
结果写入 results/，运行记录写入 logs/
```

前一个步骤产生的数据会保存在 `pipeline.py` 的 `self.state` 中，再交给后一个
步骤使用。可以把 `self.state` 理解成一个带标签的共享资料箱。

## 六个核心 Python 文件

### `main.py`：总开关

这是运行整个项目时需要打开的文件。它只做两件事：

1. 告诉 Python 项目文件夹在哪里。
2. 调用 `run_pipeline_mode("all")`，运行全部步骤。

一般不在这里写数据清洗或回归代码。

### `config.py`：设置面板

`Paper3Config` 保存项目的公共设置，包括：

- `data_dir`：原始数据所在文件夹。
- `output_dir`：表格、图片和 Excel 结果所在文件夹。
- `log_dir`：运行日志所在文件夹。
- `run_mode`：本次要运行哪些分析。

换电脑或移动数据后，通常只需要在这里更新路径。

### `pipeline.py`：项目调度器

`Paper3Pipeline` 是一个 `class`（类），负责：

- 按正确顺序运行数据准备和实证分析。
- 检查运行模式是否有效。
- 把各步骤产生的数据和结果保存在 `self.state` 中。
- 保证 Bootstrap 等步骤开始前，所需的主回归和事件研究已经完成。

这里决定“先运行什么、后运行什么”，具体回归公式仍在编号文件中。

### `direct_run.py`：直接运行适配器

编号文件位于不同子文件夹中。直接点击 Run 时，Python 有时找不到其他模块。
这个文件会自动添加项目路径，再调用 pipeline 或单个准备步骤。因此现在不需要
先在 Terminal 输入命令。

它主要处理“怎样启动代码”，不处理经济学分析本身。

### `_deps.py`：公共工具箱

这里集中导入项目经常使用的库，例如：

- `pandas`：读取和处理表格数据。
- `numpy`：数值计算。
- `matplotlib`：绘图。
- `linearmodels`：面板回归。
- `scipy`、`sklearn`：统计检验和主成分分析。

这样编号文件可以共用同一套依赖，不必反复写相同的导入语句。

### `__init__.py`：包的入口说明

这个文件告诉 Python：`workflow` 是一个可以被导入的包。它还把
`Paper3Config` 和 `Paper3Pipeline` 暴露为 workflow 的主要接口。

初学阶段一般不需要修改这个文件。它不是数据初始化程序，也不会自动清洗数据。

## `prep/01_data_prep.py` 的作用

这是最基础的数据准备文件，相当于 Stata 项目中的 `01_data_prep.do`。它会：

- 读取 `paper3_data.dta`。
- 保留研究需要的年份和省份。
- 处理特殊缺失值。
- 构造农业劳动时间、政策处理、事件时间和人口特征等变量。
- 返回后续回归所需的数据及变量列表。

如果要改变样本筛选、变量定义、处理组或对照组，主要修改这个文件。它负责把
原始数据整理成“可以回归的数据”，但本身不负责主回归。

## 怎样在 PyCharm 运行

### 运行完整项目

1. 在左侧文件树中打开 `paper3_python/workflow/main.py`。
2. 确认 PyCharm 使用项目的 `.venv` Python interpreter。
3. 点击编辑器右上角的绿色三角形 Run。
4. 在 Run 窗口查看进度；完成后到 `results/` 查看结果。

### 只运行一个步骤

打开需要的编号文件，例如 `analyses/02_farm_main.py`，点击 Run。每个编号文件
都包含直接运行入口，会自动准备它所需要的数据。

第一次运行前，如果 PyCharm 提示缺少库，可在 Python Interpreter 设置中根据
`workflow/requirements.txt` 安装依赖。

## 常见修改应该去哪里

| 想做的事情 | 应修改的文件 |
|---|---|
| 更换原始数据或输出路径 | `config.py` |
| 修改样本、年份、省份或变量定义 | `prep/01_data_prep.py` |
| 修改非农结果变量 | `prep/01_nonfarm_outcomes.py` |
| 修改 friction index | `prep/05_friction_index.py` |
| 修改农业劳动主回归 | `analyses/02_farm_main.py` |
| 修改非农回归 | `analyses/03_nonfarm_main.py` |
| 修改分解分析 | `analyses/04_decomposition.py` |
| 修改事件研究 | `analyses/06_event_study.py` |
| 修改安慰剂检验 | `analyses/07_placebo.py` |
| 修改 Bootstrap | `analyses/08_bootstrap.py` |
| 修改完整流程的先后顺序 | `pipeline.py` |

## 初学者需要认识的 Python 写法

- `import`：使用另一个文件或第三方库提供的功能。
- `def name(...):`：定义一个可以重复调用的函数。
- `class Paper3Pipeline:`：定义一类对象，用于集中保存数据和组织操作。
- `self.state["hh"]`：从共享字典中取出名为 `hh` 的结果。
- `Path(...)`：表示文件或文件夹路径。
- `if __name__ == "__main__":`：只有直接点击 Run 运行当前文件时，才执行下面的代码；被其他文件导入时不会自动执行。

## 推荐的修改和检查顺序

1. 一次只修改一个编号文件。
2. 先直接 Run 这个编号文件，确认没有报错。
3. 检查生成的回归表、图片、Excel 和日志。
4. 最后 Run `main.py`，确认完整流程仍能运行。

报错时先看 Run 窗口最下面几行。最后一行通常是错误类型，向上找到第一个指向
`workflow/` 内文件的行，就能定位发生问题的文件和行号。
