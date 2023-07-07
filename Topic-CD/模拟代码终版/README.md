## 模拟代码说明

#### 主要程序文件

1. 生成数据程序：`gendata.py`，在命令行下执行

   `python gendata.py --experiment_dir {模拟数据存放文件夹} --prefix {生成模拟数据的前缀}`，如

   `python gendata.py --experiment_dir experiments/single --prefix 0`

2. 模型算法程序：`dphmm.py`，在命令行下执行

   `python dphmm.py --experiment_dir {模拟数据存放文件夹} --prefix {生成模拟数据的前缀}`，如

   `python gendata.py --experiment_dir experiments/single --prefix 0`

3. 在执行`gendata.py`与`dphmm.py`前需要在目标文件夹有参数配置文件，如`experiments/single/params.json`

4. 也可以选择直接跑`run.py`程序，不用事先配置`params.json`文件，在`run.py`文件中修改for循环中的参数即可。在命令行下执行

   `python run.py`

#### 其他程序文件

- `utils.py`：一些模型算法程序中用到的函数工具。

- `getPrecisionAndRecall.py`：从模拟结果中计算出精准度、召回率等指标。在命令行下执行

  `python getPrecisionAndRecall.py --input_path {output.txt文件路径}`，如

  `python getPrecisionAndRecall.py --input_path experiments/single/output.txt`