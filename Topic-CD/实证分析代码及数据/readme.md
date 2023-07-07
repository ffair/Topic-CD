1. 文档说明
* 实验配置：experiments/cell/params.json
* 模型文件：dphmm.py
* 辅助文件：utils.py，utils_pre.py
* 实证数据预处理：experiments/cell/preprocess/
* 运行实证实验：run_cell.sh


2. 实证分析-使用说明
* S1, 创建实验并配置参数
  * 新建目录 experiments/cell
  * 进入目录 experiments/cell，新建params.json并配置参数
  * 退出到experiments/的上级目录，在utils.py设置随机数种子
* S2，预处理，生成实证数据
  * 进入目录 experiments/cell/preprocess
  * 执行命令 bash preprocess.sh
* S3 运行实证实验
  * 退出到experiments/的上级目录
  * 执行命令 nohup bash run_cell.sh &
* S4, 查收结果
 * 目录 experiments/cell下会生成logs.txt任务日志)，output.txt，phi.txt(主题词)，vocab.json，文件名前缀可在run_cell.sh中设置
