用于计算average_mae和average_cc的测试脚本。

python main.py --golden_path ./data --pred_path ./pred_static_ir_report

### 最终测试时，要求同学们给出1.自己代码的绝对路径，2.代码的运行方式，3.输出文件夹的绝对路径（格式参考本例的pred_static_ir_report）

我们会用linux的time命令来统计同学们程序的实际运行时间(wall time)，然后按照给定的输出文件夹路径，用本脚本计算metric。

最后统计所有队伍的metric和runtime结果，计算得分。