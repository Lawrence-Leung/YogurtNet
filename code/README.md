# README YogurtNet类图像神经网络预测IR Drop使用说明

2023.11.29 v1.0



### 概要

本程序可执行脚本总共分为4部分

1：数据预处理脚本 utils_preprocess.py

2：pix2pix模型推理脚本 Class_pix2pix_infer.py

3：pyramid网络推理脚本 Class_pyramid.py

4：输出最终文件脚本 utils_postprocess.py

5：批处理调用脚本 autoprocess.py

（其中utils_preprocess.py和Class_pix2pix_infer.py是由autoprocess.py调用）

完成整个程序的运行，需要逐步运行上述提到的四个脚本，以得到最终的 pred_static_ir

**该程序不包含对压缩原数据文件解压功能。如果存在压缩，请手动解压到对应目录。**

**该程序的最终输出的预测文件pred_static_ir_xxx存放路径为初始数据文件夹内，（即跟 *inst.power.rpt，min_path_res.rpt，eff_res.rpt存放在一起）**

`我们是类图像处理方法，所以使用的数据源文件目录位于 /home/phlexing/test_data/data，而不是/home/phlexing/test_data/dspf。`

### 程序目录结构

**yogurtnet**

​               |**pix2pix**

​                               |Class_pix2pix_infer.py

​               |**pyramid**

​                               |Class_pyramid.py

​               |**Voc_Table**

​               |autoprocess.py

​               |utils_postprocess.py

​               |utils_preprocess.py

### 运行

进入项目目录：`cd /home/eda230622/yogurtnet`

#### 第一步，运行autoprocess.py脚本，对原始数据进行批处理

```shell
python autoprocess.py --entrance=需要批量预处理数据的入口，例如：/data-zero-riscy-0/out --pyscript=utils_preprocess.py
```

#### 第二步，运行autoprocess.py脚本，对预处理后的数据进行批量pix2pix推理

```shell
python autoprocess.py --entrance=需要批量推理数据的入口，例如：/data-zero-riscy-0/out --pyscript=pix2pix/Class_pix2pix_infer.py
```

#### 第三步，运行Class_pyramid.py脚本，对pix2pix推理后的结果进行进一步的处理

注意！！！：这一步需要修改Class_pyramid.py脚本的数据入口路径**hardcode**为同上的--entrance的值（例如：/data-zero-riscy-0/out），该hardcode位于**第321行的变量entrance_path**

然后直接不携带任何参数运行Class_pyramid.py

```shell
python pyramid/Class_pyramid.py
```

#### 第四步，运行utils_postprocess.py 脚本，输出最终文件

```
python utils_postprocess.py --entrance=需要批量推理数据的入口，例如：/data-zero-riscy-0/out
```

同时，此步骤将会删除所有中间文件。只留下pred_static_ir_xxx文件

### 运行例子

进入项目目录：`cd /home/eda230622/yogurtnet`

#### 第一步，运行autoprocess.py；

```shell
python autoprocess.py --entrance=/home/ed0622/my_test_data --pyscript=utils_preprocess.py
```

output log:

```
(pytorch2.0) -bash-4.2$ python autoprocess.py --entrance=/home/ed0622/my_test_data/ --pyscript=./utils_preprocess.py
开始搜索 '/home/eda230622/my_test_data/' 以找到包含 'pulpino_top.t.power.rpt' 的目录
['/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa5_p_5_fi_ap_', '/home/eda230622/my_test_data/zero-riscy_freq_50_m_fpu_55_fpa_2.0_p_0_fi_ar_', '/home/eda230622/my_test_data/zero-ry_freq_50_mp_1_fpu_60_fpa_1.5_p_0_fi_ap_']
开始运行脚本...
现在是第1个,正在运行脚本 './utils_preprocess.py' 使用数据目录: /h/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_i_ap_
现在是第2个,正在运行脚本 './utils_preprocess.py' 使用数据目录: /h/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_55_fpa_2.0_p_i_ar_
现在是第3个,正在运行脚本 './utils_preprocess.py' 使用数据目录: /h/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_i_ap_
(pytorch2.0) -bash-4.2$ /home/eda230622/my_test_data/zero-riscy_f
chip type is zero-riscy
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_55_fpa_2
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1
chip type is zero-riscy
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1
chip type is zero-riscy
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1
(pytorch2.0) -bash-4.2$ size: 256
size: 256
size: 256
NCPR Done.
NCPR Done.
NCPR Done.
infer done...result in output_coord2_final.json
infer Done.
ncpr loaded.
coord2 loaded
矩阵已保存到 /home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_55_fpa_2.0_p_0_fi_ar_/static_ir_PREDATA.npy
Convent Done
infer done...result in output_coord2_final.json
infer Done.
ncpr loaded.
coord2 loaded
矩阵已保存到 /home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_5_fi_ap_/static_ir_PREDATA.npy
Convent Done
infer done...result in output_coord2_final.json
infer Done.
ncpr loaded.
coord2 loaded
矩阵已保存到 /home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_0_fi_ap_/static_ir_PREDATA.npy
Convent Done
```

static_ir_PREDATA.npy 为第一步预处理输出的文件（该步骤本地电脑测试时间约为3min）

#### 第二步，运行autoprocess.py；(--pyscript=pix2pix/Class_pix2pix_infer.py)

```shell
python autoprocess.py --entrance=/home/eda230622/my_test_data/ --pyscript=./pix2pix/Class_pix2pix_infer.py
```

output log:

```
(pytorch2.0) -bash-4.2$ python autoprocess.py --entrance=/home/eda230622/my_test_data/ --pyscript=./pix2pix/Class_pix2pix_infer.py
开始搜索 '/home/eda230622/my_test_data/' 以找到包含 'pulpino_top.inst.power.rpt' 的目录
['/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_5_fi_ap_', '/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu0_p_0_fi_ar_', '/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_0_fi_ap_']
开始运行脚本...
现在是第1个,正在运行脚本 './pix2pix/Class_pix2pix_infer.py' 使用数据目录: /home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.p_
现在是第2个,正在运行脚本 './pix2pix/Class_pix2pix_infer.py' 使用数据目录: /home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_55_fpa_2.r_
现在是第3个,正在运行脚本 './pix2pix/Class_pix2pix_infer.py' 使用数据目录: /home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.p_
(pytorch2.0) -bash-4.2$ /home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_55_fpa_2.0_p_0_fi_ar_
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_5_fi_ap_
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_0_fi_ap_
chip type is zero-riscy
chip type is zero-riscy
chip type is zero-riscy
initialize network with normal
initialize network with normal
initialize network with normal
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_55_fpa_2.0_p_0_fi_ar_ static_ir_POSTDATApredmidPR.npy
yes
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_0_fi_ap_ static_ir_POSTDATApredmidPR.npy
yes
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_5_fi_ap_ static_ir_POSTDATApredmidPR.npy
yes
All Finished! -------------------
Exist Indexes =  1
Effective Indexes =  1
All Finished! -------------------
Exist Indexes =  1
Effective Indexes =  1
All Finished! -------------------
Exist Indexes =  1
Effective Indexes =  1
```

#### 第三步，运行Class_pyramid.py

```shell
python pyramid/Class_pyramid.py
```

output log:

```
(pytorch2.0) -bash-4.2$ python ./pyramid/Class_pyramid.py
Warning: logging configuration file is not found in logger/logger_config.json.
['/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_5_fi_ap_', '/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu0_p_0_fi_ar_', '/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_0_fi_ap_']
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_5_fi_ap_
chip type is zero-riscy
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_5_fi_ap_/static_ir_POSTDATApredmidPR.npy
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_5_fi_ap_ static_ir_POSTDATApredmidPR.npy
yes
All Finished! -------------------
Exist Indexes =  1
Effective Indexes =  1
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_55_fpa_2.0_p_0_fi_ar_
chip type is zero-riscy
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_55_fpa_2.0_p_0_fi_ar_/static_ir_POSTDATApredmidPR.npy
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_55_fpa_2.0_p_0_fi_ar_ static_ir_POSTDATApredmidPR.npy
yes
All Finished! -------------------
Exist Indexes =  1
Effective Indexes =  1
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_0_fi_ap_
chip type is zero-riscy
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_0_fi_ap_/static_ir_POSTDATApredmidPR.npy
/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_0_fi_ap_ static_ir_POSTDATApredmidPR.npy
yes
All Finished! -------------------
Exist Indexes =  1
Effective Indexes =  1
```

#### 第四步，运行utils_postprocess.py

```
python utils_postprocess.py --entrance=/home/eda230622/my_test_data/
```

output log:

```
(pytorch2.0) -bash-4.2$ python utils_postprocess.py --entrance=/home/eda230622/m
                          y_test_data/
Running
处理完成：/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_
                          0_fi_ap_/pred_static_ir_xxx
处理完成：/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_60_fpa_1.5_p_
                          5_fi_ap_/pred_static_ir_xxx
处理完成：/home/eda230622/my_test_data/zero-riscy_freq_50_mp_1_fpu_55_fpa_2.0_p_
                          0_fi_ar_/pred_static_ir_xxx
Done!
```

注意：同时，此步骤将会删除所有程序中间生成的文件。只在原数据文件夹生成最终的预测文件pred_static_ir_xxx

### 第三方库说明

#### ntlk库

您可以从以下网址下载 `nltk-3.8.1-py3-none-any.whl` 文件：

https://files.pythonhosted.org/packages/source/n/nltk/nltk-3.8.1-py3-none-any.whl

关于使用 `pip` 安装ntlk的 `.whl` 文件，一般过程是首先将 `.whl` 文件下载到您的本地机器，然后运行带有文件路径的 `pip install` 命令。命令通常如下所示：

```bash
cd /home/eda2306**/ext_library/
pip install nltk-3.8.1-py3-none-any.whl
```

#### gensim库

源码下载地址：

https://files.pythonhosted.org/packages/77/68/074333a52f6fa82402332054ca0dfa721f7dcfa7eace313f64cdb44bacde/gensim-4.3.2.tar.gz

安装（已经将源码的tar.gz格式文件解压到对应目录中）：

```bash
cd /home/eda2306**/ext_library/gensim-4.3.2
python setup.py install
```
