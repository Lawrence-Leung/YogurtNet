![YogurtNetLogoWithCaption](doc/YogurtNet.png)

## 项目简介

YogurtNet 是参加 2023 年第 5 届全国大学生集成电路 EDA 精英挑战赛的作品，针对行芯（Phlexing）公司的企业命题赛题：基于机器学习的 SoC 电源网络静态压降预测。该项目通过融合类图像处理技术、非监督学习和神经网络技术，实现了对集成电路电源网络静态压降（IR Drop）的精确预测。

## 技术概述

YogurtNet 系统基于 Python 和 PyTorch 构建，主要技术组件包括：

- **Word2Vec**：通过聚类方法，将实例名称映射到名称坐标系中，以支持后续的处理。
- **Pix2Pix**：将 17 个通道的原始数据转化为 2 个通道的预测数据，利用成熟的图像转换技术优化数据处理流程。
- **YogurtPyramid**：一个自主开发的浅层神经网络，专注于进一步优化和逼近预测结果。

模型训练依赖于五种主流开源芯片的相关数据，并在赛题提供的评价指标上表现出色，在服务器中具体表现为平均 wall time 为 69.94 秒，平均 MAE 值为 5.2784。

## 成果与荣誉

YogurtNet 在全国总决赛中荣获三等奖，并基于该作品撰写的论文已成功投稿至 2024 第 3 届国际电子与集成电路技术会议（EICT 2024），待论文在 Journal of Physics 正式出版后将更新引用信息。

## 团队成员

- **Lawrence Leung**：负责算法设计。
- **Yuxiang Xian (Github 用户：@Silhouette6 )**：负责代码框架设计。

## 仓库内容

- `/code`：包含工程主体代码和相应的 README.md 文件，后者提供了软件的使用说明。
- `/cal_metrics`：赛题方提供的量化评价工具。
- `/reference_papers`：参考文献。
- `design_report.pdf`：完整的参赛作品报告。
- `original_contest_problem.pdf`：原始赛题全文。
- `defense_presentation.pdf`：决赛答辩演示文稿。

## 环境要求

- Python 3.8
- PyTorch 2.0.1

## 版本信息

当前软件版本：v0.1.0

## 使用说明

请参考 `/code/README.md` 中提供的详细说明来配置和运行 YogurtNet。

## 致谢

特别感谢所有支持我们的人，以及所有为这个项目付出努力的人。我们期待与更多对 EDA 和机器学习感兴趣的朋友交流和合作。

感谢指导老师 Zhuoming Xie 与 Huaien Gao 的指导，以及母校集成电路学院的大力支持。
