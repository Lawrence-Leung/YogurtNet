![YogurtNetLogoWithCaption](doc/YogurtNet.png)

## Project Description

YogurtNet is a work participating in the 5th National College Student Integrated Circuit EDA Elite Challenge in 2023. It is aimed at the corporate proposition of Phlexing Company: SoC power network static voltage drop prediction based on machine learning. This project achieves accurate prediction of the static voltage drop (IR Drop) of the integrated circuit power network by integrating image processing technology, unsupervised learning and neural network technology.

## Technical Overview

The YogurtNet project is built on Python and PyTorch. The main technical components include:

- **Word2Vec**: Map instance names to the name coordinate system through clustering method to support subsequent processing.

- **Pix2Pix**: Convert 17 channels of raw data into 2 channels of predicted data, using mature image conversion technology to optimize the data processing process.

- **YogurtPyramid**: A self-developed shallow neural network focused on further optimizing and approximating prediction results.

Model training relies on relevant data from five mainstream open source chips and performs well on the evaluation indicators provided by the competition questions. The specific performance in the server is that the average wall time is 69.94 seconds and the average MAE value is 5.2784.

## Achievements and Honors

YogurtNet won the third prize in the national finals, and the paper based on this work has been successfully submitted to the 2024 3rd International Conference on Electronics and Integrated Circuit Technology (EICT 2024), Nanjing, China. The citation information is listed below:

Liang, L., Xian, Y., Guo, S., & Xie, Z. (2024). YogurtNet: Enhanced machine learning approach for voltage drop prediction. Journal of Physics: Conference Series, 2810(1), 012002. https://doi.org/10.1088/1742-6596/2810/1/012002

## Team Members

- **Lawrence Leung**: Responsible for algorithm design.

- **Yuxiang Xian (Github user: @Silhouette6 )**: Responsible for code framework design.

## Repository Contents

- `/code`: Contains the main project code and the corresponding README.md file, which provides instructions for using the software.

- `/cal_metrics`: Quantitative evaluation tool provided by the contest organizer.

- `/reference_papers`: References.

- `design_report.pdf`: Full entry report.

- `original_contest_problem.pdf`: Full text of the original contest question.

- `defense_presentation.pdf`: Final defense presentation.

## Environmental requirements

- Python 3.8

- PyTorch 2.0.1

## Version Information

Current software version: v0.1.0

## Instructions for use

Please refer to the detailed instructions provided in `/code/README.md` to configure and run YogurtNet.

## Acknowledgments

Special thanks to everyone who supported us and everyone who worked hard on this project. We look forward to communicating and collaborating with more friends who are interested in EDA and machine learning.

Thanks to the guidance of instructors Zhuoming Xie and Huaien Gao, and the strong support of the School of Integrated Circuits of my alma mater.
