# Code for link prediction task for COVID-19

## Environments Set
To prepare the environment, the following commands are recommended.

```bash
conda create -n <env_name> python 3.8
pip install -r  requirements.txt
```


## Run
To run the code, execute th following command.

```bash
# 运行该命令将在 8 号 GPU 执行使用 dynst 模型从历史 7 天预测未来 3 天病例，并将结果保存至 results/results_test/test/dataforgood/exp_test1 目录中
python main.py --xdays 7 --ydays 3 --model dynst --dataset dataforgood --result-dir test --exp test1 --device 8
```