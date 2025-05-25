# Code for Prediction Task on Epidemic Dynamic

## Environments Set
To prepare the environment, the following commands are recommended.

```bash
conda create -n <env_name> python=3.8
conda activate <env_name>
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r  requirements.txt
```

## Run

### Continuous Forecasting
Continuous forecasting is designed to predict the next y days based on historical data from the past x days, such as forecasting days 7 to 9 using data from days 0 to 6. The corresponding command-line command is as follows.
```bash
# Running this command will use the dynst model to predict the next 3 days of cases from the past 7 days on GPU 8, and save the results to the results/results_test/test/dataforge/ext_test1 directory
python main.py --xdays 7 --ydays 3 --model dynst --dataset dataforgood --device 8
```

### Alternate-Day Forecasting
Alternate-day forecasting aims to predict specific future days with a day's interval, using historical data points that are also a day apart. For example, it entails forecasting days 9 and 10 based on the data from days 0 to 6. The corresponding command-line command is as follows.
```bash
# Running this command will execute on GPU 8, using the dynst model to predict the third day's cases from the past 7 days, and save the results to the results/results_test/test/dataforge/ext_test1 directory
python main.py --xdays 7 --ydays 1 --shift 2 --model dynst --dataset dataforgood --result-dir test --exp test1 --device 8
```

### Missing Node Forecasting

For missing node forecasting, specify the percentage of observed nodes in the config file `args.cfg` (default `cfg/config.yaml`) using the `node_observed_ratio` parameter. For example:

```yaml
node_observed_ratio: 100.0  # Percentage of observed nodes
```

Setting this to a value below 100 will randomly mask some nodes, allowing the model to predict the missing ones based on the remaining observed nodes.