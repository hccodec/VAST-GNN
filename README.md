# Code for VAST-GNN

## Environment Setup

To prepare your environment, run the following commands:

```bash
conda create -n vastgnn python=3.8 -y
conda activate vastgnn
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## Running Experiments

### Alternate-Day Forecasting

Alternate-day forecasting predicts specific future days at intervals, using historical data points also spaced apart (e.g., forecast days 9 and 10 using data from days 0–6).

```bash
# This command uses the VAST_GNN model to predict the third day's cases from the past 7 days on GPU 8.
python main.py --xdays 7 --ydays 1 --shift 2 --model vast_gnn --dataset dataforgood --country England --result-dir test --exp test1 --device 8
```

### Missing Node Forecasting

For missing node forecasting, set the percentage of observed nodes directly via the command line using the `--node-observed-ratio` parameter:

```bash
python main.py --xdays 7 --ydays 1 --model vast_gnn --dataset dataforgood --country England --node-observed-ratio 80
```

Setting this below 100 will randomly mask some nodes, allowing the model to predict missing nodes based on the observed ones.

---

## Quick Model Evaluation

To quickly validate a trained model, simply unzip `checkpoints.zip` in the project root directory. Then run `test.py` with the desired arguments. The script will automatically extract and use the checkpoints.

Example usage:
```bash
python test.py --country EN --model vast_gnn --node-observed-ratio 80 -y 3
```
- `--country` Country code (EN, FR, ES, IT, JP, h1n1, h3n2, BY, BV)
- `--model` Model type (lstm, mpnn_lstm, mpnn_tl, vast_gnn)
- `--node-observed-ratio` Percentage of observed nodes (e.g., 80 or 50)
- `--ydays` Prediction days ahead (e.g., 3, 7, 14)

All arguments are optional and have default values. For more options, run:
```bash
python test.py --help
```

After running, the script will automatically load the corresponding model and output evaluation metrics such as MAE and HITS@10.

For more details, please refer to the comments in the code.