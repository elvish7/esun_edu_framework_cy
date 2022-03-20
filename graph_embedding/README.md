# Graph Embedding Package

## Running the package

- Recommend as a pure collabarative filtering model
```bash
python3 graph_embedding_main.py --date 2019-06-30 --eval_duration 1m
```
## Parameters
* `--date`：`YYYY-MM-DD`，Recommendation date。
* `--eval_duration`：`1m` 或 `7d`，The length of your evaluation data。
* `--mode`: Specify the optimization algorithm。Default 'warp'。

There is also a script provided: `graph_embedding_exp.sh` 
