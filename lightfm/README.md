# LightFM Package

## Running the package

- Recommend as a pure collabarative filtering model
```bash
python3 lightfm_main.py --date 2019-06-30 --eval_duration 1m
```
- Recommend as a hybrid collabarative filtering model
```bash
python3 lightfm_main.py --date 2019-06-30 --eval_duration 1m --user_ft --item_ft
```
## Parameters
* `--date`：`YYYY-MM-DD`，Recommendation date。
* `--eval_duration`：`1m` 或 `7d`，The length of your evaluation data。
* `--user_ft`：Use meta user features if specified / if not, user features will only be identity ids。
* `--item_ft`：Use meta item features if specified / if not, item features will only be identity ids。
* `--dim`: Specify the dimensions of feature embeddings。 Default dim=128
* `--epoch`: Specify the number of epochs。Default epoch=20。

There is also a script provided: `lightfm_exp.sh` 
