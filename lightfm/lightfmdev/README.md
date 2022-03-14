# LightFM Package for Local Developement

## Running the package

- Recommend as a pure collabarative filtering model
```bash
python3 local_lightfm_main.py --train w103_train.csv --evaluation w103_test.csv
```
- Recommend as a hybrid collabarative filtering model
```bash
python3 local_lightfm_main.py --train w103_train.csv --evaluation w103_test.csv --user_ft cm_customer_m.csv --item_ft w106.csv
```
## Parameters
* `--user_ft`：Use meta user features if specified path/ if not, user features will only be identity ids。
* `--item_ft`：Use meta item features if specified path/ if not, item features will only be identity ids。
* `--dim`: Specify the dimensions of feature embeddings。 Default dim=128
* `--epoch`: Specify the number of epochs。Default epoch=20。

