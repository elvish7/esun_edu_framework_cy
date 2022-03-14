
# 簡介: 

此程式框架的用途是幫助多任務實驗的協作與開發，提供了實驗的訓練、測試、Debug、前處理用的共用模組，並且支援Checkpoint，讓每次實驗產生的最佳模型可以被儲存以供測試使用，也提供logging的機制，以讓訓練過程中的模型的成效可以用Tensorboard來隨時檢視。

於每次實驗，根據我們所制定的規範建立一個全新的實驗設定資料夾，在其中定義模型、模型參數、各任務衡量指標、資料前處理方法，即可與我們的實驗共用模組進行串接整合，讓每一次的實驗都可以被簡易地複製、衡量、調整。

以下將進一步介紹此框架的 1. [資料夾架構](https://github.com/Esun-DF-develop/edu_framework#%E8%B3%87%E6%96%99%E5%A4%BE%E6%9E%B6%E6%A7%8B) 2. [實驗執行方法](https://github.com/Esun-DF-develop/edu_framework#%E5%AF%A6%E9%A9%97%E5%9F%B7%E8%A1%8C%E6%96%B9%E6%B3%95
)

此程式為beta版，若於使用中有疑問或建議，可以於[Issues](https://github.com/Esun-DF-develop/edu_framework/issues)提供給我們，並參考[回饋方式](https://github.com/Esun-DF-develop/edu_framework#%E5%9B%9E%E9%A5%8B%E6%96%B9%E5%BC%8F)，我們將會對此框架進行調整。
另外，在把實驗納入此框架的過程中，麻煩也幫我們填寫[實驗紀錄表](https://github.com/Esun-DF-develop/edu_framework#%E5%AF%A6%E9%A9%97%E8%A8%98%E9%8C%84%E8%A1%A8)，以方便我們追蹤進度。

## 目錄: 

- [簡介](https://github.com/Esun-DF-develop/edu_framework#%E7%B0%A1%E4%BB%8B)
- [實驗記錄表](https://github.com/Esun-DF-develop/edu_framework#%E5%AF%A6%E9%A9%97%E8%A8%98%E9%8C%84%E8%A1%A8)
- [回饋方式](https://github.com/Esun-DF-develop/edu_framework#%E5%9B%9E%E9%A5%8B%E6%96%B9%E5%BC%8F)
- [資料夾架構](https://github.com/Esun-DF-develop/edu_framework#%E8%B3%87%E6%96%99%E5%A4%BE%E6%9E%B6%E6%A7%8B)
- [實驗執行方法](https://github.com/Esun-DF-develop/edu_framework#%E5%AF%A6%E9%A9%97%E5%9F%B7%E8%A1%8C%E6%96%B9%E6%B3%95)
    - [Step 1: 安裝dependencies](https://github.com/Esun-DF-develop/edu_framework#step-1-%E5%AE%89%E8%A3%9Ddependencies)
    - [Step 2: 放置原始資料](https://github.com/Esun-DF-develop/edu_framework#step-2-%E6%94%BE%E7%BD%AE%E5%8E%9F%E5%A7%8B%E8%B3%87%E6%96%99)
    - [Step 3: 測試實驗是否可執行](https://github.com/Esun-DF-develop/edu_framework#step-3-%E6%B8%AC%E8%A9%A6%E5%AF%A6%E9%A9%97%E6%98%AF%E5%90%A6%E5%8F%AF%E5%9F%B7%E8%A1%8C)
    - [Step 4: 建置新實驗](https://github.com/Esun-DF-develop/edu_framework#step-4-%E5%BB%BA%E7%BD%AE%E6%96%B0%E5%AF%A6%E9%A9%97)
        - [1) 實驗模組](https://github.com/Esun-DF-develop/edu_framework#1-%E5%AF%A6%E9%A9%97%E6%A8%A1%E7%B5%84-experiment_modulepy)
        - [2) 模型](https://github.com/Esun-DF-develop/edu_framework#2-%E6%A8%A1%E5%9E%8B-modelpy)
        - [3) 前處理](https://github.com/Esun-DF-develop/edu_framework#3-%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86)
    - [Step 5: 執行新實驗](https://github.com/Esun-DF-develop/edu_framework#step-5-%E5%9F%B7%E8%A1%8C%E6%96%B0%E5%AF%A6%E9%A9%97)
        -  [1) 新實驗測試與模型Debug](https://github.com/Esun-DF-develop/edu_framework#1-%E6%96%B0%E5%AF%A6%E9%A9%97%E6%B8%AC%E8%A9%A6%E8%88%87%E6%A8%A1%E5%9E%8Bdebug)
        -  [2) 模型訓練與測試](https://github.com/Esun-DF-develop/edu_framework#2-%E6%A8%A1%E5%9E%8B%E8%A8%93%E7%B7%B4%E8%88%87%E6%B8%AC%E8%A9%A6)
        -  [3) TensorBoard-訓練成效監控](https://github.com/Esun-DF-develop/edu_framework#3-tensorboard-%E8%A8%93%E7%B7%B4%E6%88%90%E6%95%88%E7%9B%A3%E6%8E%A7)
        -  [4) CPU/GPU加速](https://github.com/Esun-DF-develop/edu_framework#4-gpucpu%E5%8A%A0%E9%80%9F)
- [小工具](https://github.com/Esun-DF-develop/edu_framework#%E5%B0%8F%E5%B7%A5%E5%85%B7)
    - [blockPrinting](https://github.com/Esun-DF-develop/edu_framework#utilsblockprint)

# 實驗記錄表
|負責團隊(玉山/中研)|實驗名稱|模型名稱|任務中英名稱|已建構完成實驗資料夾|fastdebug運作無誤|fit1batch運作無誤|train運作無誤|參數調整完成|最佳模型test無誤|最佳模型.ckpt路徑|
|--|--|:--:|--|--|--|--|--|--|--|--|
|玉山|ex4|ETRNN|下月消費總金額(objmean)、下月消費次數(tscnt)、下月是否消費(label_0)|V|V|V|V| | | |

# 回饋方式 

產學合作過程中，關於框架與資料的問題或是回饋，可以在[Issues](https://github.com/Esun-DF-develop/edu_framework/issues)提供給我們，我們會盡快回覆。另外，我們提供[四種labels](https://github.com/Esun-DF-develop/edu_framework/labels)，也請幫我們標上:  

- data_question: 如果對於玉山提供的資料有疑問 請打上這個Label。
- framework_bug: 如果在使用玉山框架時，遇到難解的BUG 請打上這個Label。
- framework_question: 如果對於玉山產學框架的使用方式有疑問 請打上這個Label。
- framework_enhancement: 如果對於玉山產學框架有新功能的建議 請打上這個Label。


# 資料夾架構 
以下為資料夾架構，標上 * 的檔案為實驗執行後，才會生成的檔案或資料夾；標上 V 的為特定實驗專屬之實驗檔或檔案夾；

標上🟠為須由實驗者[自行建置](https://github.com/Esun-DF-develop/edu_framework#step-4-%E5%BB%BA%E7%BD%AE%E6%96%B0%E5%AF%A6%E9%A9%97)之實驗檔或資料夾；標上🟢為須執行之檔案，包含[實驗運行](https://github.com/Esun-DF-develop/edu_framework#step-5-%E5%9F%B7%E8%A1%8C%E6%96%B0%E5%AF%A6%E9%A9%97)。

```
.
├── data                                                 # 實驗資料
|    ├── source                                            # 存放原始資料 
|    ├── *V [experiment_group_name]                      # 存放特定類型的實驗(e.g., rnn)所需之資料
|    |      ├── * tmp                                      # 中繼檔
|    |      └── * result                                   # 結果檔
|    ├── *V [experiment_name]                            # 存放特定實驗(e.g., ex1)用資料
|    |      ├── * tmp                                      # 中繼檔
|    |      └── * result                                   # 結果檔
|    ├── *V [experiment_name]
|    └── ... 
|
├── common                                               # 共用模組           
|    ├── ETLBase.py                                        # 前處理用小工具 
|    ├── utils.py                                          
|    ├── pl_module.py                                    # 內含多任務實驗共用之抽象模組 
|    └── __init__.py 
|    
├── experiments                                          # V 實驗模組 
|    ├──V🟠[experiment_name]                              # V 特定實驗之實驗模組
|    |      ├──V🟠__init__.py          
|    |      ├──V🟠experiment_module.py                    # V 實驗設定模組                                
|    |      ├──V🟠model.py                                # V 模型模組 
|    |      └──V🟠preprocess                              # V 客製化的前處理模組資料夾
|    |             ├──V🟠config.py                        # V 資料前處理參數與串接設定 (see ex4)
|    |             ├──V🟠ops.py                           # V 資料前處理函數          (see ex4)
|    |             └──V🟠connect.py                       # V 資料前處理串接方式定義   (see ex4)
|    ├──V🟠[experiment_name]
|    └── ...
|
├── *checkpoint                                          # 儲存模型暫存檔 
|    ├── *V [experiment_name]                              # 儲存特定實驗的模型暫存檔
|    |      ├── *V epoch964-loss0.00.ckpt                    # V 最佳模型的暫存檔
|    |      └── *V last.ckpt                                 # V 最後一個epoch的模型暫存檔   
|    ├── *V [experiment_name]
|    └── ... 
|
├── *logs                                                # 存放訓練LOG: 各任務by-iteration的成效、實驗參數、模型架構圖 
|    └── tensorboard 
|           ├── *V [experiment_name]                       # 儲存特定實驗的LOG 
|           |       ├── *V version_0                         # 儲存第1次實驗的LOG 
|           |       ├── *V version_1                         # 儲存第2次實驗的LOG 
|           |       └── ... 
|           ├── *V [experiment_name]                      
|           └── ... 
├──🟢run_project.py                                      # 實驗執行檔 
├── requirements.txt 
└── ReadMe.md 
```


# 實驗執行方法 

## Step 1: 安裝dependencies 

首先將相關套件進行安裝。

執行: 

```bash
sh install_packages.sh
(sudo) apt install graphviz 
pip install graphviz
```

若要使用Nvidia GPU，須安裝GPU版本pytorch，詳情請見：https://pytorch.org。

## Step 2: 放置原始資料 

資料下載與存取方式詢問專案的負責人，存取後請將資料表放置於data/source中，已供實驗執行時自動載入。
`ex4`的原始資料參考data/source中的jupyter notebook進行下載。

## Step 3: 測試實驗是否可執行 

執行FastDebug: 
`python run_project.py -m fastdebug -e ex4` 


此程式會對experiments/ex4資料夾所定義之實驗進行debug。過程中會對原始資料進行前處理，並將結果與佔存檔儲存於`data/sample`、`data/rnn/tmp`、`data/rnn/result`、`data/ex4/tmp`、`data/ex4/result`，接著，資料處理完後，就會進行1個step的training和validation，以快速驗證模型、程式的運作正常。

## Step 4: 建置新實驗: 

可以複製[ex4](https://github.com/udothemath/edu_framework/tree/main/experiments/ex4)資料夾，必將其改為實驗者欲命名的實驗名稱（e.g., ex4)，並修改其中的`experiment_module.py`/`model.py`/`preprocess/config.py`/`preprocess/connect.py`/`preprocess/ops.py`。其中`experiment_module.py`為實驗模組，`model.py`為模型，`preprocess`內檔案為前處理程式。

以下將以ex4為範例，分別說明此三類程式的建構方式: 

### 1) 實驗模組 (`experiment_module.py`)

實驗模組必須包含三個Class: `ExperimentConfig`、`ExperimentalMultiTaskDataModule`和`ExperimentalMultiTaskModule`，`ExperimentConfig`定義了實驗名稱、實驗參數、以及最佳模型的暫存檔名稱，`ExperimentalMultiTaskDataModule`定義了資料前處理、訓練以及測試資料，`ExperimentalMultiTaskModule`定義了多任務模型、任務名稱、任務目標函數以及任務成效衡量指標。

以下將針對撰寫方式進一步說明: 

**1. ExperimentConfig**:
```python 

class ExperimentConfig:
    # @ExperimentDependent 
    best_model_checkpoint = 'epoch07-loss0.00.ckpt' # 最佳模型的暫存檔名稱

    name = experiment_name  # 實驗名稱，需和資料夾名稱相同(e.g., ex1) 

    experiment_parameters = {                    # 實驗參數 
        "model_parameters": {                      # 模型參數: 會放到model.py的參數。
            "data_independent":{                     # 和資料無關的模型參數，(e.g., hidden_dims 中間層維度, n_layers 層數, ...) 
                'hidden_dims': 64,              
                'n_layers': 2, 
                'cell': 'LSTM', 
                'bi': False
            },
            "data_dependent": {                      # 和資料相關之模型參數，(e.g., dense_dims 輸入之數值型特徵維度, use_chid 是否使用顧客ID為特徵, out_dims 模型輸出的維度)  
                'dense_dims': dense_dims, 
                'sparse_dims': sparse_dims,
                'use_chid': use_chid, 
                'out_dims': out_dims,
                'class_outputs': class_outputs 
            }
        },
        "training_parameters":{                      # 訓練參數，(e.g., dropout, warmup_epochs, annealing_cycle_epochs) 
            "dropout": 0.5, 
            "warmup_epochs": 5, 
            "annealing_cycle_epochs": 40
        }
    }
```

* 其中，dropout/warmup_epochs/annealing_cycle_epochs若不提供，預測則分別會給0.5, 5, 40。warmup_epochs和annealing_cycle_epochs會輸入以下learning rate scheduler，進行learning rate動態調整，以加速模型訓練。

```python 
LinearWarmupCosineAnnealingLR(optimizer, 
 warmup_epochs=self._warmup_epochs, 
 max_epochs=self._annealing_cycle_epochs 
)
```
**2. ExperimentalMultiTaskDataModule**:


```python 
class ExperimentalMultiTaskDataModule(BaseMultiTaskDataModule):
    # @blockPrinting
    def prepare_data(self):
        self.train_dataset = train_dataset.run()[0] 
        self.test_dataset = test_dataset.run()[0]
```

* 於prepare_data定義train_dataset和test_dataset。此兩個物件須為torch.utils.data的TensorDataset物件。

**3. ExperimentalMultiTaskModule**:

```python 
import torch.nn.functional as F
class ExperimentalMultiTaskModule(BaseMultiTaskModule):

    def config_model(self, model_parameters, dropout): # 此處引入model.py的最top-level的nn.Module，此nn.Module吃model_parameters和dropout兩個參數，後面將於model.py進一步說明建構方式。
        return MultiTaskModel(
                model_parameters,
                dropout = dropout
            )
            
    def config_task_names(self):                                 # 此處定義模型輸出所對應之任務名稱 
        return ['objmean', 'tscnt', 'label_0']
        
    def config_loss_funcs(self): 
        return [F.mse_loss, F.mse_loss, F.binary_cross_entropy]  # 此處定義各任務之目標函數 
    
    def config_task_metrics(self):                               # 此處定義個任務之衡量指標名稱 
        return {
            'objmean': ['mse', 'mae'], 
            'tscnt': ['mse', 'mae'], 
            'label_0': ['acc', 'auc']
        }
    
    def config_metric_calculators(self):                         # (optional) 定義個指標名稱所對應之指標計算元件。若有新指標(非mse/mae/acc/auc)才需實作此函數。
        from torchmetrics import MeanSquaredError, MeanAbsoluteError, Accuracy, AUROC
        return {
            'mse': lambda: MeanSquaredError(compute_on_step=False), 
            'mae': lambda: MeanAbsoluteError(compute_on_step=False), 
            'acc': lambda: Accuracy(compute_on_step=False),
            'auc': lambda: AUROC(compute_on_step=False, pos_label=1)
        }
    def batch_forward(self, batch):                              # 此處根據模型的forward方式定義一個batch的forward，以供共用模組的training/validation/testing將ground_truth和outputs套用到loss上。
        x_dense, x_sparse, objmean, tscnt, label_0 = batch
        outputs = self(x_dense, x_sparse)
        ground_truths = objmean, tscnt, label_0
        return outputs, ground_truths
```

### 2) 模型 (`model.py`)

model.py的top-level module依照nn.Module既有方法進行實作，但須注意__init__需吃`model_parameters`和`dropout`，model_parameters為一dictionary，
內容為`ExperimentConfig.experiment_parameters["model_parameters"]` 中`"data_independent"`以及`"data_dependent"`底下的所有參數。

```python 
class MultiTaskModel(torch.nn.Module):
    def __init__(self, model_parameters, dropout=0.5):
        super(MultiTaskModel, self).__init__()

        # Load parameters: 
        hidden_dims = model_parameters['hidden_dims']
        n_layers = model_parameters['n_layers'] 
        cell = model_parameters['cell'] 
        bi = model_parameters['bi'] 
        dense_dims = model_parameters['dense_dims'] 
        sparse_dims = model_parameters['sparse_dims'] 
        use_chid = model_parameters['use_chid'] 
        out_dims = model_parameters['out_dims'] 
        class_outputs = model_parameters['class_outputs']

        # Build model blocks: 
        self.rnn = ET_Rnn(
            dense_dims, 
            sparse_dims, 
            hidden_dims, 
            n_layers=n_layers, 
            use_chid=use_chid,
            cell=cell, 
            bi=bi, 
            dropout=dropout
        )
        
        self.mlps = nn.ModuleList([
            MLP(
                self.rnn.out_dim, hidden_dims=[self.rnn.out_dim // 2], out_dim=od
            ) for od in out_dims
        ])

        # Parameters used in forward 
        self.class_outputs = class_outputs

    def forward(self, *x):
        x_dense, x_sparse = x 
        logits = self.rnn(x_dense, x_sparse)
        outs = []
        for mlp, is_class in zip(self.mlps, self.class_outputs):
            out = mlp(logits)
            if is_class:
                out = torch.sigmoid(out)
            outs.append(out)
        return outs


```

### 3) 資料前處理 

使用我們的`common/ETLBase.py`的`ProcessBase`物件進行處理模組的參數與函數定義。


#### 3.1) 資料前處理工具: ETLBase 簡介

為了能讓資料轉換為能夠輸入模型的形式，實驗建置過程中，往往會需要耗費許多的心力來進行資料的前處理，而對於不同的模型版本，又有可能會有相應的不一樣的前處理方式，隨著實驗的增加，前處理的程式也相應得變得越來越難以維護。另外，建立前處理程式的過程中，往往涉及到大量冗長的資料轉換，因此在開發過程中也容易因資料轉換而耽誤了開發時程。

因此，我們提供前處理工具（`ETLBase`)，希望不只讓前處理程式更易於理解，也可以開發更快速。此前處理工具可以透過視覺化的方式，將前處理過程中的模塊、模塊的輸入、輸出，以及模塊之間的串連方式，以[有向圖(DAG)](https://zh.wikipedia.org/wiki/File:Tred-G.svg)的方式呈現，讓前處理的步驟與邏輯可以一目了然。另外，此工具也加入了資料中繼檔暫存功能，讓前處理過程中的中間產物，可以被以檔案的方式儲存起來，讓後續使用此中間產物的處理模塊可以快速載入，進行後續模塊的調整。

此工具使用方式為繼承我們的`common/ETLBase`中的`ProcessBase`類別，並覆蓋其中的函數，達到規格化地定義前處理模塊、模塊串接方式、前處理輸入輸入參數的功能。以下我們將對此工具的使用方式進行簡單說明，詳細操作方式請參考[Jupyter Notebook - Tutorial of Pipeline Tools.ipynb](https://github.com/Esun-DF-develop/edu_framework/blob/main/Tutorial%20of%20Pipeline%20Tools.ipynb)。

#### 3.2) 前處理定義方式

假設前處理涉及兩個參數a,b，我們想要讓c = a + b, d = a + c, e = d + d，最後輸出三行的pandas.DataFrame，每一行的內容為e，我們可以以下面方式進行串接:

```python 
from common.ETLBase import ProcessBase
class PreProcess(ProcessBase):
    # Step 1: 模塊名稱定義
    def module_name(self):
        return "tutorial_preprocess"
    # Step 2.1: 輸入參數定義    
    def inputs(self):
        return [
            'a', 
            'b'
        ]
    # Step 2.2: 輸出參數定義 
    def outputs(self):
        return ['table'] 
    
    # Step 3: 模塊定義 
    def define_functions(self, pipe):
        import numpy as np
        import pandas as pd 
        @pipe._func_
        def plus_a_b(a=0,b=0):
            return a+b
             
        @pipe._func_
        def repeat(a,b=3):
            return np.repeat(a,b)
        
        @pipe._func_
        def to_dataframe(seq):
            return pd.DataFrame(seq)
        
    # Step 4: 串接方式定義
    def connections(self, **kargs):
        conns = [
            'c = plus_a_b(a,b)', 
            'd = plus_a_b(a,c)', 
            'e = plus_a_b(d,d)',
            'e_array = repeat(e)',
            'table = to_dataframe(e_array)'
        ]
        return conns
```

以下為步驟說明：

* 模塊名稱定義
	
```python
    def module_name(self):
        return "tutorial_preprocess"
```

* 輸入輸出參數定義
	
```python
    def inputs(self):
        return [
            'a', 
            'b'
        ]
        
    def outputs(self):
        return ['table'] 
```

* 模塊定義 
	
```python
    def define_functions(self, pipe):
        import numpy as np
        import pandas as pd 
        @pipe._func_
        def plus_a_b(a=0,b=0):
            return a+b
             
        @pipe._func_
        def repeat(a,b=3):
            	return np.repeat(a,b)
            	
        @pipe._func_
        def to_dataframe(seq):
        	return pd.DataFrame(seq)
```

* 串接方式定義
	
```python
    def connections(self, **kargs):
        conns = [
            'c = plus_a_b(a,b)', 
            'd = plus_a_b(a,c)', 
            'e = plus_a_b(d,d)',
            'e_array = repeat(e)',
            'table = to_dataframe(e_array)'
        ]
        return conns
```

#### 3.2) 使用.py定義前處理模組與串接方式：

```python
from common.ETLBase import ProcessBase
from common.process_compiler import block_str_generator

class PreProcess(ProcessBase):
    def module_name(self):
        return "tutorial_preprocess"
    def inputs(self):
        return [
            'a', 
            'b'
        ]
    def outputs(self):
        return ['table']
    def packages(self):
        return ['tutorial.ops']
        
    def connections(self, **kargs):
        conns = block_str_generator('tutorial/connect.py')
        return conns
```

說明：
1. 將`define_functions`中函式定義於一獨立.py檔中(參見：`tutorial/ops`)
2. 覆寫`ProcessBase`的`packages`以載入ops.py
3. 將`connections`中python字串撰寫於一獨立.py中(參見：`tutorial/connect.py`)  
4. 使用`common.process_compiler.block_str_generator`載入connect.py



#### 3.3) 前處理輸入參數設定方式

假設我們希望我們的前處理輸入值a=1,b=2，可以透過以下方式進行設定 

```python
preprocess = PreProcess() 
preprocess.config(a=1, b=2, verbose=True) 
``` 
#### 3.4) 前處理執行方式

前處理在串接時不會直接執行，只有要實際獲取結果時，才會進行執行。

獲取方式如下：

```
preprocess.pipe.table.get(verbose=False)
>>> 
	0
0	8
1	8
2	8

``` 

並且除了最終輸出結果可以進行獲取之外，前處理過程中定義的中間參數都可以獲取:

```
preprocess.pipe.e_array.get(verbose=False)
>>> array([8, 8, 8])
```

#### 3.5) 前處理視覺化介紹  

在開發過程中可以透過以下方式對前處理進行視覺化，幫助理解與呈現前處理的步驟與流程：

```python
preprocess.pipe.view(summary=False)  
```

![image](https://github.com/Esun-DF-develop/edu_framework/blob/main/tutorial/image/tutorial.svg)
 
我們也提供Dependency Hightlight的功能，幫助辨識各前處理模塊的前繼模塊。
 
 ```python
 preprocess.pipe.view_dependency('c', summary=False)
``` 

![image](https://github.com/Esun-DF-develop/edu_framework/blob/main/tutorial/image/dependency.svg)


#### 3.6) 中繼檔暫存功能使用方式：

若想要將前處理過程產物進行暫存，操作步驟如下：

1. 在定義前處理模組（i.e., 建立ProcessBase物件時)，於`connection`中定義中繼檔名稱，指定方式如下：

```python
    def connections(self, **kargs):
        conns = [
            'c = plus_a_b(a=a,b=b)', 
            'd = plus_a_b(a,c)', 
            'e = plus_a_b(d,d)',
            ('e_array = repeat(e)','e_array.npy'),
            ('table = to_dataframe(e_array)','table.feather')
        ]
        return conns
```

目前支援`pandas.DataFrame`和`numpy`的暫存。(`pandas.DataFrame`儲存格式為`.feather`，`numpy.array`儲存格式為`.npy`)



若是以載入.py的方式建置`connections`，可於以下方式於模塊後方指定中繼檔名稱：

```python
c = plus_a_b(a,b)
d = plus_a_b(a,c)
e = plus_a_b(d,d)
e_array = repeat(e)
('e_array.npy')
table = to_dataframe(e_array)
('table.feather')
```

若有多個輸出，請用逗點隔開：

```python
table, array = two_output_example(x)
('table.feather', 'array.npy')
```


2. 初始化ProcessBase物件時，指定`save_tmp=True`，並指定儲存資料夾名稱：

```python
preprocess = PreProcess(save_tmp=True, experiment_name='example') 
preprocess.config(a=1, b=2, verbose=True) 
``` 
中繼檔(`e_array.npy`、`table.feather`)預設會被存在`data/[experiment_name]/[module_name]/tmp`資料夾中(這裏[`module_name=tutorial_preprocess`])，建議不同處理模塊需有不同名稱避免暫存檔存取衝突。另外，暫存檔主檔名須與輸出參數名稱相同。

若未指定`experiment_name`，中繼檔會被儲存在`data/[module_name]/tmp`資料夾中。

3. 執行前處理運算時，指定`load_tmp=True`，若未指定，則前處理會重新執行。 

```
preprocess.pipe.table.get(verbose=True, load_tmp=True)
```


## Step 5: 執行新實驗: 

當新的實驗建構完成，建議依以下順序進行debug與訓練: 

### 1) 新實驗測試與模型Debug:

**fastdebug**

首先，執行fastdebug，確保即使模型與實驗設定修改後，訓練與驗證皆能順利執行，執行時會進行一個epoch的測試: 

`python run_project.py -m fastdebug -e [實驗資料夾名稱]` 

**fit1batch** 

接著，為了確保模型設計是合理的，讓模型overfit一個訓練的batch，正常的狀況，loss要能夠持續下降，loss的監控參考 [TensorBoard-訓練成效監控](https://github.com/udothemath/edu_framework/blob/main/README.md#3-tensorboard-%E8%A8%93%E7%B7%B4%E6%88%90%E6%95%88%E7%9B%A3%E6%8E%A7)。

`python run_project.py -m fit1batch -e [實驗資料夾名稱] (-l [log_dir])` 

執行過程中，會存放TensorBoard的Log於資料夾`./logs/tensorboard/[實驗資料夾名稱]/version_[x]`。

### 2) 模型訓練與測試

**train** 

執行訓練: 

`python run_project.py -m train -e [實驗資料夾名稱] (-l [log_dir])`  

模型的validation/training performance的監控參考 [TensorBoard-訓練成效監控](https://github.com/Esun-DF-develop/edu_framework#3-tensorboard-%E8%A8%93%E7%B7%B4%E6%88%90%E6%95%88%E7%9B%A3%E6%8E%A7)。

訓練過程中，表現最佳的模型以及最後一個epoch的模型暫存檔(.ckpt)會被保存於`./checkpoint/[實驗資料夾名稱]`中，每增加一個epoch，就會被更新一次。

**test**

若要用測試資料檢測模型的最終結果，可以將`./checkpoint/[實驗資料夾名稱]`中，最佳模型的.ckpt檔指定給`experiment_module.py`中`ExperimentConfig`的`best_model_checkpoint`參數，接著執行: 

`python run_project.py -m test -e [實驗資料夾名稱]`

### 3) TensorBoard-訓練成效監控 

於terminal輸入`tensorboard --logdir logs/tensorboard/[實驗資料夾名稱]`，即可於瀏覽器開啟TensorBoard查看訓練狀況(http://localhost:6006/ )。

* _Note: _若是執行fit1batch，其中val的圖表顯示的為在一個訓練的batch上衡量的成效。

若要修改log資料夾，可以於執行時，用`-l`指定新log資料夾路徑，e.g., 

`python run_project.py -m [fit1batch/train] -e [實驗資料夾名稱] -l MyDirectory`，此時，則於terminal輸入`tensorboard --logdir MyDirectory/[實驗資料夾名稱]`，即可查看該實驗結果。

若有進行實驗參數調整，可於 http://localhost:6006/#hparams 查看各實驗參數下的模型成效。

### 4) GPU/CPU加速：

```
python run_project.py -m [fit1batch/train] -e [實驗資料夾名稱] -g [GPU數量] -c [使用的cpu worker數量]
```
# 小工具 


## utils.blockPrint
用來把函數中會print出來的資訊都隱藏起來。

使用方法: 

```python 
from common.utils import blockPrinting

@blockPrinting
def function_to_block():
      print('message to be blocked')
```

# MODIFICATION: 

* [X] 資料的下載方式 @甫璋。
    - 新增資料庫連線方式的說明文件於專案資料夾內
    - 新增範例資料表
* [X] 建立'資料問題討論'與'框架使用問題討論'的Issue tracking方式
* [X] 行內相容性問題調整   
    - [X] cpu 環境
    - [X] gpu 環境
