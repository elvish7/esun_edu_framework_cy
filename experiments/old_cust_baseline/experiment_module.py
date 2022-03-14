from experiments.old_cust_baseline.dataloader.utils import get_conn 
from experiments.old_cust_baseline.dataloader.utils import get_data_start_dt

def old_cust_CF_baseline(today = '2019-06-30', mode='all'):
    """
    說明: 
    
    此為舊顧客的推薦程式，新顧客推薦邏輯後續提供。
    
    mode可以用來選擇要串接的階段多寡。主要包含四個階段: 
    1. dataloading: 把CF模型會用到的資料從DB抓進來。
    2. preprocess: 資料前處理，使資料符合CF模型能輸入的形式。
    3. model: CF模型，包含基金相似性計算與矩陣相乘 
    4. rank_n_filter: 將模型輸出的分數進行排序，並進行初步的篩選(非法規or公司規定的篩選)，可以自由調整。
    (5. 最終final_filtering會進行法規和公司規定的篩選，因此final_filtering資料夾請保持不變)
    mode == 'all': 串接所有final filtering之前的推薦流程
    mode == 'dataload': 串接dataloading 
    mode == 'preprocess': 串接preprocess以前的推薦流程 
    mode == 'model': 串接model以前的推薦流程 
    
    """
    assert mode == 'all' or mode == 'dataload' or mode == 'preprocess' or mode == 'model'
    
    rawdata_conn = get_conn('edu')
    before_18m_dt = get_data_start_dt(today, 18)
    before_6m_dt = get_data_start_dt(today, 6)
    before_3m_dt = get_data_start_dt(today, 3)
    before_1m_dt = get_data_start_dt(today, 1)

    from experiments.old_cust_baseline.dataloader.basic_edu.config import DataLoader
    dataloader = DataLoader(experiment_name='old_cust_baseline') 
    dataloader.config(
        rawdata_conn=rawdata_conn,
        nav_start_dt=before_1m_dt,
        nav_end_dt=today,
        txn_start_dt=before_18m_dt, 
        txn_end_dt=today,
        txn_amt_end_dt=today,
        verbose=True) 
    pipe = dataloader
    if mode == 'preprocess' or mode == 'model' or mode == 'all':
        from experiments.old_cust_baseline.preprocess_edu.config import PreProcess  
        preprocess=PreProcess(
            required_process=dataloader, 
            experiment_name='old_cust_baseline') 
        preprocess.config(
            dig_start_dt=before_3m_dt,
            dig_end_dt=today
        )
        pipe = preprocess 
    if mode == 'model' or mode == 'all':
        from experiments.old_cust_baseline.model.config import Model  
        model = Model(
            required_process=preprocess, 
            experiment_name='old_cust_baseline'
        ) 
        model.config()
        pipe = model 
    if mode == 'all':
        from experiments.old_cust_baseline.dataloader.filter_edu.config import FilteringDataLoader
        filter_dataloader = FilteringDataLoader(
            required_process=model,
            experiment_name='old_cust_baseline'
        )  
        filter_dataloader.config(
            rawdata_conn=rawdata_conn,
            txn_exclude_start_dt=before_3m_dt, # cust_txn_exclude
            txn_exclude_end_dt=today,
            new_cust_start_dt=before_6m_dt, # 6m_new_cust
            new_cust_end_dt=today
        )
        from experiments.old_cust_baseline.rank_n_filter.config import Rank_n_Filter
        rank_n_filter_module = Rank_n_Filter(
            required_process=filter_dataloader,
            experiment_name='old_cust_baseline'
        ) 
        rank_n_filter_module.config()
        pipe = rank_n_filter_module
    return pipe