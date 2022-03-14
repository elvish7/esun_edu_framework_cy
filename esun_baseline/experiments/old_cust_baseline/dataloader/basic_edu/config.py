from common.ETLBase import ProcessBase 
from common.process_compiler import block_str_generator

class DataLoader(ProcessBase):
    # Step 0: 
    def packages(self):
        return 'experiments.old_cust_baseline.dataloader.basic_edu.ops'
    # Step 1: 模塊名稱定義
    def module_name(self):
        return "dataloader"
    # Step 2.1: 輸入參數定義
    def inputs(self):
        return [
            'rawdata_conn',
            'nav_start_dt',
            'nav_end_dt',
            'txn_start_dt', #
            'txn_end_dt',
            'txn_amt_end_dt'
        ]
    # Step 2.2: 輸出參數定義 
    def outputs(self):
        return [
            'w106',
            'w118', 
            'w103',
            'cust_txn_amt'
        ]
    
    # Step 4: 串接方式定義
    def connections(self, **kargs):
        conns = block_str_generator('experiments/old_cust_baseline/dataloader/basic_edu/connect.py')
        return conns