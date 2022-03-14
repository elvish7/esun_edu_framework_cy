from common.ETLBase import ProcessBase 
from common.process_compiler import block_str_generator

class FilteringDataLoader(ProcessBase):
    # Step 0: 
    def packages(self):
        return ['experiments.old_cust_baseline.dataloader.filter_edu.ops', 
                'experiments.old_cust_baseline.dataloader.basic_edu.ops']
    # Step 1: 模塊名稱定義
    def module_name(self):
        return "filtering_dataloader"
    # Step 2.1: 輸入參數定義
    def inputs(self):
        return [
            'rawdata_conn',
            'txn_exclude_start_dt', # cust_txn_exclude
            'txn_exclude_end_dt',
            'new_cust_start_dt', # 6m_new_cust
            'new_cust_end_dt'
        ]
    # Step 2.2: 輸出參數定義 
    def outputs(self):
        return [
            'w107',
            'cust_txn_amt',
            'cust_txn_exclude',
            'featured_funds',
            'available_funds', 
            'last_6m_new_cust'
        ] 
    # Step 4: 串接方式定義
    def connections(self, **kargs):
        conns = block_str_generator(
            'experiments/old_cust_baseline/dataloader/filter_edu/connect.py')
        return conns