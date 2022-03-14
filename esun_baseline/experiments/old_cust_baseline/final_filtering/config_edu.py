# The following connection should not be modified 
# 下面的篩選邏輯是公司規定必須保留，不需修改 

from dev.etl3.common.ETLBase import ProcessBase 
from dev.etl3.common.process_compiler import block_str_generator
class FinalFilter(ProcessBase):
    # Step 0: 
    def packages(self):
        return [
            'dev.etl3.dataloader.basic_edu.ops',
            'dev.etl3.dataloader.filter_edu.ops', 
            'dev.etl3.rank_n_filter.ops', 
            'dev.etl3.process_utils.risk_management'
        ]
    # Step 1: 模塊名稱定義
    def module_name(self):
        return "final_filtering"
    # Step 2.1: 輸入參數定義
    def inputs(self):
        return [
            'cust_item_rating_matrix',
            'cust_item_rank_matrix',
            'item2nid',
            'user2nid',
            'rawdata_conn'
        ]
    # Step 2.2: 輸出參數定義 
    def outputs(self):
        return [
           'output'
        ]
    # Step 4: 串接方式定義
    def connections(self, **kargs):
        conns = block_str_generator(
            'dev/etl3/final_filtering/connect.py')
        return conns