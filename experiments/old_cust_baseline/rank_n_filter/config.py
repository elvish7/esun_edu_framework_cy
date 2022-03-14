from common.ETLBase import ProcessBase 
from common.process_compiler import block_str_generator
class Rank_n_Filter(ProcessBase):
    # Step 0: 
    def packages(self):
        return ['experiments.old_cust_baseline.rank_n_filter.ops']
    # Step 1: 模塊名稱定義
    def module_name(self):
        return "rank_n_filter"
    # Step 2.1: 輸入參數定義
    def inputs(self):
        return [
        ]
    # Step 2.2: 輸出參數定義 
    def outputs(self):
        return [
           'cust_item_rating_matrix_out', 
           'cust_item_rank_matrix_out', 
           'user2nid', 
           'item2nid_out'
        ]
    # Step 4: 串接方式定義
    def connections(self, **kargs):
        conns = block_str_generator(
            'experiments/old_cust_baseline/rank_n_filter/connect.py'
        )
        return conns