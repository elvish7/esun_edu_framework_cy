from common.ETLBase import ProcessBase 
from common.process_compiler import block_str_generator
class Model(ProcessBase):
    # Step 0: 
    def packages(self):
        return [
            'experiments.old_cust_baseline.model.ops', 
            'experiments.old_cust_baseline.process_utils.mat_op']
    # Step 1: 模塊名稱定義
    def module_name(self):
        return "model"
    # Step 2.1: 輸入參數定義
    def inputs(self):
        return []
    # Step 2.2: 輸出參數定義 
    def outputs(self):
        return [
           'cust_item_rating_matrix', 
           'user2nid', 
           'item2nid'
        ]
    # Step 4: 串接方式定義
    def connections(self, **kargs):
        conns = block_str_generator(
            'experiments/old_cust_baseline/model/connect.py'
        )
        return conns