from dev.etl3.common.ETLBase import ProcessBase 
from dev.etl3.common.process_compiler import block_str_generator
class PreProcess(ProcessBase):
    # Step 0: 
    def packages(self):
        return 'dev.etl3.preprocess.ops'
    # Step 1: 模塊名稱定義
    def module_name(self):
        return "preprocess"
    # Step 2.1: 輸入參數定義
    def inputs(self):
        return [
            'dig_start_dt', # 基金原始資料表 
            'dig_end_dt' # 基金淨值表 
        ]
    # Step 2.2: 輸出參數定義 
    def outputs(self):
        return [
            'item_feature', 
            'cust_txn_rating'
        ] 
    # Step 4: 串接方式定義
    def connections(self, **kargs):
        conns = block_str_generator('dev/etl3/preprocess/connect.py')
        return conns