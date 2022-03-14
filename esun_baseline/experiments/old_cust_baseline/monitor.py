from dev.etl3.final_filtering.config_edu import FinalFilter 
from dev.etl3.dataloader.utils import get_conn
def final_filtering(
    cust_item_rating_matrix,
    cust_item_rank_matrix,
    item2nid, 
    user2nid
):
    rawdata_conn = get_conn('edu') 
    final_filter = FinalFilter() 
    final_filter.config(
        cust_item_rating_matrix = cust_item_rating_matrix,
        cust_item_rank_matrix = cust_item_rank_matrix,
        item2nid = item2nid,
        user2nid = user2nid,
        rawdata_conn = rawdata_conn
    )
    output = final_filter.pipe.output.get(verbose=True, load_tmp=True)
    return output