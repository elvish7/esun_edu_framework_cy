# Data Descrption

## Transation
    - cust_no: [ID]
    - wm_prod_code: [ID]
    - txn_amt: [int]
    + item_rating: 
        (1) First
        (2) Average 

#=====[unused features]=====#
    - deduct_cnt: [int]

    - txn_dt: [date]
    - etl_dt: [date]

#=====[discarded features]=====#
    - dta_src: <empty>/A0/B1

## User
    - cust_no: [ID]
    - age: [int] 
        (1*) binary (median)
        (2) 10-binning
    - cust_vintage: [int]
        (1*) binary (median)
        (2) quarter-quantile-binning
    - gender_code: F/M
    - income_range_code: 1/2/3/4
    - risk_type_code: 02/03/04

#=====[unused features]=====#
    - cc_cust_level: A/B/C/D/E
    - children_cnt: 0/1/2/3
    - cust_status_code: 'b'/'55'
    - edu_code: 0/1/2/3/4/5/6
    - house_own_type_code: 0/1/2/3/b
    - marital_status_code: 1/2
    - personal_income: [int]
    - position_code: 2/4/5
    - wm_club_class_code: 0/P/Q/Z/O/N/1/2/3/M/H
    - work_years: [int]

    - data_ym: [date]
    - cust_start_dt: [date]
    - etl_dt: [date]

#=====[discarded features]=====#
    - biz_line_code: P (no variance)
    - country_code: TW (no variance)
    - debt_nego_code: b (no variance)
    - nation_code: 1 (no variance)


## Item
    - wm_prod_code: [ID]
    - can_rcmd_ind: 0/1
    - prod_ccy: USD/TWD/EUR/CNY/ZAR/JPY/NZD/GBP/SGD/CHF/CAD/HKD/SEK
        (1*) categorization (USD/TWD/EUR/OTHERS)
    - prod_risk_code: RR1/RR2/RR3/RR4/RR5
    - prod_detail_type_code: FNDF/FNDD

#=====[unused features]=====#
    - mkt_rbot_ctg_ic: F0801/F0201/F1501/F0102/F0101/.....53 in total(<100)

#=====[discarded features]=====#
