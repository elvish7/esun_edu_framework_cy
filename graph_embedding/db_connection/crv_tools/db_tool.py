import os as _os
import getpass as _getpass
import psycopg2 as _psycopg2
import base64 as _base64
import pandas as _pd
import sys as _sys

__all__=['get_conn','resize']

def _encode_str2str(in_str):
    out_str=str(_base64.standard_b64encode(in_str.encode('utf-8')))
    return out_str[2:len(out_str)-1]
def _decode_str2str(in_str):
    out_str=str(_base64.standard_b64decode(in_str.encode('utf-8')))
    return out_str[2:len(out_str)-1]

def get_conn():
    """
    Get a rawdata db connection, wrtie a encrypt key file '/home/javyan/key'

    Parameters:

    Returns:
        return a psycopg2 connection

    Raises:
        1: password wrong (do not write a key file)
        2: no os environ, looking for DE team
        3: can not write a key file ( still return a connection)

    """
    user=_os.environ['JUPYTERHUB_USER']
    try:
        file=open('/home/jovyan/key','r', encoding='UTF-8')
        password_encrypt=file.read()
        password=_decode_str2str(password_encrypt)
        print('read key file')
        file.close()
        _os.remove('/home/jovyan/key')
    except:
        password=_getpass.getpass(prompt='password: ')            
    if 'SQL-CONNECT-1_ENV1' in _os.environ:
        con_str=_os.environ['SQL-CONNECT-1_ENV1']
    elif 'SQL-CONNECT-2_ENV2' in _os.environ:
        con_str=_os.environ['SQL-CONNECT-2_ENV2']
    elif 'SQL-CONNECT-EDU_ENV1' in _os.environ:
        con_str=_os.environ['SQL-CONNECT-EDU_ENV1']
    else:
        raise KeyError('no environ "SQL-CONNECT", looking for DE team')
    con_str=con_str.replace('USER',user)\
                   .replace('PASSWORD',password)\
                   .replace('10.240.205.111','10.240.205.110')\
                   .replace('aicloud', 'industry')
    print('login as ' + user)
    conn = _psycopg2.connect(con_str)
    try:
        file=open('/home/jovyan/key','w', encoding='UTF-8')
        password_encrypt=_encode_str2str(password)
        file.write(password_encrypt)
        file.close()
    except:
        print('can not write key file')
    return conn

def sql_to_df(sql):
    """
    excute sql code and return result in a dataframe
    
    Parameters:
        param1: sql code
    
    Returns:
        return a dataframe

    Raises:

    """
    conn = get_conn()
    df = _pd.read_sql(sql,conn)
    conn.close()
    print('DataFrame shape: ' + str(df.shape))
    print('DataFrame memoryusage: ' + str(round(_sys.getsizeof(df)/1024/1024, 2)) + ' MB' )
    return df

def optmize_datatype(df):
    """
    Optmizing subtype of columns, to reduce memory usage. 

    Parameters:
        param1: input a dataframe

    Returns:
        return a dataframe with optmized subtype

    Raises:
        1:  
    """
    # 優化 float ，自動選擇最佳格式
    df_float=df.select_dtypes(include=['float'])
    df[df_float.columns] = df_float.apply(_pd.to_numeric, downcast='float')
    # --------------------------
    # 優化 int ，自動選擇最佳格式
    df_int=df.select_dtypes(include=['int'])
    df[df_int.columns] = df_int.apply(_pd.to_numeric, downcast='signed')
    df[df_int.columns] = df_int.apply(_pd.to_numeric, downcast='unsigned')
    # --------------------------
    # 優化 object ，若該欄位唯一值低於總筆數的一半，則改為 category
    df_obj=df.select_dtypes(include=['object'])
    for col in df_obj.columns:
        unique = len(df_obj[col].unique())
        total = len(df_obj[col])
        if unique < total*0.5:
            df.loc[:,col] = df_obj[col].astype('category')

def list_schemas():
    """
    List all schemas you can access, excpt some system schemas.

    Parameters:

    Returns:
        no return , just print the result

    Raises:
        1: 
    """
    conn=get_conn()
    sql = """select distinct table_schema 
             from information_schema.tables 
             where table_schema not in ('pg_catalog','sys','information_schema');""" 
    df = _pd.read_sql(sql,conn)
    conn.close()
    print(df)

def list_tables(schema,fuzzy_string=''):
    """
    List all tables in a specific schema, you can use fuzzy comparson to search tables.

    Parameters:
        param1: table schema
        param2: (optional) substring of table name
    Returns:
        no return, just print the result

    Raises:
        1:
    """
    conn=get_conn()
    sql = f"""
    SELECT table_name
      FROM information_schema.tables
     WHERE table_schema='{schema}' 
           AND table_name like '%{fuzzy_string}%';
    """
    df = _pd.read_sql(sql, conn)
    conn.close()
    print(df)

def list_columns(schema, table):
    """
    List all columns in a specific table.

    Parameters:
        param1: table schema
        param2: table name
    Returns:
        no return, just print the result

    Raises:
        1
    """
    conn=get_conn()
    sql = f"""
    SELECT column_name,data_type
      FROM information_schema.columns
     WHERE table_schema = '{schema}'
           AND table_name = '{table}';
    """
    df = _pd.read_sql(sql, conn)
    conn.close()
    print(df)


def list_etldt(schema, table):
    """
    List all etldt of table.

    Parameters:
        param1: table schema: 'rawdata', 'mlass_rawdata'
        param2: table name
    Returns:
        no return, just print the result

    Raises:
        1
    """
    conn=get_conn()
    sql = f"""
    SELECT * 
      FROM info.rawdata_etldt
     WHERE schema_name={schema} and table_name={table}"""
    df = _pd.read_sql(sql, conn)
    conn.close()
    print(df)
