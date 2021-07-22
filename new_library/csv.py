import re
import pandas as pd

orig_csv_spec = {
    'sep' : ';',
    'encoding' : 'ISO-8859-1',
    'decimal' : ',',
}

csv_spec = {
    'sep' : ';',
    'encoding' : 'utf-8',
    'decimal' : ',',
}

def dateparser(date:str):
    return pd.to_datetime(date, format="%d.%m.%Y %H:%M:%S")

def load_csv(csv_fpath, csv_spec, dtypes=None, parse_dates=None, dateparser=None, index_col=None, usecols=None):

    if dateparser is not None:
        df = pd.read_csv(
            csv_fpath,
            header=0, 
            **csv_spec,
            #dtype=dtypes,
            parse_dates=['CreateTimeStamp', 'LastUpdateTimeStamp'],
            index_col=index_col,
            date_parser=dateparser,
            usecols=usecols,
            low_memory=False,
        )
    else:
        df = pd.read_csv(
            csv_fpath,
            header=0,
            parse_dates=parse_dates,
            **csv_spec,
            index_col=index_col
        )
        
    if dtypes is not None:
        for col_name, col_dtype in dtypes.items():
            df[col_name] = df[col_name].astype(col_dtype)

    # Drop last column
    df.drop(columns=df.filter(regex="Unnamed").columns.to_list(), inplace=True)

    df.sort_index(inplace=True)
        
    return df