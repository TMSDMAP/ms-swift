import pandas as pd
import pyreadstat

path = "/home/ljh/data1/patent/原始数据dta省份汇总2003年之后的专利/上海市/上海市_2009.dta"

for enc in ["gb18030", "gbk", "latin1", "utf-8", "mac_roman", "cp1252"]:
    print(f"\n--- Testing pandas with encoding: {enc} ---")
    try:
        df = pd.read_stata(path, encoding=enc, convert_categoricals=False, convert_dates=False)
        print("SUCCESS pandas", enc, len(df))
    except Exception as e:
        print("FAIL pandas:", type(e).__name__, str(e)[:200])

    print(f"\n--- Testing pyreadstat with encoding: {enc} ---")
    try:
        df, meta = pyreadstat.read_dta(path, encoding=enc)
        print("SUCCESS pyreadstat", enc, len(df))
    except Exception as e:
        print("FAIL pyreadstat:", type(e).__name__, str(e)[:200])
