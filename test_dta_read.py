import pandas as pd
import pyreadstat

path = "/home/ljh/data1/patent/原始数据dta省份汇总2003年之后的专利/上海市/上海市_2009.dta"

print("1. pandas direct")
try:
    df = pd.read_stata(path, convert_categoricals=False, convert_dates=False)
    print("SUCCESS pandas direct", len(df))
except Exception as e:
    print("FAIL pandas direct:", e)

print("2. pandas read_stata, convert_strings=False")
try:
    with pd.io.stata.StataReader(path, convert_dates=False, convert_categoricals=False) as reader:
        # We can't actually set convert_strings in read_stata in all versions, let's see
        pass
    df = pd.read_stata(path, convert_dates=False, convert_categoricals=False)
    print("SUCCESS pandas string=False", len(df))
except Exception as e:
    print("FAIL pandas string=False:", e)

print("3. pyreadstat disable datetime")
try:
    df, meta = pyreadstat.read_dta(path, disable_datetime_conversion=True)
    print("SUCCESS pyreadstat", len(df))
except Exception as e:
    print("FAIL pyreadstat:", e)

