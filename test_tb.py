import pandas as pd
path = "/home/ljh/data1/patent/原始数据dta省份汇总2003年之后的专利/上海市/上海市_2009.dta"
try:
    df = pd.read_stata(path)
except Exception as e:
    import traceback
    traceback.print_exc()
