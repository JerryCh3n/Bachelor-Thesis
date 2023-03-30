import pandas as pd
from os.path import exists
# --------------------Data frame creation--------------------
file_path = "dataset_main.xlsx"
sheetname = "dataset_1"
df = pd.read_excel(io=file_path, sheet_name=sheetname)

# --------------------Dropping invalid date "X"--------------------
invalid_rows = []
for ind in df.index:
    if 'X' in str(df['Invalid (X/N)'][ind]).upper():
        invalid_rows.append(ind)
df = df.drop(index=invalid_rows)

# --------------------Dropping Specific Columns--------------------
# print(df.columns)

keep_cols = ['Q (uL/min)', 'Vg (mm/min)', 'LDR (μL/mm)', 'Print Height (mm)',
             'Average Width (μm)','Average Thickness (μm)']

drop_cols = [i for i in list(df.columns) if i not in keep_cols]

df = df.drop(columns=drop_cols)


df = df[keep_cols]
# --------------------Export Back out--------------------
export_path = "cleandata.csv"
if exists(export_path) == True:
    print("File Already Exists")
else:
    df.to_csv(export_path,index=False)
