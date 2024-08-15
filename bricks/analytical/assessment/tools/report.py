# ---------------------------------------------------------------------------- #
#                             Process and report df                            #
# ---------------------------------------------------------------------------- #
ijsselsteinseweg.dataframes['sri']

df = ijsselsteinseweg.dataframes['gf_params']
# Drop columns
df = df.drop(['x_gauss', 'length','height','limit_line'], axis=1)
df['ax'] = df['ax'].apply(lambda x: [min(x), max(x)])
df

df2 = ijsselsteinseweg.dataframes['LTSM-GF']
df2.drop(['e_bh', 'e_bs','e_sh','e_ss','e_h','lh_s','lh_h','dl_s','dl_h'], axis=1)

ijsselsteinseweg.dataframes['LTSM-MS']

rr = ijsselsteinseweg.dataframes['LTSM-MS']
rr.drop(['e_bh', 'e_bs','e_sh','e_ss','e_h','lh_s','lh_h','dl_s','dl_h'], axis=1)

dicts = [ijsselsteinseweg.soil['sri'],
         ijsselsteinseweg.process['params'],
         ijsselsteinseweg.assessment['ltsm']['greenfield']['results'],
         ijsselsteinseweg.assessment['ltsm']['measurements']['results'],
         ijsselsteinseweg.soil['shape']]
names = ['sri','gf_params','LTSM-GF','LTSM-MS', 'MS-VAR']
ijsselsteinseweg.process_dfs(dicts, names)

drop_1 = ['lh_s','dl_s','lh_h','dl_h']
drop_2 = ['e_bh', 'e_bs','e_h','e_sh','e_ss','lh_s','lh_h','dl_s','dl_h']

df1 = ijsselsteinseweg.dataframes['LTSM-MS']
df1 = df1.drop(drop_1, axis = 1)

df2 = ijsselsteinseweg.dataframes['LTSM-MS']
df2 = df2.drop(drop_2, axis=1)

import pandas as pd

with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')

# No need to explicitly save, as `with` context will handle that.