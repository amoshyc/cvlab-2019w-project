import json
import pandas as pd

df_true = pd.read_csv('./test_true.csv')
df_pred = pd.read_csv('./test_pred.csv')

assert len(df_pred) == len(df_true), 'Size is not 1000'
for i in range(1000):
    assert df_pred.loc[i, 'name'] == f'{i:03d}.jpg', f'Row {i} is not {i:03d}.jpg'

df_true = df_true.drop('name', axis=1)
df_pred = df_pred.drop('name', axis=1)

df_mse = ((df_true - df_pred)**2).mean(axis=0)
br_mse = (df_mse['BR_x'] + df_mse['BR_y']) / 2.0
bl_mse = (df_mse['BL_x'] + df_mse['BL_y']) / 2.0
tl_mse = (df_mse['TL_x'] + df_mse['TL_y']) / 2.0
tr_mse = (df_mse['TR_x'] + df_mse['TR_y']) / 2.0

print('MSE of each corner:')
print('BR:', br_mse)
print('BL:', bl_mse)
print('TL:', tl_mse)
print('TR:', tr_mse)
print('-' * 20)

print('mMSE:', (br_mse + bl_mse + tl_mse + tr_mse) / 4.0)

