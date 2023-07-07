import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path',default ='experiments/cp1_k10_beta1/output.txt',
                    help='the file path of output.txt.')
args = parser.parse_args()
path = args.input_path

inp = pd.read_csv(path)

mean_cp_num = round(inp.iloc[:,1].mean(),2)  ## 平均检测到的变点数
per_correct_cp_num = round((inp.iloc[:,1] == inp.iloc[:,4]).sum()/inp.shape[0]*100,2)  ## 变点数量检测正确的比例

n_cp_dete = inp.iloc[:,1].sum()
precision = inp.iloc[:,2].apply(lambda x: np.isin(str(x).split(' '),str(inp.iloc[0,5]).split(' ')).sum()).sum()/n_cp_dete
precision = round(precision*100,2)

n_cp_true = inp.iloc[:,4].sum()
recall = inp.iloc[:,2].apply(lambda x: np.isin(str(x).split(' '),str(inp.iloc[0,5]).split(' ')).sum()).sum()/n_cp_true
recall = round(recall*100,2)

print('input file: {}'.format(path))
print('mean cp number detected: {}'.format(mean_cp_num))
print('percent of correct cp number detected: {}'.format(per_correct_cp_num))
print('precision rate: {}'.format(precision))
print('recall rate: {}'.format(recall))


