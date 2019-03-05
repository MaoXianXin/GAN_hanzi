# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd

path = '~/Downloads/dataset/ChnSentiCorp_htl_all.csv'

dataframe = pd.read_csv(path)

type(dataframe['review'])

new_path = './data/ChnSentiCorp_htl_all.txt'
new_file = open(new_path, 'w', encoding='utf8')

for line in dataframe['review']:
    new_file.write(str(line))

new_file.close()


