import pandas as pd 


basefile = '../../BrightnessModule/intermediate_results/train_syn.csv'
df = pd.read_csv(basefile)
print(df.head())