# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = 'data/survey_results_public.csv'
#DATA_PATH = 'data/small_sample.csv'

df = pd.read_csv(DATA_PATH)
#print(df.head())
#print(df.describe())

fig_1 = plt.figure(num = 1, figsize = (5, 5))
salary = fig_1.add_subplot()

salary.hist(df.ConvertedCompYearly)

plt.show()