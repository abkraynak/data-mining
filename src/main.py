# main.py

import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = 'data/survey_results_public.csv'
#DATA_PATH = 'data/small_sample.csv'

#data pre-processing
#read all information in and copy values we like into a new csv file 
#disregard students, people living in the United States

#Insights to find
#-correlation between language and salary
#-with years of experience and languages known, predict salary
#-level of degree correlation with salary
#-age correlation with languages known
#-age 1st coded correlation with salary and degree recieved

df = pd.read_csv(DATA_PATH)
print(df.head())
print(df.describe())

fig_1 = plt.figure(num = 1, figsize = (5, 5))
salary = fig_1.add_subplot()

salary.hist(df.ConvertedCompYearly)

plt.show()

