# main.py

import pandas as pd
import matplotlib.pyplot as plt
import csv

FILE_PATH = 'data/survey_results_public.csv'
#FILE_PATH = 'data/small_sample.csv'

#data pre-processing
#read all information in and copy values we like into a new csv file 
#disregard students, people living in the United States

#Insights to find
#-correlation between language and salary
#-with years of experience and languages known, predict salary
#-level of degree correlation with salary
#-age correlation with languages known
#-age 1st coded correlation with salary and degree recieved

def preprocess():
    # Rows from CSV file to skip
    skip = []

    # Countries we want data from
    countries = [
        'United States of America', 
        'United Kingdom of Great Britain and Northern Ireland',
        'France',
        'Germany',
        'Switzerland',
        'Canada',
        'Portugal',
        'Spain',
        'Netherlands',
        'Italy',
        'Poland',
        'Sweden',
        'Belgium'
        'Finland',
        'Denmark',
        'Austria',
        'Bulgaria',
        'Croatia',
        'Estonia',
        'Czech Republic',
        'Hungary',
        'Ireland',
        'Luxembourg'
    ]

    with open(FILE_PATH, 'r') as csvfile:
        dr = csv.reader(csvfile)
        print('here')
        for row in dr:
            if row[3] not in countries or row[2] != 'Employed full-time' or row[47] == 'NA':
                if row[0] != 'ResponseId':
                    skip.append(int(row[0]))

        # Print 
        print(len(skip))

        df = pd.read_csv(FILE_PATH, skiprows=skip)
        print(df.head())
        #print(df.describe())

        fig_1 = plt.figure(num = 1, figsize = (5, 5))
        salary = fig_1.add_subplot()

        #salary.hist(df.ConvertedCompYearly)

        #plt.show()

if __name__ == '__main__':
    preprocess()
