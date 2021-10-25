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

def preprocess(path: str):
    # Rows from CSV file to skip
    skip = []

    # Countries we want data from
    countries = [
        'Austria',
        'Belgium',
        'Canada',
        'Denmark',
        'Finland',
        'France',
        'Germany',
        'Iceland',
        'Ireland',
        'Italy',
        'Luxembourg'
        'Netherlands',
        'Norway',
        'Poland',
        'Portugal',
        'Spain',
        'Sweden',
        'Switzerland',
        'United Kingdom of Great Britain and Northern Ireland',
        'United States of America', 
    ]

    with open(path, 'r') as csvfile:
        dr = csv.reader(csvfile)
        for row in dr:
            if row[3] not in countries or row[2] != 'Employed full-time' or row[47] == 'NA':
                if row[0] != 'ResponseId': # Skip the first row
                    skip.append(int(row[0]))

        # print(len(skip))

        df = pd.read_csv(path, skiprows = skip)
        return df

if __name__ == '__main__':
    df = preprocess(FILE_PATH)
    print(df.head())
    print(df.describe())
