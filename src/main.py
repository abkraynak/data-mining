# main.py

import pandas as pd
import matplotlib.pyplot as plt
import csv
import statistics as st

FILE_PATH = 'data/survey_results_public.csv'
#FILE_PATH = 'data/small_sample.csv'

#data pre-processing
#read all information in and copy values we like into a new csv file 
#disregard students, people living in the United States
#DONE


#Insights to find
#-correlation between language and salary
#-with years of experience and languages known, predict salary
#-level of degree correlation with salary
#-age correlation with languages known
#-age 1st coded correlation with salary and degree recieved

age_dict = {'Younger than 5 years': 0, '5 - 10 years': 1, '11 - 17 years': 2, '18 - 24 years': 3, 
    '25 - 34 years': 4, '35 - 44 years': 5, '45 - 54 years': 6, '55 - 64 years': 7, 'Older than 64 years': 8}
agefirstcoded = [[], [], [], [], [], [], [], [], []]

def preprocess(path: str, values: list):
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
                #print(row[3])
                if row[0] != 'ResponseId': # Skip the first row
                    skip.append(int(row[0]))

        # print(len(skip))
    with open(path, 'r') as csvfile:
        dr = csv.reader(csvfile)
        for row in dr:
            if row[47] != 'NA' and row[7] != 'NA':
                if row[0] != 'ResponseId': # Skip the first row
                    values[age_dict[row[7]]].append(int(row[47]))
        #print(values)
        #NBone_att(dr, agefirstcoded)

        df = pd.read_csv(path, skiprows = skip)
        return df

def stats_calc(l: list) -> list:
    res = []
    res.append(st.mean(l))
    res.append(st.stdev(l))
    res.append(st.variance(l))
    return res

def get_nb_stats(l) -> list:
    res = []
    for element in l:
        res.append(stats_calc(element))

    return res


def NBone_att(dr, values: list):
    print(dr)
    for row in dr:
        print(row)
        values[age(row[7])].append(row[47])
    print(values)
    

#def NBagefirstcoded()


if __name__ == '__main__':
    df = preprocess(FILE_PATH, agefirstcoded)
    #print(df.head())
    #print(df.describe())

    print(get_nb_stats(agefirstcoded))

