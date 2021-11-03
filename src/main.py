# main.py

import pandas as pd
import matplotlib.pyplot as plt
import csv
import statistics as st
import math as mth

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
# Determine age1stCode category using Naive Bayes using yearly salary, gender, operating system, and degree recieved

age_dict = {'Younger than 5 years': 0, '5 - 10 years': 1, '11 - 17 years': 2, '18 - 24 years': 3, 
    '25 - 34 years': 4, '35 - 44 years': 5, '45 - 54 years': 6, '55 - 64 years': 7, 'Older than 64 years': 8}
salary_age1stcode = [[], [], [], [], [], [], [], [], []]
gender_dict = {'Man': 0, 'Woman': 1, 'Non-binary, genderqueer, or gender non-conforming': 2}
gender_age1stcode = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]



def preprocess(path: str, values: list, gender: list):
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

    with open(path, 'r') as csvfile:
        dr = csv.reader(csvfile)
        for row in dr:
            if (row[39] == 'Man' or row[39] == 'Woman' or row[39] == 'Non-binary, genderqueer, or gender non-conforming') and row[7] != 'NA':
                if row[0] != 'ResponseId': # Skip the first row
                    gender[age_dict[row[7]]][gender_dict[row[39]]] += 1
                    
        print(gender)
        #NBone_att(dr, agefirstcoded)

        df = pd.read_csv(path, skiprows = skip)
        return df

def stats_calc(l: list) -> list:
    # Returns list containing mean, stdev, and variance from list
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

def calc_nb(x: float, mean: float, stdev: float, var: float) -> float:
    # Naive Bayes probabilistic density formula
    return (1 / (mth.sqrt(2 * mth.pi) * stdev) ) * mth.exp(-((x - mean) ** 2) / (2 * var))

def nb(x: float, stats) -> list:
    res = []
    for l in stats:
        res.append(calc_nb(x, l[0], l[1], l[2]))
    return res

def NBone_att(dr, values: list):
    print(dr)
    for row in dr:
        print(row)
        values[age(row[7])].append(row[47])
    print(values)
    

#def NBagefirstcoded()


if __name__ == '__main__':
    df = preprocess(FILE_PATH, salary_age1stcode, gender_age1stcode)
    #print(df.head())
    #print(df.describe())
    #print(calc_nb(75, 73.28, 5.4989, 30.238))
    #print(nb(500000, get_nb_stats(agefirstcoded)))

