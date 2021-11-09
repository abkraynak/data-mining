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

    fields = []
    testing_rows = []
    training_rows = []

    with open(path, 'r') as csvfile:
        dr = csv.reader(csvfile)
        for row in dr:
            if row[0] == 'ResponseId': 
                fields = row
            else:
                if row[3] in countries and row[2] == 'Employed full-time' and row[47] != 'NA' and (row[39] == 'Man' or row[39] == 'Woman' or row[39] == 'Non-binary, genderqueer, or gender non-conforming') and row[7] != 'NA':
                    if int(row[0]) % 5 == 0:
                        testing_rows.append(row)
                    else:
                        training_rows.append(row)
                
        #print(len(testing_rows))
        #print(len(training_rows))


                #if row[3] not in countries or row[2] != 'Employed full-time' or row[47] == 'NA':
                #    skip.append(int(row[0]))
                #if row[47] != 'NA' and row[7] != 'NA':
                #    values[age_dict[row[7]]].append(int(row[47]))
                #if (row[39] == 'Man' or row[39] == 'Woman' or row[39] == 'Non-binary, genderqueer, or gender non-conforming') and row[7] != 'NA':
                #    gender[age_dict[row[7]]][gender_dict[row[39]]] += 1

    with open('testing.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerows(testing_rows)

    with open('training.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerows(training_rows)

def nb_model(path: str, values: list, gender: list):
    with open(path, 'r') as csvfile:
        dr = csv.reader(csvfile)
        for row in dr:
            if row[0] != 'ResponseId': # Skip the first row
                if row[47] != 'NA' and row[7] != 'NA':
                    values[age_dict[row[7]]].append(int(row[47]))
                if (row[39] == 'Man' or row[39] == 'Woman' or row[39] == 'Non-binary, genderqueer, or gender non-conforming') and row[7] != 'NA':
                    gender[age_dict[row[7]]][gender_dict[row[39]]] += 1

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

def nominal_prob(query: str, data: list, lookup: dict) -> float:
    # Calculates probability for nominal attributes under Naive Bayes rules
    return data[lookup[query]] / sum(data)

def nominal_prob_list(query: str, data: list, lookup: dict) -> list:
    res = []
    for i in range(len(data)):
        res.append(nominal_prob(query, data[i], lookup))
    return res

def column_sum(lst):  
    return [sum(i) for i in zip(*lst)]

def totalclasslabelprob(data: list, lookup: dict, query: str):
    #print(sum(list(data[j] for j in range (len(data[0])))))
    return (sum(data[lookup[query]])) / sum(column_sum(data))

def get_final_probs(l1: list, l2: list):
    res = []

    for i in range(len(l1)):
        res.append(l1[i] * l2[i])

    return res

def get_category(lookup: dict, pos: int) -> str:
    itemsList = lookup.items()
    for item in itemsList:
        if item[1] == pos:
            return item[0]


def validate(sal: int, gen: str, target: str):
    salary_probs = nb(sal, get_nb_stats(salary_age1stcode))
    gender_probs = nominal_prob_list(gen, gender_age1stcode, gender_dict)
    res = get_final_probs(salary_probs, gender_probs)
    return get_category(age_dict, res.index(max(res))) == target

def get_accuracy(path: str):
    total = 0
    valid = 0
    with open(path, 'r') as csvfile:
        dr = csv.reader(csvfile)
        for row in dr:
            if row[0] != 'ResponseId': # Skip the first row
                print(int(row[47]), row[39], row[7])
                if validate(int(row[47]), row[39], row[7]):
                    valid += 1
                total += 1
    
    return valid // total

def NBone_att(dr, values: list):
    print(dr)
    for row in dr:
        print(row)
        values[age(row[7])].append(row[47])
    print(values)


if __name__ == '__main__':
    print('Running pre-processing . . .')
    preprocess(FILE_PATH)
    nb_model('training.csv', salary_age1stcode, gender_age1stcode)
    print('Ready')
    print()
    #print(df.head())
    #print(df.describe())
    #print(calc_nb(75, 73.28, 5.4989, 30.238))


    print(get_accuracy('testing.csv'))


    while True:
        # Get testing salary
        sal = int(input('Enter your yearly salary in USD: '))
        salary_probs = nb(sal, get_nb_stats(salary_age1stcode))
        print()
        #print(salary_probs)

        # Get testing gender
        print('Man: 0\nWoman: 1\nNon-binary, genderqueer, or gender non-conforming: 2')
        gen_idx = int(input('Select your gender: '))
        gen = get_category(gender_dict, gen_idx)
        gender_probs = nominal_prob_list(gen, gender_age1stcode, gender_dict)
        print()
        #print(gender_probs)

        res = get_final_probs(salary_probs, gender_probs)
        #print(res)

        print('We think you first started coding at an age', get_category(age_dict, res.index(max(res))))
        #print(totalclasslabelprob(gender_age1stcode, age_dict, '11 - 17 years'))


