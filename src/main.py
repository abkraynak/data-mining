# main.py

import pandas as pd
import matplotlib.pyplot as plt
import csv
import statistics as st
import math as mth
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import test_train_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pydotplus
from sklearn import tree
import matplotlib.image as pltimg
import numpy as np

FILE_PATH = 'data/original_dataset.csv'
#FILE_PATH = 'data/original_dataset_sample.csv'

eu = [
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
    ]

us = ['United States of America']



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

del_cols = [46, 45, 44, 43, 42, 41, 37, 36, 32, 31, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 14, 13, 11, 8, 6, 5, 0]

def rem_cols(row: list, cols: list):
    for col in cols:
        row.pop(col)   
    return row

def preprocess(path: str, countries: list, sel: str):
    # Countries we want data from
    fields = []
    testing_rows = []
    training_rows = []

    with open(path, 'r') as csvfile:
        dr = csv.reader(csvfile)
        i = 0
        for row in dr:
            row = rem_cols(row, del_cols)
            if row[0] == 'MainBranch': 
                fields = row
            else:
                if row[2] in countries and row[1] == 'Employed full-time' and row[16] != 'NA' and (row[14] == 'Man' or row[14] == 'Woman' or row[14] == 'Non-binary, genderqueer, or gender non-conforming') and row[5] != 'NA':
                    if i % 5 == 0:
                        testing_rows.append(row)
                    else:
                        training_rows.append(row)
            i += 1


    with open('data/' + sel + '_testing.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerows(testing_rows)

    with open('data/' + sel + '_training.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerows(training_rows)

cols = [
            [], [], [], [], [],
            [], [], [], [], [],
            [], [], [], [], [],
            [], []
]

def read_to_cols(cols: list, path: str):
    with open(path, 'r') as csvfile:
        dr = csv.reader(csvfile)
        for row in dr:
            if row[0] != 'MainBranch': 
                for i in range(17):
                    if row[i] not in cols[i]:
                        cols[i].append(row[i])
    #print(cols)

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
    print(salary_age1stcode)
    salary_probs = nb(sal, get_nb_stats(salary_age1stcode))
    gender_probs = nominal_prob_list(gen, gender_age1stcode, gender_dict)
    res = get_final_probs(salary_probs, gender_probs)
    return get_category(age_dict, res.index(max(res))) == target

def get_accuracy(path: str):
    total = 0
    valid = 0
    with open(path, 'r') as csvfile:
        dr = csv.reader(csvfile)
        print('open')
        for row in dr:
            if row[0] != 'ResponseId': # Skip the first row
                if total > 50: 
                    break
                if validate(int(row[47]), row[39], row[7]):
                    valid += 1
                total += 1
        print(valid)
    return valid // total

def NBone_att(dr, values: list):
    print(dr)
    for row in dr:
        print(row)
        values[age(row[7])].append(row[47])
    print(values)


def gen_dict(vals):
    d = {}
    i = 0
    for val in vals:
        d[val] = i
        i += 1

    return d


def gen_all_dicts():
    df = pd.read_csv('data/us_training.csv')
    read_to_cols(cols, 'data/us_training.csv')

    mainbr_dict = gen_dict(cols[0])
    df['MainBranch'] = df['MainBranch'].map(mainbr_dict)
    print(mainbr_dict)


    employ_dict = gen_dict(cols[1])
    df['Employment'] = df['Employment'].map(employ_dict)
    print(employ_dict)

    country_dict = gen_dict(cols[2])
    df['Country'] = df['Country'].map(country_dict)
    print(country_dict)

    usstate_dict = gen_dict(cols[3])
    df['US_State'] = df['US_State'].map(usstate_dict)
    print(usstate_dict)



    edlev_dict = gen_dict(cols[6])


    age_dict_2 = gen_dict(cols[4])
    print(age_dict_2)

    #yrscode_dict = gen_dict(cols[5])


    #yrscode_pro_dict = gen_dict(cols[6])

    orgsz_dict = gen_dict(cols[7])
    print(orgsz_dict)

    compfreq_dict = gen_dict(cols[8])
    print(compfreq_dict)

    opsys_dict = gen_dict(cols[9])
    print(opsys_dict)

    so_visit_freq_dict = gen_dict(cols[10])
    print(so_visit_freq_dict)


    so_accnt_dict = gen_dict(cols[11])
    print(so_accnt_dict)
    so_partic_dict = gen_dict(cols[12])
    print(so_partic_dict)


    #age_dict = gen_dict(cols[13])
    gen_dict_2 = gen_dict(cols[14])
    print(gen_dict_2)
    trans_dict = gen_dict(cols[15])
    print(trans_dict)
    #
    #print(edlev_dict)
    #df['EdLevel'] = df['EdLevel'].map(edlev_dict)

    compfreq_dict = gen_dict(cols[15])
    #df['CompFreq'] = df['CompFreq'].map(compfreq_dict)

    #features = ['MainBranch', 'Employment', 'CompFreq']
    #x = df[features]
    #y = df['Country']


    x = df.iloc[:, 0:16].values
    y = df.iloc[:, 16].values

    x_



    #dtree = DecisionTreeClassifier()
    #dtree = dtree.fit(x, y)

    #data = tree.export_graphviz(dtree, out_file=None, feature_names = features)
    #graph = pydotplus.graph_from_dot_data(data)
    #print(graph.to_string())
    
    #graph.write_png('mytree.png')
    #img = pltimg.imread('mytree.png')

    #imgplot = plt.imshow(img)
    #plt.show()




    #print(age_dict_2)
    #print(age_dict)


if __name__ == '__main__':
    print('Running pre-processing . . .')
    preprocess(FILE_PATH, us, 'us')
    preprocess(FILE_PATH, eu, 'eu')

    #nb_model('training.csv', salary_age1stcode, gender_age1stcode)
    print('Ready')
    print()

    #print(age_dict)

    gen_all_dicts()

    #print(df.head())
    #print(df.describe())
    #print(calc_nb(75, 73.28, 5.4989, 30.238))

    #print(get_accuracy('data/us_testing.csv'))


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


