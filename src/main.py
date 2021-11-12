# main.py

import pandas as pd
import matplotlib.pyplot as plt
import csv
import statistics as st
import math as mth
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import * #test_train_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import * #classification_report, confusion_matrix, accuracy_score
import pydotplus
from sklearn import tree
import matplotlib.image as pltimg
import numpy as np

from preprocess import preprocess

OG_FILE_PATH = 'data/original_dataset.csv'
#OG_FILE_PATH = 'data/original_dataset_sample.csv'

US_FILE_PATH = 'data/us_clean.csv'
EU_FILE_PATH = 'data/eu_clean.csv'

EU = [
        'Austria',
        'Belgium',
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

US = ['United States of America']


#Insights to find
#-correlation between language and salary
#-with years of experience and languages known, predict salary
#-level of degree correlation with salary
#-age correlation with languages known
# Determine age1stCode category using Naive Bayes using yearly salary, gender, operating system, and degree recieved

age_dict = {'Younger than 5 years': 0, '5 - 10 years': 1, '11 - 17 years': 2, '18 - 24 years': 3, 
    '25 - 34 years': 4, '35 - 44 years': 5, '45 - 54 years': 6, '55 - 64 years': 7, 'Older than 64 years': 8, 'NA': 9}
salary_age1stcode = [[], [], [], [], [], [], [], [], []]
gender_dict = {'Man': 0, 'Woman': 1, 'Non-binary, genderqueer, or gender non-conforming': 2}
gender_age1stcode = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

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

def gen_dict(vals):
    d = {}
    i = 0
    for val in vals:
        d[val] = i
        i += 1

    return d

def gen_all_dicts():
    df = pd.read_csv('data/us_clean.csv')

  

    read_to_cols(cols, 'data/us_clean.csv')

    mainbr_dict = gen_dict(cols[0])
    df['MainBranch'] = df['MainBranch'].map(mainbr_dict)
    #print(mainbr_dict)


    employ_dict = gen_dict(cols[1])
    df['Employment'] = df['Employment'].map(employ_dict)
    #print(employ_dict)

    country_dict = gen_dict(cols[2])
    df['Country'] = df['Country'].map(country_dict)
    #print(country_dict)

    usstate_dict = gen_dict(cols[3])
    df['US_State'] = df['US_State'].map(usstate_dict)
    #print(usstate_dict)



    edlev_dict = gen_dict(cols[6])


    age_dict_2 = gen_dict(cols[4])
    #print(age_dict_2)

    #yrscode_dict = gen_dict(cols[5])


    #yrscode_pro_dict = gen_dict(cols[6])

    orgsz_dict = gen_dict(cols[7])
    #print(orgsz_dict)

    compfreq_dict = gen_dict(cols[8])
    #print(compfreq_dict)

    opsys_dict = gen_dict(cols[9])
    #print(opsys_dict)

    so_visit_freq_dict = gen_dict(cols[10])
    #print(so_visit_freq_dict)


    so_accnt_dict = gen_dict(cols[11])
    #print(so_accnt_dict)
    so_partic_dict = gen_dict(cols[12])
    #print(so_partic_dict)


    #age_dict = gen_dict(cols[13])
    gen_dict_2 = gen_dict(cols[14])
    #print(gen_dict_2)
    trans_dict = gen_dict(cols[15])
    #print(trans_dict)
    #
    #print(edlev_dict)
    #df['EdLevel'] = df['EdLevel'].map(edlev_dict)

    compfreq_dict = gen_dict(cols[15])
    #df['CompFreq'] = df['CompFreq'].map(compfreq_dict)

    #features = ['MainBranch', 'Employment', 'CompFreq']
    #x = df[features]
    #y = df['Country']


    #x = df.iloc[:, 0:16].values
    #y = df.iloc[:, 16].values

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    #sc = StandardScaler()
    #x_train = sc.fit_transform(x_train)
    #x_test = sc.transform(x_test)

    #regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    #regressor.fit(x_train, y_train)
    #y_pred = regressor.predict(x_test)

    #rint(confusion_matrix(y_test,y_pred))
    #print(classification_report(y_test,y_pred))
    #print(accuracy_score(y_test, y_pred))



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

def plot_attr(df, attr: str):
    fig = plt.figure(figsize=(5, 5))
    df.hist(column=attr)
    plt.xlabel(attr, fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.show()

if __name__ == '__main__':
    # Preprocess datafile into cleaned US/EU files
    us_model = preprocess(OG_FILE_PATH, US_FILE_PATH, US, 'us')
    eu_model = preprocess(OG_FILE_PATH, EU_FILE_PATH, EU, 'eu')

    #plot_attr(us_model, 'ConvertedCompYearly')
    #plot_attr(eu_model, 'ConvertedCompYearly')

    names = list(us_model.drop(['ConvertedCompYearly'], axis=1))
    namesdf = us_model.drop(['ConvertedCompYearly'], axis=1)

    x = np.asarray(us_model.drop(['ConvertedCompYearly'], axis=1))
    y = np.asarray(us_model['ConvertedCompYearly'])

    x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.2, random_state=12345)

    


    #print(age_dict)

    #gen_all_dicts()

    #print(df.head())
    #print(df.describe())
    #print(calc_nb(75, 73.28, 5.4989, 30.238))

    #print(get_accuracy('data/us_testing.csv'))


    while True:
        break

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


