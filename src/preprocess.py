# preprocess.py

import csv as csv
import pandas as pd

# List of column indices in original dataset to remove
del_cols = [46, 45, 44, 43, 42, 41, 37, 36, 32, 31, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 14, 13, 11, 8, 6, 5, 0]

# List of columns that contain nominal (categorical) data
nom_cols = ['MainBranch', 'Employment', 'Country', 'US_State', 'Age1stCode', 'OrgSize', 'CompFreq', 'OpSys', 'SOVisitFreq', 'SOAccount', 'SOPartFreq', 'Age', 'Gender', 'Trans']

# Given a list row, remove the elements at the positions in cols
def rem_cols(row: list, cols: list):
    for col in cols:
        row.pop(col)   
    return row

# Debug: check for missing values
def print_missing_value_counts(df):
    print('Number of missing values per column:')
    for i in df:
        print(i, ': ', end='')
        if df[i].count() < df.shape[0]:
            print(df.shape[0] - df[i].count(), end='')
        print()

# Debug: check number of records and attributes
def print_num_recs_attr(df):
    print('Number of records:', df.shape[0])
    print('Number of attributes:', df.shape[1])

# Debug: check missing values and record/attribute count
def debug_data_summary(df):
    print_num_recs_attr(df)
    print_missing_value_counts(df)

# Given a CSV file, generate a new CSV file that contains only the rows/columns we are looking for
def clean_csv(path: str, countries: list, sel: str):
    fields = []
    rows = []

    with open(path, 'r') as csvfile:
        dr = csv.reader(csvfile)
        for row in dr:
            row = rem_cols(row, del_cols) # Remove certain columns
            if row[0] == 'MainBranch': # First row of data is the column names
                fields = row
            else:
                if row[2] in countries and row[1] == 'Employed full-time' and row[16] != 'NA' and (row[14] == 'Man' or row[14] == 'Woman' or row[14] == 'Non-binary, genderqueer, or gender non-conforming') and row[5] != 'NA':
                    rows.append(row)

    # Write cleaned data to a new CSV file
    with open('data/' + sel + '_clean.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerows(rows)

# Read CSV file with pandas, correct missing values, return updated dataframe
def clean_df(path: str):
    df = pd.read_csv(path)
    #debug_data_summary(df) # Data summary

    # Replace missing values with NA or 0
    df['US_State'].fillna('NA', inplace=True)
    df['Age1stCode'].fillna('NA', inplace=True)
    df['YearsCode'].replace('Less than 1 year', 0, inplace=True)
    df['YearsCode'].replace('More than 50 years', 50, inplace=True)
    df['YearsCodePro'].fillna(0, inplace=True)
    df['YearsCodePro'].replace('Less than 1 year', 0, inplace=True)
    df['YearsCodePro'].replace('More than 50 years', 50, inplace=True)
    df['OrgSize'].fillna('NA', inplace=True)
    df['OpSys'].fillna('NA', inplace=True)
    df['SOVisitFreq'].fillna('NA', inplace=True)
    df['SOAccount'].fillna('NA', inplace=True)
    df['SOPartFreq'].fillna('NA', inplace=True)
    df['Age'].fillna('NA', inplace=True)
    df['Trans'].fillna('NA', inplace=True)

    #debug_data_summary(df) # Data summary
    return df

def encode(df, nom_cols: list):
    model = pd.get_dummies(df, columns=nom_cols, drop_first=False)
    
    #print_num_recs_attr(model) # Data summary
    return model

def preprocess(og_path: str, co_path: str, countries: list, sel: str):
    print('Running pre-processing . . . ')
    clean_csv(og_path, countries, sel)
    df = clean_df(co_path)
    df_model = encode(df, nom_cols)
    print('Complete')
    return df_model
