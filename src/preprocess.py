# preprocess.py

import csv

# List of column indices in original dataset to remove
del_cols = [46, 45, 44, 43, 42, 41, 37, 36, 32, 31, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 14, 13, 11, 8, 6, 5, 0]

# Given a list row, remove the elements at the positions in cols
def rem_cols(row: list, cols: list):
    for col in cols:
        row.pop(col)   
    return row

# Given a CSV file, generate a new CSV file that contains only the rows/columns we are looking for
def preprocess(path: str, countries: list, sel: str):
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