# main.py

from preprocess import preprocess
from decision_tree import decision_tree
from random_forest import random_forest
from plot import plot_attr

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

TEST_SPLIT = 0.2
CROSS_VAL = 5

if __name__ == '__main__':
    # Preprocess datafile into cleaned US/EU files
    us_model = preprocess(OG_FILE_PATH, US_FILE_PATH, US, 'us')
    eu_model = preprocess(OG_FILE_PATH, EU_FILE_PATH, EU, 'eu')

    # Decision tree model
    us_dt = decision_tree(us_model, 'US', TEST_SPLIT, CROSS_VAL, True)
    eu_dt = decision_tree(eu_model, 'EU', TEST_SPLIT, CROSS_VAL, True)

    # Random forest model
    us_rf = random_forest(us_model, 'US', TEST_SPLIT, CROSS_VAL, True)
    eu_rf = random_forest(eu_model, 'EU', TEST_SPLIT, CROSS_VAL, True)

    # Plot original salary frequencies
    #plot_attr(us_model, 'ConvertedCompYearly')
    #plot_attr(eu_model, 'ConvertedCompYearly')