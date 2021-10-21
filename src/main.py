# main.py

import pandas as pd
import numpy as np

DATA_PATH = 'data/survey_results_public.csv'

df = pd.read_csv(DATA_PATH)
df.describe()
