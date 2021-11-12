# plot.py 

import matplotlib.pyplot as plt

def plot_attr(df, attr: str):
    fig = plt.figure(figsize=(5, 5))
    df.hist(column=attr)
    plt.xlabel(attr, fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.show()