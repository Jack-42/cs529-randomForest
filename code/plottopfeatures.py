import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

    matplotlib.rc('font', **font)

    df = pd.read_csv("../data/r99_allalphas_topfeature.csv")
    x = df['attr']
    avg = df['avg']
    min = df['min']
    max = df['max']
    up = df['max'] - df['avg']
    down = df['avg'] - df['min']
    plt.bar(x, avg, yerr=(down, up))
    plt.savefig("../data/topfeat.png", dpi=500)