import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(file): 

    data = pd.read_csv(file)

    feat_raw = data['quizzes']
    label_raw = data['solutions']

    X = np.empty((len(feat_raw),9,9,1))
    y = np.empty((len(label_raw),   81,1))

    for i, f in enumerate(feat_raw):
        X[i,] = (np.array(list(map(int,list(f)))).reshape((9,9,1))/9)-0.5 # normalizar los sudokus para llenar
    

    for i,f in enumerate(label_raw):
        y[i,] = np.array(list(map(int,list(f)))).reshape((81,1))-1 # dejo los numeros del 0 al 8 en los sudokus llenos

    del(feat_raw)
    del(label_raw)    

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    return x_train, x_test, y_train, y_test