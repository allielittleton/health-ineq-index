import statistics
import numpy as np
import math
from numpy import array, exp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb
import numpy as np
import statsmodels.api as sm


def run_simple_regression(data, curr_var, X, y):

    # regressor = LinearRegression()
    # regressor.fit(X, y)

    # predictions = regressor.predict(X)
    X = data[curr_var]
    y = data['Standard Deviation Index Value']
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2, missing='drop')
    est2 = est.fit()
    return est2.pvalues[1], est2.rsquared


def regression(data):
    vars = data.columns
    n_cols = len(vars)

    y = data[vars[1]].values.reshape(-1, 1)

    p_vals = []
    rsquared_vals = []
    for i in range(1, n_cols):
        curr_var = vars[i]
        X = data[curr_var].values.reshape(-1, 1)
        p, r_sq = run_simple_regression(data, curr_var, X, y)
        p_vals.append(p)
        rsquared_vals.append(r_sq)

    d = {'Variable': vars[1:], 'p': p_vals, 'R-squared': rsquared_vals}

    df = pd.DataFrame(d)
    df.to_excel("Univariate Regression Table.xlsx")

def correlogram(data):
    # creating mask
    mask = np.triu(np.ones_like(data.corr()))

    # plotting a triangle correlation heatmap
    dataplot = sb.heatmap(data.corr(), annot=True, mask=mask)

    # displaying heatmap
    mp.show()

def random_forest(df):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_data = imp.fit_transform(df)
    features = pd.DataFrame(data=imputed_data, columns=df.columns)

    print(df)
    print(features)

    labels = np.array(features['Standard Deviation Index Value'])
    features = features.drop('Standard Deviation Index Value', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)
    train_test_split(features, labels, test_size=0.25, random_state=42)

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', np.mean(errors))

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')



def main():
    df = pd.read_excel("Inequality Index Values 2019.xlsx")
    himr_data = df[df["Modal Age"] == 0]
    limr_data = df[df["Modal Age"] != 0]
    correlogram(himr_data)
    #correlogram(limr_data)
    regression(himr_data)
    #regression(limr_data)

    random_forest(limr_data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
