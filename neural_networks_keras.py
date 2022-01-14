"""==========================================="""
""" !!!!!!!!!NEED TO INSTALL TENSOR FLOW!!!!!"""
"""==========================================="""


if 1 == 1:
    #from keras.callbacks import ModelCheckpoint
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten
if 1 == 1:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
import bokeh.plotting as bp
import plotly.graph_objects as go

if 1 == 0:
    # import seaborn as sb
    from xgboost import XGBRegressor

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

"""data"""
"""https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data"""

"""source"""
"""https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33"""


def get_data():
    # get train data
    train_data_path = 'train.csv'
    train = pd.read_csv(train_data_path)

    # get test data
    test_data_path = 'test.csv'
    test = pd.read_csv(test_data_path)

    return train, test


def get_combined_data():
    # reading train data
    train, test = get_data()

    target = train.SalePrice
    train.drop(['SalePrice'], axis=1, inplace=True)

    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'Id'], inplace=True, axis=1)
    return combined, target


# Load train and test data into pandas DataFrames
train_data, test_data = get_data()

# Combine train and test data to process them together
combined, target = get_combined_data()

print(combined.describe())


def get_cols_with_no_nans(df, col_type):
    """
    Arguments :
    df : The dataframe to process
    col_type :
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans
    """
    if col_type == 'num':
        predictors = df.select_dtypes(exclude=['object'])
    elif col_type == 'no_num':
        predictors = df.select_dtypes(include=['object'])
    elif col_type == 'all':
        predictors = df
    else:
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans


"""columns that do not have any missing values"""
num_cols = get_cols_with_no_nans(combined, 'num')
cat_cols = get_cols_with_no_nans(combined, 'no_num')

print('Number of numerical columns with no nan values :', len(num_cols))
print('Number of nun-numerical columns with no nan values :', len(cat_cols))

combined = combined[num_cols + cat_cols]

"""plot histograms"""
if 1 == 1:
    def make_hist_plot(title, hist, edges):
        """define functions to plot histogram"""
        """open figure"""
        p = bp.figure(title=title, tools='', background_fill_color="#fafafa")
        """plot histogram"""
        p.quad(top=hist,
               bottom=0,
               left=edges[:-1],
               right=edges[1:],
               fill_color="navy",
               line_color="white",
               alpha=0.5)
        """cuntomise plot"""
        p.y_range.start = 0
        p.xaxis.axis_label = 'x'
        p.yaxis.axis_label = 'Pr(x)'
        p.grid.grid_line_color = "white"

        return p


    hist1, edges1 = np.histogram(combined.iloc[:, 0], density=True, bins=10)
    hist2, edges2 = np.histogram(combined.iloc[:, 1], density=True, bins=10)
    hist3, edges3 = np.histogram(combined.iloc[:, 2], density=True, bins=10)
    hist4, edges4 = np.histogram(combined.iloc[:, 3], density=True, bins=10)

    h1 = make_hist_plot(combined.columns[0], hist1, edges1)
    h2 = make_hist_plot(combined.columns[1], hist2, edges2)
    h3 = make_hist_plot(combined.columns[2], hist3, edges3)
    h4 = make_hist_plot(combined.columns[3], hist4, edges4)

    bp.output_file('histogram.html', title="histogram")
    bp.show(bp.gridplot([h1, h2, h3, h4], ncols=5, plot_width=400, plot_height=400, toolbar_location=None))

"""The correlation between the feature"""
if 1 == 1:
    train_data = train_data[num_cols + cat_cols]
    train_data['Target'] = target

    C_mat = train_data.corr()

    fig = go.Figure(data=go.Heatmap(
        z=C_mat.values,
        x=C_mat.columns.values,
        y=C_mat.columns.values)
    )
    fig.show()

"""From the correlation heat map above,
we see that about 15 features are highly correlated with the target.
One Hot Encode The Categorical Features :
We will encode the categorical features using one hot encoding."""
if 1 == 1:
    def oneHotEncode(df, colNames):
        for col in colNames:
            if (df[col].dtype == np.dtype('object')):
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)

                # drop the encoded column
                df.drop([col], axis=1, inplace=True)
        return df


    print('There were {} columns before encoding categorical features'.format(combined.shape[1]))
    combined = oneHotEncode(combined, cat_cols)
    print('There are {} columns after encoding categorical features'.format(combined.shape[1]))

"""Now, split back combined dataFrame to training data and test data"""
if 1 == 1:
    def split_combined():
        global combined
        train = combined[:1460]
        test = combined[1460:]

        return train, test


    train, test = split_combined()

"""Second : Make the Deep Neural Network
Define a sequential model
Add some dense layers
Use ‘relu’ as the activation function for the hidden layers
Use a ‘normal’ initializer as the kernal_intializer
Initializers define the way to set the initial random weights of Keras layers.
We will use mean_absolute_error as a loss function
Define the output layer with only one node
Use ‘linear ’as the activation function for the output layer"""

if 1 == 1:
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(128, kernel_initializer='normal', input_dim=train.shape[1], activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    NN_model.summary()