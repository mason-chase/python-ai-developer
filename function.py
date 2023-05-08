import pandas as pd
import numpy as np
import plotly.express as px  
import plotly.io as pio
import numpy as np
from scipy.stats import skew
import os
import pickle
import warnings
warnings.filterwarnings("ignore")



#---------------------------------------------------------------------------------------------------------

def plot_2d_feauture(df,feature_1,feature_2 ):
    fig = px.scatter(
        data_frame=df,
        x=feature_1,
        y=feature_2,
        color="Churn",
        color_discrete_sequence=['red', 'green'],
         template='plotly',     # 'ggplot2', 'seaborn', 'simple_white', 'plotly',
                                # 'plotly_white', 'plotly_dark', 'presentation',
                                # 'xgridoff', 'ygridoff', 'gridon', 'none'
         title='visualization: {} and {} '.format(feature_1, feature_2),
         hover_name='Churn',       
         height=500,                
        #  width=500
    )
    # pio.show(fig)
    return fig

#---------------------------------------------------------------------------------------------------------

def plot_3d_feauture(df, feature_1,feature_2,  feature_3):
    # Use for animation rotation at the end
    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5
    fig = px.scatter_3d(
         data_frame=df,
         x=feature_1,
         y=feature_2,
         z=feature_3,
        color="Churn",
        color_discrete_sequence=['red', 'green'],
        template='plotly',    # 'ggplot2', 'seaborn', 'simple_white', 'plotly',
                            # 'plotly_white', 'plotly_dark', 'presentation',
                            # 'xgridoff', 'ygridoff', 'gridon', 'none'
        title=f'see {feature_1}, {feature_2} and {feature_3}',
        hover_name='Churn',        # values appear in bold in the hover tooltip
        height=700,                 # height of graph in pixels
         )

    # pio.show(fig)
    return fig

#---------------------------------------------------------------------------------------------------------

def read_data():
    dir_read_data = "./data"
    """Read splited data for trian , test and validation set"""
    try:
        x_train = pd.read_csv(dir_read_data + "/x_train.csv")
        x_train.set_index("customerID", inplace = True)

        x_test= pd.read_csv(dir_read_data + "/x_test.csv")
        x_test.set_index("customerID", inplace = True)

        x_val= pd.read_csv(dir_read_data + "/x_val.csv")
        x_val.set_index("customerID", inplace = True)


        y_train= pd.read_csv(dir_read_data + "/y_train.csv")

        y_val= pd.read_csv(dir_read_data + "/y_val.csv")

        y_test= pd.read_csv(dir_read_data + "/y_test.csv")

    except Exception as er:
        print(er)
    return x_train, x_test, x_val, y_train, y_val, y_test


#---------------------------------------------------------------------------------------------------------

