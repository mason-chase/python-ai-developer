import pandas as pd
import numpy as np
import plotly.express as px  
import plotly.io as pio
import numpy as np
from scipy.stats import skew
import os
import pickle


import pytest
import os
from sys import path
path.append(os.path.join(os.getcwd(), '')) 
from function import plot_3d_feauture,plot_2d_feauture

import warnings
warnings.filterwarnings("ignore")


import logging
report_name = "log_test_EDA" 
logging.basicConfig(filename='logs/{}'.format(report_name),
                     format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)



#---------------------------------------------------------------------------------------------------------

def test_plot_2d_feauture():
    dir_read_data = "./data"
    filename_csv = "/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(dir_read_data + filename_csv )

    feature_1 = "MonthlyCharges"
    feature_2 = "TotalCharges"

    dir_read_data = "./figures"
    filename_html = f"/2d_plot_{feature_1}_and_{feature_2}.html"
    # output_file = dir_read_data.join(filename_html)
    output_file = dir_read_data + filename_html


    fig = plot_2d_feauture(df,feature_1,feature_2 )
    pio.write_html(fig, file=dir_read_data +filename_html )
    assert os.path.exists(output_file),  "figur does not exist"


    print("test plot 2d pass", "\n")

#---------------------------------------------------------------------------------------------------------

def test_plot_3d_feauture():
    dir_read_data = "./data"
    filename_csv = "/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(dir_read_data + filename_csv )

    feature_1 = "MonthlyCharges"
    feature_2 = "TotalCharges"
    feature_3 = "tenure"

    dir_read_data = "./figures"
    filename_html = f"/3d_plot_{feature_1}_and_{feature_2}_and_{feature_3}.html"
    # output_file = dir_read_data.join(filename_html)
    output_file = dir_read_data + filename_html


    fig = plot_3d_feauture(df,feature_1,feature_2, feature_3 )
    pio.write_html(fig, file=dir_read_data +filename_html )
    assert os.path.exists(output_file), "figur does not exist"

    print("test plot 3d pass", "\n")

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
   x =  pytest.main(args=['-sv', os.path.abspath(__file__)])
   logger.info(x)