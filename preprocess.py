import os, pickle
import geopandas as gpd
from utils.datetime import *
from utils.data_utils import progress_indicator

data_dir = "data"
dataset_name = ["tgnn_data", "multiwave_data"]

def preprocess_tokyo_WMN(): # multiwave

    shapefile_path = "tokyo_shapefile/tokyo.shp"
    mobility_dir = "mobility"
    
    # read files
    cwd = os.getcwd()

    os.chdir(os.path.join(data_dir, dataset_name))

    # 1. Tokyo 23 zone shapefile
    jcode_23 = gpd.read_file(shapefile_path)["JCODE"][:23]

    # 2. Mobility data: {"20200201":{('123','123'):12345,...},...}
    #    20200201 to 20210620: 506 days
    mobilities = dict()
    qbar1 = progress_indicator(desc='Reading mobility data')
    



    os.chdir(cwd)
    del cwd