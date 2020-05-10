import os
import pandas as pd
from dotenv import load_dotenv

# Custom imports
from ehr import edw

#%% Files and directories
data_root = os.path.normpath('/mnt/obi0/andreas/data/ecg')
parquet_dir = os.path.join(data_root, 'parquet')

dotenv_file = os.path.normpath('/mnt/obi0/andreas/config/credentials.env')
load_dotenv(dotenv_file)