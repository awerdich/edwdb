import os
import pandas as pd
from dotenv import load_dotenv

# Custom imports
from ehr import edw

#%% Files and directories
data_root = os.path.normpath('/mnt/obi0/andreas/data/ecg')
parquet_dir = os.path.join(data_root, 'parquet')
mgh_meta_file = os.path.join(parquet_dir, 'MGH_RAW_all.parquet')

dotenv_file = os.path.normpath('/mnt/obi0/andreas/config/credentials.env')
load_dotenv(dotenv_file)

#%% ECG meta data
# Read all patient metadata
df = pd.read_parquet(mgh_meta_file)
print(f'Loaded meta data for {len(df.file.unique())} ecgs from {len(df.PatientID.unique())} mrns.')
# rename mrn and select only the important columns
df = df.rename(columns={'PatientID': 'mrn'}).reset_index(drop=True)

#%% get PatientIDs
# get a few mrns for testing
mrn_list = list(df.mrn.unique()[0:10])

# Epic class for pulling data from edw
epic = edw.Epic(edw_user = os.environ['EDW_USER'],
                edw_password = os.environ['EDW_PASSWORD'],
                db = 'PHS',
                out_dir = data_root,
                dataset_name = 'ecg')

