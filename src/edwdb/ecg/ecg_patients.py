import os
import pandas as pd
from dotenv import load_dotenv

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
ecg_cols = ['PatientID', 'TestDate', 'TestTime', 'file']
df_ecg = df[ecg_cols].rename(columns={'PatientID': 'mrn'}).reset_index(drop=True)
df_ecg_file = os.path.join(parquet_dir, 'MGH_RAW_all_mrn.parquet')
df_ecg.to_parquet(df_ecg_file)