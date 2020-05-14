""" Transfer ECG patient data from EDW to the internal database """

import os
import pandas as pd
import glob
from dotenv import load_dotenv

random_state = 123 # Random state for pd sampling.

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

from ehr.ehr_db import EhrDb
from ehr.edw import Epic, ExternalIdentity

#%% Database connection

dotenv_file = os.path.normpath('/mnt/obi0/andreas/config/credentials.env')
load_dotenv(dotenv_file)

# Epic instance
def create_epic():
    epic = Epic(edw_user=os.environ['EDW_USER'],
                edw_password=os.environ['EDW_PASSWORD'],
                db='PHS',
                out_dir=data_root)
    return epic

# Development database
db = EhrDb(user = os.environ['EHR_USER'],
           password = os.environ['EHR_PASSWORD'],
           host='obi-cpu8',
           port='5432',
           db='ehr_dev_andreas')

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

#%% Files and directories
data_root = os.path.normpath('/mnt/obi0/andreas/data/ecg')
parquet_dir = os.path.join(data_root, 'parquet')

# Updated data file
ecg_meta_file = os.path.join(parquet_dir, 'MGH_RAW_FROM2001_cleaned.parquet')
ecg_df = pd.read_parquet(os.path.join(parquet_dir, ecg_meta_file))
ecg_df = ecg_df[['MRN', 'TestDate', 'TestTime', 'file']]

# Old data file
ecg_old_dir = os.path.normpath('/mnt/obi0/phi/ecg')
ecg_meta_file_old = 'MGH_RAW_all_mrn.parquet'
ecg_raw = pd.read_parquet(os.path.join(ecg_old_dir, ecg_meta_file_old))
ecg_raw = ecg_raw.rename(columns={'mrn': 'MRN'})
# MRNs that we already know are not working
broken_mrn_file = os.path.join(ecg_old_dir, 'MGH_RAW_all_mrn.parquet.broken.txt')
broken_mrn_list = list(set(pd.read_csv(broken_mrn_file, header=None).loc[:,0]))
ecg_raw2 = ecg_raw.loc[~ecg_raw.MRN.isin(broken_mrn_list)].sample(frac=1, random_state=random_state)
# We need to clean these MRNS
# Filter more bad mrns
mrnlen=7
ecg_df_mrn_list = list(ecg_raw2.MRN.unique())
ecg_df_mrn_list_filtered = [s for s in ecg_df_mrn_list if (len(s)==mrnlen) & (s.isdigit())]
ecg_df2 = ecg_raw2[ecg_raw2.MRN.isin(ecg_df_mrn_list_filtered)]

ecg_df2 = ecg_df2.astype({'TestDate': 'datetime64[ns]',
                          'MRN': 'int'})

#%% Filter those MRNs that are only in the new file and not in the old one
# We want to know the MRNs that we need to update in the database
ecg_df_filtered = ecg_df.loc[~ecg_df.MRN.isin(ecg_df2.MRN.unique())]
print(f'MRNs in the new file {len(ecg_df.MRN.unique())}')
print(f'MRNs in the old file {len(ecg_df2.MRN.unique())}')
print(f'MRNs in the filtered file {len(ecg_df_filtered.MRN.unique())}')
update_mrn_list_file = 'MGH_RAW_all_mrn_update.parquet'
ecg_df_filtered.to_parquet(os.path.join(ecg_old_dir, update_mrn_list_file))
mrnlist = list(ecg_df_filtered.MRN.unique())
print(f'Number of MGH MRNs to load into the database: {len(mrnlist)}')
mrnlist_chunks = list(chunks(mrnlist, 800))

#%% Get the EDW PatientIDs
df_list = []
for c, idchunk in enumerate(mrnlist_chunks):
    print(f'Running query {c+1} of {len(mrnlist_chunks)}.')
    epic = create_epic()
    dfid = epic.patientids_from_external(idchunk, external_identity=ExternalIdentity.MGHMRN)
    df_list.append(dfid)
    epic.close()

#%% Save the data
df_id = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
df_id_file = 'MGH_RAW_all_mrn_update_PatientID.parquet'
df_id.to_parquet(os.path.join(ecg_old_dir, df_id_file))


