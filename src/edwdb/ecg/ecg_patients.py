import os
import typing
import pandas as pd
import numpy as np
from dotenv import load_dotenv

random_state = 123 # Random state for pd sampling.

from sqlalchemy import create_engine, MetaData

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#%% Files and directories
data_root = os.path.normpath('/mnt/obi0/andreas/data/ecg')
parquet_dir = os.path.join(data_root, 'parquet')
dotenv_file = os.path.normpath('/mnt/obi0/andreas/config/credentials.env')
load_dotenv(dotenv_file)

# ECG meta data
ecg_meta_file = os.path.join(parquet_dir, 'MGH_RAW_FROM2001.parquet')
ecg_raw = pd.read_parquet(ecg_meta_file)

# MRNs that we already know are not working
broken_mrn_file = os.path.join(data_root, 'MGH_RAW_all_mrn.parquet.broken.txt')
broken_mrn_list = list(set(pd.read_csv(broken_mrn_file, header=None).loc[:,0]))

#%% Some helper functions
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def ids_to_str(id_list):
    if isinstance(id_list, (tuple, list, set, pd.Series, np.ndarray, typing.MappingView)):
        return "(" + ",".join(["'{}'".format(n) for n in id_list]) + ")"
    elif isinstance(id_list, str):
        return id_list
    else:
        print(type(id_list))
        raise NotImplementedError

#%% Query functions

def medications(id_list):
    query = (
        f'SELECT "patientids"."PatientID", '
        f'"EMPI", "MGHMRN", "OrderStartDTS", "OrderEndDTS", "MedicationID" , "MedicationDSC" '
        f'FROM patientids '
        f'INNER JOIN medications '
        f'ON "patientids"."PatientID" = "medications"."PatientID" '
        f'WHERE "MGHMRN" IN {ids_to_str(id_list)}')
    return pd.read_sql(query, engine)

#%% Clean and shuffle data frame
ecg_raw = ecg_raw.rename(columns={'PatientID': 'mrn'})
ecg_df = ecg_raw.loc[~ecg_raw.mrn.isin(broken_mrn_list)].sample(frac=1, random_state=random_state).\
                reset_index(drop=True)
print(f'Original ECG data: {len(ecg_raw.file.unique())} .xml files from {len(ecg_raw.mrn.unique())} mrns')
print(f'Cleaned ECG data : {len(ecg_df.file.unique())} .xml files from {len(ecg_df.mrn.unique())} mrns')

# Divide mrns in chunks of 1000
n_patients_per_chunk = 1000
mrn_chunk_list = list(chunks(list(ecg_df.mrn.unique()), n_patients_per_chunk))
print(f'Divided mrn list into {len(mrn_chunk_list)} chunks of {n_patients_per_chunk} mrns.')

#%% Database connection

def db_connection():
    db_connect_str = os.environ['EHR_PROD_URL']
    engine = create_engine(db_connect_str)
    meta = MetaData()
    meta.reflect(bind=engine)
    return engine, meta

# Reference medication table from db
engine, meta = db_connection()
query='SELECT * FROM reference_medication'
ref_med = pd.read_sql(query, engine)

#%% Define medication class for betablocker

# CV MEDS
TherapeuticClassDSC = 'CARDIOVASCULAR'
ref_med_cv = ref_med[ref_med.TherapeuticClassDSC==TherapeuticClassDSC].\
    drop_duplicates().reset_index(drop=True)
TherapeuticClassCD = ref_med_cv.TherapeuticClassCD.unique()[0]
cv_MedicationID_list = list(ref_med_cv.MedicationID.unique())
print(f'There are {len(cv_MedicationID_list)} in the TherapeuticClass {TherapeuticClassDSC}.')

# BETA BLOCKER MEDS
PharmaceuticalClassDSC = 'BETA-ADRENERGIC BLOCKING AGENTS'
ref_med_beta = ref_med_cv[ref_med_cv.PharmaceuticalClassDSC==PharmaceuticalClassDSC].\
    drop_duplicates().reset_index(drop=True)
PharmaceuticalClassCD = ref_med_beta.PharmaceuticalClassCD.unique()[0]
beta_MedicationID_list = list(ref_med_beta.MedicationID.unique())
print(f'There are {len(beta_MedicationID_list)} medications in the PharmaceuticalClass {PharmaceuticalClassDSC}.')

#%% Get some patients from the list
patient_list = mrn_chunk_list[0][0:10]

#%% Query
meds = medications(patient_list)

# Filter the output
meds_beta = meds[meds.MedicationID.isin(beta_MedicationID_list)].drop_duplicates().reset_index(drop=True)

# Need to answer the question: For every ECG, was the patient on a particular medication?
beta_PatientID_list = meds_beta.PatientID.unique()
PatientID = beta_PatientID_list[0]
meds_beta_PatientID = meds_beta[meds_beta.PatientID==PatientID]















