import os
import typing
import pandas as pd
import numpy as np
from dotenv import load_dotenv

random_state = 123 # Random state for pd sampling.

from sqlalchemy import create_engine, MetaData

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

#%% Files and directories
data_root = os.path.normpath('/mnt/obi0/andreas/data/ecg')
parquet_dir = os.path.join(data_root, 'parquet')
dotenv_file = os.path.normpath('/mnt/obi0/andreas/config/credentials.env')
load_dotenv(dotenv_file)

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

def med_interval(s):
    """ Convert OrderStartDTS and OrderEndDTS in medications table
    into a ps.Interval object """
    if not pd.isna(s['OrderStartDTS']):
        start = pd.Timestamp(s['OrderStartDTS'])
        if not pd.isna(s['OrderEndDTS']):
            end = pd.Timestamp(s['OrderEndDTS'])
        else:
            end = pd.Timestamp.today()
        if start>=end:
            end=start
    return pd.Interval(left=start, right=end, closed='neither')

def concat_rx(df):
    df_concat = df.groupby('file').\
        agg({'MedicationID': lambda x: list(x),
             'MedicationDSC': lambda x: list(x)}).\
        reset_index(drop=False).\
        rename(columns={'MedicationID': 'MedicationIDList',
                        'MedicationDSC': 'MedicationDSCList'})
    df_rest = df.drop(columns=['MedicationID', 'MedicationDSC']).drop_duplicates()
    df_concat = df_concat.merge(df_rest, on='file', how='left')
    return df_concat

def ecg_in_TestOrderInterval(idlist, ecg_df, med_df, medid_target_list, PharmaceuticalClassDSC, TherapeuticClassDSC):

    # ECGs with medications in target list
    ecg_med_target_list = []
    # All other ECGs with medication lists
    ecg_med_nontarget_list = []

    # Go through each patient and combine ECGs with medications
    for p, mrn in enumerate(idlist):

        if (p+1) % 100 == 0:
            print(f'Processing mrn {p+1} of {len(idlist)}')

        # ECGs and meds for that patient
        mrn_ecg = ecg_df[ecg_df.MRN==mrn]
        mrn_med = med_df[med_df.MRN==mrn]

        # Only use mrns that have medication information attached
        if mrn_med.shape[0] > 0:
            mrn_med_target = mrn_med[mrn_med.MedicationID.isin(medid_target_list)]
            if mrn_med_target.shape[0] > 0:
                # Generate OrderInterval from start/end dates
                mrn_med_target = mrn_med_target.assign(OrderInterval = mrn_med_target.apply(med_interval, axis = 1))

                # Combine TestDate and TestTime columns to generate a TimeStamp
                mrn_ecg = mrn_ecg.assign(TestTimestamp = pd.to_datetime(mrn_ecg['TestDate'].apply(str)+' '+mrn_ecg['TestTime']))

                # Merge med on ecg table and determine if TestTimesamp is in Orderinterval for each ECG and each Medication
                ecg_med = mrn_ecg.merge(mrn_med_target, on = 'MRN', how='left')
                ecg_med = ecg_med.assign(TestInOrderInterval = ecg_med.\
                                         apply(lambda s: s['TestTimestamp'] in s['OrderInterval'], axis = 1))

                # We are only interested in those ECGs that occurred within the OrderInterval
                ecg_med_int = ecg_med[ecg_med.TestInOrderInterval==True]

                if ecg_med_int.shape[0] > 0:
                    ecg_med_int_concat = concat_rx(ecg_med_int)
                    ecg_med_int_concat = ecg_med_int_concat.assign(PharmaceuticalClassDSC=PharmaceuticalClassDSC,
                                                                   TherapeuticClassDSC=TherapeuticClassDSC)
                    ecg_med_target_list.append(ecg_med_int_concat)
            else:
                # If there are no medications in target list, save ECGs and meds in a separage dataframe
                ecg_med = mrn_ecg.merge(mrn_med, on=['MRN'], how='left')
                ecg_med_concat = concat_rx(ecg_med).drop_duplicates(subset=['MRN', 'file']).reset_index(drop=True)
                # We do not need the dates any more (because we concatenated the meds)
                ecg_med_concat = ecg_med_concat.drop(columns=['OrderStartDTS', 'OrderEndDTS'])
                ecg_med_nontarget_list.append(ecg_med_concat)

    target_df = pd.concat(ecg_med_target_list, ignore_index=True).reset_index(drop=False)
    nontarget_df = pd.concat(ecg_med_nontarget_list, ignore_index=True).reset_index(drop=False)

    return target_df, nontarget_df

#%% Query functions

def medications(id_list, mrn_type='MGHMRN'):
    query = (
        f'SELECT "patientids"."PatientID", '
        f'"EMPI", "{mrn_type}", "OrderStartDTS", "OrderEndDTS", "MedicationID" , "MedicationDSC" '
        f'FROM patientids '
        f'INNER JOIN medications '
        f'ON "patientids"."PatientID" = "medications"."PatientID" '
        f'WHERE "{mrn_type}" IN {ids_to_str(id_list)}')
    return pd.read_sql(query, engine)

#%% Load and clean ECG meta data

# ECG meta data (still need to update ECG patients in the db)
ecg_meta_file = os.path.join(parquet_dir, 'MGH_RAW_FROM2001.parquet')
ecg_raw = pd.read_parquet(ecg_meta_file)

ecg_raw = ecg_raw.rename(columns={'PatientID': 'MRN'})
ecg_raw2 = ecg_raw.loc[~ecg_raw.MRN.isin(broken_mrn_list)].sample(frac=1, random_state=random_state)

# Filter more bad mrns
mrnlen=7
ecg_df_mrn_list = list(ecg_raw2.MRN.unique())
ecg_df_mrn_list_filtered = [s for s in ecg_df_mrn_list if (len(s)==mrnlen) & (s.isdigit())]
ecg_df = ecg_raw2[ecg_raw2.MRN.isin(ecg_df_mrn_list_filtered)]

ecg_df = ecg_df.astype({'AcquisitionDate': 'datetime64[ns]',
                        'TestDate': 'datetime64[ns]',
                        'MRN': 'int'})

# Saving the cleaned data
ecg_meta_cleaned_file = ecg_meta_file.replace('.parquet', '_cleaned.parquet')
ecg_df.to_parquet(os.path.join(parquet_dir, ecg_meta_cleaned_file))

print(f'Original ECG data: {len(ecg_raw.file.unique())} .xml files from {len(ecg_raw.MRN.unique())} mrns')
print(f'Cleaned ECG data : {len(ecg_df.file.unique())} .xml files from {len(ecg_df.MRN.unique())} mrns')

#%% Chunks

# Divide mrns in chunks of 1000
n_patients_per_chunk = 1000
mrn_chunk_list = list(chunks(list(ecg_df.MRN.unique()), n_patients_per_chunk))
mag = int(np.floor(np.log10(len(mrn_chunk_list)))) + 1
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
ref_med = ref_med.astype({'MedicationID': 'int'})

# CV MEDS
TherapeuticClassDSC = 'CARDIOVASCULAR'
ref_med_cv = ref_med[ref_med.TherapeuticClassDSC==TherapeuticClassDSC].\
    drop_duplicates().dropna(subset=['MedicationID']).reset_index(drop=True)
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

# Medication target list: beta blockers
medid_target_list = beta_MedicationID_list
# Cardiovascular medications list: for controls
medid_cv_list = cv_MedicationID_list

#%% Sort out ECG lists

for chunk, idlist in enumerate(mrn_chunk_list):

    print(f'Starting part {chunk+1} or {len(mrn_chunk_list)}.')
    print(f'Querying database for {len(idlist)} mrns.')

    # Database query for medications
    #idlist=idlist[0:300]
    med_df = medications(idlist)
    med_df = med_df.dropna(subset=['MGHMRN', 'OrderStartDTS', 'MedicationID']).\
        astype({'MGHMRN': 'int', 'MedicationID': 'int'}).\
        rename(columns={'MGHMRN': 'MRN'}).reset_index(drop=True)

    print(f'Completed. Found {len(med_df.PatientID.unique())} mrns with medication data.')

    target_df, nontarget_df = ecg_in_TestOrderInterval(idlist=idlist,
                                                       ecg_df=ecg_df,
                                                       med_df=med_df,
                                                       medid_target_list=medid_target_list,
                                                       PharmaceuticalClassDSC=PharmaceuticalClassDSC,
                                                       TherapeuticClassDSC=TherapeuticClassDSC)

    # Save part file
    base_part_name = os.path.basename(ecg_meta_file).split('.')[0] + '_'+str(chunk).zfill(mag) + '.parquet'
    target_part_name = 'beta_' + base_part_name
    nontarget_part_name = 'other_' + base_part_name

    # Need to convert OrderInterval column to string
    target_df = target_df.astype({'OrderInterval': 'str'})
    target_df.to_parquet(os.path.join(data_root, target_part_name))
    nontarget_df.to_parquet(os.path.join(data_root, nontarget_part_name))













