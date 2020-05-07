import os
import glob
import pandas as pd

#%% directories and files
data_root = os.path.normpath('/home/andreas/data')
enc_dir = os.path.join(data_root, 'TXT', 'ENC')
txt_file_list = glob.glob(enc_dir + '/*.txt')

#%% Encounters -- Visit information and diagnoses
# Loop over all text files in file_list
for f, txt_file in enumerate(txt_file_list):

    print('File {}/{}'.format(f+1, len(txt_file_list)))
    df = pd.read_csv(txt_file, sep = '|')

    # Type adjustments
    df.Admit_Date = pd.to_datetime(df.Admit_Date)
    df.Discharge_Date = pd.to_datetime(df.Discharge_Date)
    df.MRN_Type = df.MRN_Type.astype('category')
    df.Hospital = df.Hospital.astype('category')

    # Save data
    mag = 2
    basefile = os.path.basename(txt_file).split('.')[0]
    filename = basefile + '_' + str(f).zfill(mag) + '.parquet'
    filepath = os.path.join(enc_dir, filename)
    df.to_parquet(filepath)
