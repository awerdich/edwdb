import os
import pandas as pd

from werdich_notes.utils.rpdrutils import get_file_list

#%% Parse RPDR PRG Notes

data_root = os.path.normpath('/mnt/obi0/phi/ehr/rpdr/MAMMOGRAPHY2_20190905_142750')
txt_filename_Dem = 'IH24_20190905_142750_Dem.txt'
txt_filename_Mrn = 'IH24_20190905_142750_Mrn.txt'
txt_filename_Rdt = 'IH24_20190905_142750_Rdt.txt'

file_list_Dem = sorted(list(set(get_file_list(data_root, txt_filename_Dem))))
file_list_Mrn = sorted(list(set(get_file_list(data_root, txt_filename_Mrn))))
file_list_Rdt = sorted(list(set(get_file_list(data_root, txt_filename_Rdt))))

#%% These are structured file. So let's just load and then join them.

def concat_txt2df(file_list):

    """ Load and concatenate txt files """
    df_list = []
    for f, file in enumerate(file_list):

        print('Loading file {} of {}.'.format(f+1,len(file_list)))
        try:
            df_file = pd.read_csv(file, sep = '|')
        except IOError as er:
            print('Skipping {}. Error {}'.format(file, er))
        else:
            df_list.append(df_file)

    # Concat df_list into single df
    return pd.concat(df_list, ignore_index=True)

#%% Run this to load DEM and MRN files
df_Dem = concat_txt2df(file_list_Dem)
df_Mrn = concat_txt2df(file_list_Mrn)
df_Mrn = df_Mrn.rename(columns = {'Enterprise_Master_Patient_Index': 'EMPI'})
df_Rdt = concat_txt2df(file_list_Rdt)

#%% Save data
df_Dem_Mrn = df_Dem.merge(right = df_Mrn, on = ['EMPI', 'EPIC_PMRN'], how = 'outer').\
    reset_index(drop = True)
parquet_dir = os.path.join(data_root, 'parquet')
Dem_Mrn_filename = 'IH24_20190905_142750_Dem_Mrn.parquet'
Rdt_filename = 'IH24_20190905_142750_Rdt.parquet'
df_Dem_Mrn.to_parquet(os.path.join(parquet_dir, Dem_Mrn_filename))
df_Rdt.to_parquet(os.path.join(parquet_dir, Rdt_filename))
