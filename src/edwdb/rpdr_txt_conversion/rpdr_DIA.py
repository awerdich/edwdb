import os
import time
import glob
import pandas as pd
from dateutil import parser
import dask.dataframe as dd

#%% directories and files
data_root = os.path.normpath('/home/andreas/data')
dia_dir = os.path.join(data_root, 'TXT', 'DIA')
dia_files = glob.glob(dia_dir + '/*.txt')
parquet_file = os.path.join(data_root, 'dia.parquet')

#%% Helper functions

def get_field_names(txt_file):
    df = pd.read_csv(txt_file, sep = '|', nrows = 10)
    return list(df.columns)

field_names = get_field_names(dia_files[0])

#%% Diagnosis reports

# New dask data frame
ddf = dd.from_pandas(pd.DataFrame(), npartitions = 10)

# Loop over all text files
start = time.time()
for t, txt_file in enumerate(dia_files):

    # Read the entire text into a single string
    with open(txt_file) as fd:
        next(fd)
        data = fd.read()

    # Split rows by \n
    data_split = data.split('\n')
    print('Number of rows in file', len(data_split))

    # Loop over all reports in this file
    series_list = []
    for row in range(len(data_split)):

        # split report by field and remove \n
        row_text = data_split[row].split('|')

        if len(row_text) >= len(field_names):

            ser_dict = dict(zip(field_names, row_text[0:len(field_names)]))
            # Convert datetime
            datetime_field = 'Date'
            timestamp = parser.parse(ser_dict[datetime_field])
            year = timestamp.year
            ser_dict[datetime_field] = timestamp
            ser_dict['Report_Year'] = year
            ser_dict['Report_File'] = os.path.basename(txt_file)

            # Create pandas series object
            series = pd.Series(ser_dict)

            # Add series to list
            series_list.append(series)

        if (row+1)%100000==0:
            print('File {file}/{total_files}, Note {note}/{total_notes}'.format(file = t+1,
                                                                                total_files = len(dia_files),
                                                                                note = row+1,
                                                                                total_notes = len(data_split)))
            print('Time {0:.2f} minutes.'.format((time.time()-start)/60))

    # Data frame for this file from list of pd.series objects
    df = pd.DataFrame(series_list)
    ddf_file = dd.from_pandas(df, npartitions = 5)

    # Add the ddf from this file to the large ddf
    ddf = dd.concat([ddf, ddf_file], axis = 0, interleave_partitions = True)

#%% Save to parquet and feather
ddf.to_parquet(parquet_file)
