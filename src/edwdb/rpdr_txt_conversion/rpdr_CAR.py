import os
import time
import glob
import pandas as pd
import numpy as np
from dateutil import parser

# Custom import
from werdich_notes.utils.rpdrutils import get_file_list

#%% directories and files
data_root = os.path.normpath('/mnt/obi0/phi/ehr/rpdr/MAMMOGRAPHY2_20190905_142750')
save_dir = os.path.join(data_root, 'parquet')
txt_filename = 'IH24_20190905_142750_Car.txt'

def get_field_names(txt_file):
    df = pd.read_csv(txt_file, sep = '|', nrows = 10)
    return list(df.columns)

def parse_car_report_list(report_list, field_names):
    """
    Convert a list of reports into a pd.DataFrame() object
    Args:
        report_list: list
        field_names: list
    Returns:
        pd.DataFrame()
    """
    ser_list = [] # List of pd.Series() objects
    start_time = time.time()
    for r, report in enumerate(report_list):
        ser_text = report.split('|')
        if len(ser_text) == len(field_names):

            # There are line breaks after [report_end], which end up in the first field of the next row
            ser_text[0] = ser_text[0].replace('\n', '')

            # Create a dictionary from the fields and text lists
            ser_dict = dict(zip(field_names, ser_text))

            # Convert date_time field
            datetime_field = 'Report_Date_Time'
            timestamp = parser.parse(ser_dict[datetime_field])
            year = timestamp.year
            ser_dict[datetime_field] = timestamp
            ser_dict['Report_Year'] = year

            ser_list.append(pd.Series(ser_dict))
        else:
            print('Column mismatch in report {rep} of {total}. Skipping.'.format(rep=(r+1), total=len(report_list)))
            print(ser_text)

        if (r+1)%10000 == 0:
            print('Completed report {rep} of {total} @time: {t: .1f} minutes.'.format(rep = (r+1), total = len(report_list),
                                                                           t = (time.time() - start_time)/60))

    df = pd.DataFrame(ser_list).reset_index(drop = True)
    return df

#%% Run the parsing function on all files
file_list = get_file_list(data_root, txt_filename)

for f, file in enumerate(file_list):

    field_names = get_field_names(file)

    # Read the entire text into a single string
    print('Processing file {}/{}'.format(f+1, len(file_list)))
    with open(file) as fd:
        next(fd)
        print('Reading text data into memory.')
        data = fd.read()

    # Split this large string by [report_end]
    report_list = data.split('[report_end]')
    print('Number of reports', len(report_list))

    # Convert to df
    df = parse_car_report_list(report_list, field_names)

    # Add the folder name
    report_name = os.path.basename(os.path.dirname(os.path.dirname(file)))
    parquet_name = report_name+'_CAR.parquet'
    parquet_file = os.path.join(save_dir, parquet_name)
    df = df.assign(Report=report_name)

    # Save the data frame
    df.to_parquet(parquet_file)

#%% Combine all files
file_list = glob.glob(os.path.join(save_dir, 'CAR', ''))
