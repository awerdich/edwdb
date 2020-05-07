import os
import pandas as pd

from werdich_notes.utils.rpdrutils import get_file_list, get_field_names, parse_report_list

#%% Parse RPDR PRG Notes

data_root = os.path.normpath('/mnt/obi0/phi/ehr/rpdr/MAMMOGRAPHY2_20190905_142750')
save_dir = os.path.join(data_root, 'parquet', 'PRG')
txt_filename = 'IH24_20190905_142750_Prg.txt'

file_list = get_file_list(data_root, txt_filename)

#%% Run the parsing function on all files
file_list = get_file_list(data_root, txt_filename)

for f, file in enumerate(file_list):

    field_names = get_field_names(file_list[0])

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
    df = parse_report_list(report_list = report_list,
                           field_names = field_names,
                           datetime_field_name = 'Report_Date_Time')

    # Add the folder name
    report_name = os.path.basename(os.path.dirname(os.path.dirname(file)))
    parquet_name = report_name+'_PRG.parquet'
    parquet_file = os.path.join(save_dir, parquet_name)
    df = df.assign(Report=report_name)

    # Save the data frame
    df.to_parquet(parquet_file)
