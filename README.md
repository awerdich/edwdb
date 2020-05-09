# edwdb

This package contains functions for retrieving data from edw and storing them in a local database for clinical data.
Probably most useful to be imported in other specific research projects.
The classes edw and ehr_db are in the ehr_db repository.

## installation
The python scripts require a few packages. You can either make a new environment for this 
or install it into an existing environment.

### install python module
```bash
# making a new env for this OR use existing env (see below)
conda env create -f environment.yml
conda activate ehr_db

# use an existing environment, e.g. "sysmex_tf2", "ehr_classification"
conda activate sysmex_tf2

# optional: install additional dependencies into existing environment
conda env update --file path/to/edw/environment.yml

# go into top folder of this project
cd path/to/ehr_db/

# install this as a module into conda env
python setup.py develop
```

### configuration
You can make a `.env` file with EDW_USER and EDW_PASSWORD, save it in your home directory (remember to `chmod go-rwx`),
and use environment variables with your credentials in python.

## using it

The edw module has an epic class that makes it easy to pull structured data:

```python
import os
import pandas as pd
from ehr import edw

# load environment variables
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# load patient_ids and bwh_mrns for testing from disk
patient_ids = pd.read_csv('/mnt/obi0/phi/ehr/note_pull/edw_testdir/test_patientids.tsv', sep='\t').PatientID
bwh_mrns = pd.read_csv('/mnt/obi0/phi/ehr/note_pull/edw_testdir/test_patientids.tsv', sep='\t').BWHMRN

# establish a connection
epic = edw.Epic(
    edw_user=os.getenv("EDW_USER"),
    edw_password=os.getenv("EDW_PASSWORD"),
    db='PHS',
    out_dir='.')

# you can call methods directly to get dataframes
# this fetches mrn numbers for a list of patientids
id_mapping_from_patientids = epic.patientids(patient_list=patient_ids)

# you can also map external ids to patient ids
# see edw.ExternalIdentity for what's available 
# or match against any id type in the system
id_mapping_from_bwhmrns = epic.patientids_from_external(
   external_patient_list=bwh_mrns,
   external_identity=edw.ExternalIdentity.BWHMRN
)

# this fetches encounters with icd10 codes
encounterdx = epic.encounterdx(patient_ids)

# for large queries, it is better to fetch data in chunks.
# you can adjust the chunksize depending on how long queries take:
chunk_sizes = {
    'encounterdx': 100,
    'tobacco': 1000
}

# now fetch results, in batches, return dict of dataframes
tobacco_data_dict = epic.fetch(patient_ids, chunk_sizes)

# fetch notes in chunks of 100 patients, write each chunk to disk and do not aggregate them
epic.fetch(patient_ids, chunk_sizes={'notes': 100}, overwrite=True, save_format='parquet', save_only=True)

# this is what's currently implemented and what fetch_all retrieves:
chunk_sizes = {
    'demographics': 10000,
    'patientids': 10000,
    'encounterdx': 1000,
    'admitdx': 1000,
    'medicalhx': 1000,
    'familyhx': 1000,
    'surgicalhx': 1000,
    'problemlist': 1000,
    'tobacco': 1000,
    'labs': 100,
    'vitals': 100,
    'medications': 100,
    #'notes': 100,              # notes take a long time, so retrieve them separately
}

# fetch_all is a convenience wrapper that pulls different data types at once (excluding notes)
all_data = epic.fetch_all(patient_ids)

# you can also fetch hematology results (11 parameters)
# this takes the start date and numbers of days as arguments
# 2019 and 2020 are imported already
epic.hematology(start_date='2018-12-30 00:00:00', n_days=7)

# remember to close connection when done
epic.close()
```

This also adds an `edw_pull` console script that asks for credentials and pulls default data tables.
You need to activate the python environment for this to work:

```bash
(ehr_db) mhomilius@obi-gpu1:~$ edw_pull mj715 phs /mnt/obi0/phi/ehr/note_pull/edw_testdir/test_patientids.txt --outdir .
EDW Password:
patientids: 100%|████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.05s/it]
demographics: 100%|██████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.17s/it]
tobacco: 100%|███████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.05s/it]
problemlist: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.09s/it]
medicalhx: 100%|█████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.50s/it]
encounterdx: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.11s/it]
labs: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.61s/it]
vitals: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.64s/it]
medications: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.87s/it]
```

# local database

## resetting your password and connecting
Do this on obi-cpu8, where the postgres server is running:
```bash
# connect to postgres db
psql ehr_prod
# then change password
\password
```

The connection string is something like shown below. You should add the postgres user and password to your `.env` file.
```
postgresql+psycopg2://$USER:$PASSWORD@obi-cpu8:5432/ehr_prod
```

## data import from EDW
```python
import os
import pandas as pd
from ehr import edw, ehr_db

# load environment variables
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

patient_ids = pd.read_csv('/mnt/obi0/phi/ehr/note_pull/edw_testdir/test_patientids.tsv', sep='\t').PatientID

# establish a connection to edw
epic = edw.Epic(
    edw_user=os.getenv("EDW_USER"),
    edw_password=os.getenv("EDW_PASSWORD"),
    db='PHS')

## importing data / testing development db
# establish a connection to development db
dev_db = ehr_db.EhrDb(
    user=os.getenv("EHR_USER"), 
    password=os.getenv("EHR_PASSWORD"),
    db='ehr_dev'
)

# please only do this on your development instance
# start a new import, try interrupting it after 3 chunks
dev_db.import_epic(
    name='test',
    description='import test',
    protocol='Novel Phenotyping',
    query_ids=patient_ids,
    chunk_sizes={'demographics':20},
    epic=epic,
)

# you can then resume (specify the correct import id from above)
dev_db.resume_import(import_id, epic)

# you can import a dataframe and target table directly (table needs to exist and columns need to be matching)
dev_db.import_pandas(
    name='Reference Table for Medications',
    description='',
    protocol='Novel Phenotyping',
    import_df=reference_medication_df,
    table_name='reference_medication'
)
```

## retrieving data from the production db
You don't need to use the EhrDB class for this. All you need is a connection to the postgres database, e.g. through AQLAlchemy. See step 1 for details.

```python
# step 1:  establish a connection to production db
# you can either use the EhrDb class to get a sql engine:

prod_db = ehr_db.EhrDb(
    user=os.getenv("EHR_USER"), 
    password=os.getenv("EHR_PASSWORD"),
    db='ehr_prod'
)

# or just use sqlalchemy directly for the connection:
from sqlalchemy import create_engine        
db_url = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}'
engine = create_engine(db_url, pool_size=1, echo=False)

# step 2: retrieve data from sql command to dataframe
# Note: it is necessary to quote the column names since they are capitalized to match EDW
# get 50000 notes from the sysmex import (import number 8)
notes_1 = pd.read_sql('SELECT * FROM notes WHERE "ImportID" = 8 LIMIT 50000;', engine)
# get notes for two specific patient ids
notes_2 = pd.read_sql('SELECT * FROM notes WHERE "PatientID" IN (\'Z1000000X\',\'Z1000000Y\');', engine)
```