import os
import pandas as pd
from dotenv import load_dotenv

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker

#%% Files and directories
data_root = os.path.normpath('/mnt/obi0/andreas/data/ecg')
dotenv_file = os.path.normpath('/mnt/obi0/andreas/config/credentials.env')
load_dotenv(dotenv_file)




#%% Database connection

# Create an engine
engine = create_engine(os.environ['EHR_DEV_URL'])

# Reflect all tables
meta = MetaData()
meta.reflect(bind=engine)

# Create a configured "Session" class
Session = sessionmaker(bind=engine)

#%% Table definitions
ref_med_tab = Table('reference_medication', meta, autoload=True, autoload_with=engine)
med_tab = Table('medications', meta, autoload=True, autoload_with=engine)
id_tab = Table('patientids', meta, autoload=True, autoload_with=engine)
