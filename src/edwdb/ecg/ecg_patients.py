import os
import pandas as pd
from dotenv import load_dotenv

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker

#%% Files and directories
data_root = os.path.normpath('/mnt/obi0/andreas/data/ecg')
dotenv_file = os.path.normpath('/mnt/obi0/andreas/config/credentials.env')
load_dotenv(dotenv_file)

# Create an engine
engine = create_engine(os.environ['EHR_DEV_URL'])

# Reflect all tables
meta = MetaData()
meta.reflect(bind=engine)

# Create a configured "Session" class
Session = sessionmaker(bind=engine)

#%% Table definitions
ref_med_tab = Table('reference_medication', meta, autoload=True, autoload_with=engine)

#%% Get the medication reference table
sess = Session()
query_dict = sess.query(ref_med_tab).all()
sess.close()
df = pd.DataFrame(query_dict)
