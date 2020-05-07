""" Common EDW operations."""

import os
import pandas as pd
import glob
import numpy as np
import urllib
import time
from dotenv import load_dotenv, find_dotenv

from sqlalchemy import create_engine, Table, MetaData, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.orm import sessionmaker, Load, load_only

#%% Connect to the EDW database
# Load environment variable from .env file
load_dotenv(os.path.join(os.environ['HOME'], 'edw.env'))

EDW_USER = os.environ['EDW_USER']
EDW_PASSWORD = os.environ['EDW_PASSWORD']
ACCESS = os.environ['ACCESS']

connect_str = 'DRIVER=FreeTDS;'+\
              'SERVER=phsedw.partners.org;'+\
              'PORT=1433;'+\
              'DATABASE=Epic;'+\
              'UID=Partners\\' + EDW_USER + ';'+\
              'PWD=' + EDW_PASSWORD + ';'+\
              'TDS_Version=8.0;'

# Database engine
params = urllib.parse.quote_plus(connect_str)

# Disable connection pooling by using poolclass = NullPool
# engine = create_engine('mssql+pyodbc:///?odbc_connect=%s' % params, poolclass = NullPool)

# Enable connection pooling to allow unlimited connections
# engine = create_engine('mssql+pyodbc:///?odbc_connect=%s' % params)

# Use limited number of connections
engine = create_engine('mssql+pyodbc:///?odbc_connect=%s' % params, pool_size = 2, echo = False)


Base = declarative_base()
meta = MetaData()
meta.reflect(bind=engine)

# Create a session factory
Session = sessionmaker()

# This custom-made Session class will create new session objects
Session.configure(bind=engine)

#%% Table mapping

id_tab = Table('Identity_'+ACCESS, meta,
               schema = 'Patient', autoload = True, autoload_with = engine)
class PatientID(Base):
    __table__ = id_tab

    PatientID = id_tab.c.PatientID
    LineNBR = id_tab.c.LineNBR
    PatientIdentityID = id_tab.c.PatientIdentityID
    IdentityTypeID = id_tab.c.IdentityTypeID

    __mapper_args__ = {
        'primary_key': [PatientID, LineNBR]

    }

note_tab = Table('Note_'+ACCESS, meta,
                 schema = 'Clinical', autoload = True, autoload_with = engine)
class Note(Base):
    __table__ = note_tab

    NoteID = note_tab.c.NoteID
    PatientID = note_tab.c.PatientID
    PatientLinkID = note_tab.c.PatientLinkID
    InpatientNoteTypeDSC = note_tab.c.InpatientNoteTypeDSC

    __mapper_args__ = {
        'primary_key': NoteID
    }

note_text_tab = Table('NoteText_'+ACCESS, meta,
                      schema = 'Clinical', autoload = True, autoload_with = engine)
class NoteText(Base):
    __table__ = note_text_tab

    NoteCSNID = note_text_tab.c.NoteCSNID
    LineNBR = note_text_tab.c.LineNBR
    NoteID = note_text_tab.c.NoteID
    NoteTXT = note_text_tab.c.NoteTXT

    __mapper_args__ = {
        'primary_key': [NoteCSNID, LineNBR]
    }

enc_tab = Table('PatientEncounter_'+ACCESS, meta,
                        schema = 'Encounter', autoload = True, autoload_with = engine)
class Encounter(Base):
    __table__ = enc_tab

    PatientEncounterID = enc_tab.c.PatientEncounterID

    __mapper_args__ = {
        'primary_key': PatientEncounterID
    }

encdia_tab = Table('PatientEncounterDiagnosis_'+ACCESS,
                   meta, schema = 'Encounter', autoload = True, autoload_with = engine)
class EncounterDiagnosis(Base):
    __table__ = encdia_tab

    PatientEncounterID = encdia_tab.c.PatientEncounterID
    LineNBR = encdia_tab.c.LineNBR
    DiagnosisID = encdia_tab.c.DiagnosisID

    __mapper_args__ = {
        'primary_key': [PatientEncounterID, LineNBR]
    }

icddia_tab = Table('ICDDiagnosis', meta, schema = 'Reference', autoload = True, autoload_with = engine)
class ICDDiagnosis(Base):
    __table__ = icddia_tab

    DiagnosisID = icddia_tab.c.DiagnosisID

    __mapper_args__ = {
        'primary_key': DiagnosisID
    }

# REFERENCE TABLES
med_ref_tab = Table('Medication', meta,
                    schema = 'Reference', autoload = True, autoload_with = engine)
class MedRef(Base):
    __table__ = med_ref_tab

    MedicationID = med_ref_tab.c.MedicationID
    MedicationDSC = med_ref_tab.c.MedicationDSC

    __mapper_args__ = {
        'primary_key': MedicationID

    }

identity_type_tab = Table('IdentityType', meta, schema = 'Reference', autoload = True, autoload_with = engine)

#%% Functions to pull reference tables

def identity_type_table():
    """
    Download reference table with identity types
    Args:   None
    Returns:
            pd.DataFrame()
    """

    sess = Session()
    q0 = sess.query(identity_type_tab).all()
    sess.close()

    df = pd.DataFrame(q0)
    df = df.rename(columns = {'MasterPersonIndexTypeCD': 'IdentityTypeID'})
    df = df.reset_index(drop = True)

    return df[['IdentityTypeID', 'IdentityTypeNM', 'AbbreviationTXT']]
#%% Run identity table
save_dir = os.path.normpath('/mnt/obi0/andreas/data/notes')
id_table_name = 'identity_table.parquet'
df = identity_type_table()
df.to_parquet(os.path.join(save_dir, id_table_name))


#%% Main class to pull notes

class Notebuilder:
    """
    Main class to pull notes from EDW.
    Args:
        id_list:                    List of EDW PatientIDs.
        max_patients:               Maximum number of patients to be submitted in each query.
        InpatientNoteTypeDSC_list: list of note types, e.g. ['Progress Notes', 'Discharge Summaries']
    """

    def __init__(self, id_list, max_query_items = 100, InpatientNoteTypeDSC_list = None):

        self.id_list = id_list # list of PatientIDs
        self.max_query_items = max_query_items # maximum number of samples per query
        self.InpatientNoteTypeDSC_list = InpatientNoteTypeDSC_list

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def download(self,
                 file = os.path.join(os.environ['HOME'], 'notes.parquet'),
                 n_patients_per_file = None,
                 pause = None):
        """
        Main method to download notes to disk.
        Args:
            file: parquet file name and path for the output file.
            n_patients_per_file: Number of patients per file. If none, only one file will be created.
        Returns:
            Method does not return anything. Saves file to disk.
        """

        if n_patients_per_file is None:
            # Save all patients in a single file
            n_patients_per_file = len(self.id_list)

        # Check first if files are already on disk
        n_saved_files, updated_id_list = self.verify_download(file)
        patient_id_chunks = list(self.chunks(updated_id_list, n_patients_per_file))

        mag = int(np.floor(np.log10(len(patient_id_chunks)+n_saved_files)))+1
        base_filename = os.path.basename(file).split('.')[0]
        file_dir = os.path.dirname(file)

        for c, patientid_list in enumerate(patient_id_chunks):

            if len(patient_id_chunks) > 1:
                part_name = base_filename + '_' + str(c+n_saved_files).zfill(mag) + '.parquet'
            else:
                part_name = base_filename + '.parquet'
            parquet_file = os.path.join(file_dir, part_name)

            # Save list of ids that are processing right now
            txt_filename = part_name.split('.')[0] + '.txt'
            txt_file = os.path.join(file_dir, txt_filename)
            pd.DataFrame({'id': patientid_list}).to_csv(txt_file, index = False)

            print('Building file {f} of {ftotal}: {fname}'.format(f = c+1+n_saved_files,
                                                                  ftotal = len(patient_id_chunks)+n_saved_files,
                                                                  fname = parquet_file))

            # Pull the notes: returns list of data frames with notes
            df_list_notes = self.notes_from_patientid(patientid_list)
            # Concatenate list of data frames to get the query_df
            df_notes = pd.concat(df_list_notes, axis=0, ignore_index=True)
            # Select latest notes and concatenate LineNBR in df_notes
            df_notes_concat = self.concatenate_notes(df_notes)
            # EDW Atlas: PatientID in df_notes_concat: Only populated on 10% of rows - use PatientLinkID instead.
            # The unique ID of the patient who is associated to this note.
            # This column is frequently used to link to the PATIENT table.
            # We will delete PatientID and rename PatientLinkID -> PatientID
            df_notes_concat.drop(columns=['PatientID'], inplace=True)
            df_notes_concat.rename(columns={'PatientLinkID': 'PatientID'}, inplace=True)

            # Compile list of unique PatientEncounterIDs
            encid_list = list(df_notes_concat.PatientEncounterID.dropna().unique())
            # Pull the diagnoses: returns list of data frames with diagnoses codes
            df_list_dia = self.diagnoses_from_encounterid(encid_list)
            # Concatenate list of data frames to get query_df
            df_dia = pd.concat(df_list_dia, axis=0, ignore_index=True)
            # Concatenate LineNBR in df_dia
            df_dia_concat = self.concatenate_diagnoses(df_dia)

            # Outer join of notes and diagnoses data frames
            df = df_notes_concat.merge(right = df_dia_concat, on = ['PatientID', 'PatientEncounterID'], how = 'outer')
            # Set the NoteID as the index and convert to dask data frame
            #df.set_index(keys='NoteID', inplace=True, drop=True)
            df.reset_index(drop = True, inplace = True)
            # We need an odd number of partitions for dask
            #parts = int(np.ceil((len(self.id_list)/self.max_patients)) // 2 * 2 + 1)
            #ddf = dd.from_pandas(df, npartitions = parts)
            # Save ddf to disk
            df.to_parquet(parquet_file, engine = 'pyarrow')

            if c < len(patient_id_chunks)-1 and pause is not None:
                print()
                print('Waiting {} seconds'.format(pause))
                print()
                time.sleep(pause)


    def verify_download(self, file):
        """Returns PatientIDs from files already saved by the download method.
        Args:
            file:  the same base file name provided to .download(file) method.
        Returns:
            list: len(file_list), new_id_list
        """

        file_dir = os.path.dirname(file)
        basefile = os.path.basename(file).split('.')[0]
        fext = os.path.basename(file).split('.')[-1]

        file_list = sorted(glob.glob(os.path.join(file_dir, basefile + '_*.' + fext)))
        # Read in the files and collect the PatientIDs
        saved_id_list = []
        if len(file_list) > 0:
            bad_files = []
            for f, file in enumerate(file_list):
                try:
                    df_file = pd.read_parquet(file, engine='pyarrow', columns=['PatientID'])
                    saved_id_list.extend(list(df_file.PatientID.dropna().unique()))
                except IOError as e:
                    print('Read error: {error} Skipping {error_file}'.format(error=e,
                                                                             error_file=os.path.basename(file)))
                    bad_files.append(file)
            new_id_list = list(set(self.id_list).difference(set(saved_id_list)))
        else:
            new_id_list = self.id_list
        print('Verified {n_files} files with {n_saved_id} IDs saved. Current list has {n_new_id} IDs.'. \
              format(n_files=len(file_list), n_saved_id=len(saved_id_list), n_new_id=len(new_id_list)))
        return len(file_list), new_id_list

    def patient_identities(self, patientid_list):
        """
        Returns a complete set of patient identities
        Args:
            patientid_list: List of EDW PatientIDs.
        Returns:
            pandas data frame with ['PatientID', 'PatientIdentityID', 'LineNBR'] columns .
        Raises:
            AssertionError: Raises an exception if the returned df is empty
        """

        sess = Session()
        query = sess.query(PatientID).filter(PatientID.PatientID.in_(patientid_list))
        query = query.options(Load(PatientID).\
                              load_only('PatientID', 'LineNBR', 'PatientIdentityID', 'IdentityTypeID'))

        df_q0 = pd.read_sql(query.statement, query.session.bind)
        sess.close()

        return df_q0

    def empi_from_patientid(self, patientid_list):
        """
        Converts a list of EDW PatientIDs into EDW MRNs
        Args:
            patientid_list: List of EDW PatientIDs.
        Returns:
            pandas data frame with ['PatientID', 'EMPI'] columns .
        Raises:
            AssertionError: Raises an exception if the returned df is empty
        """

        sess = Session()

        query = sess.query(PatientID).filter(and_(PatientID.IdentityTypeID == 140,
                                                  PatientID.LineNBR == 2,
                                                  PatientID.PatientID.in_(patientid_list)))

        query = query.options(Load(PatientID).load_only('PatientID', 'PatientIdentityID', 'LineNBR'))


        df_q0 = pd.read_sql(query.statement, query.session.bind)
        sess.close()

        df = df_q0.groupby(by='PatientID').first().reset_index(drop=False)
        df.drop(columns = ['LineNBR'], inplace = True)
        df.columns = ['PatientID', 'EMPI']
        # Merge with original ids so we will never return an empty df
        df_empi = pd.DataFrame({'PatientID': patientid_list}).merge(df, how='left', on='PatientID')

        return df_empi

    def notes_from_patientid(self, patientid_list):
        """
        Queries against Note and NoteText tables
        Args:
            patientid_list: List of EDW PatientIDs.
        Returns:
            list of pandas data frames with notes.
        Raises:
            AssertionError: Raises an exception if the returned df is empty
        """

        # Note columns
        load_note_cols = ['NoteID', 'PatientID', 'PatientLinkID', 'PatientEncounterID',
                          'InpatientNoteTypeDSC', 'LastFiledDTS', 'CurrentAuthorID']
        load_txt_cols = ['NoteCSNID', 'LineNBR', 'ContactDTS', 'NoteTXT']

        # Open one connection for this data pull
        sess = Session()
        # Divide the list of IDs into chunks to make smaller queries
        idchunks = list(self.chunks(patientid_list, self.max_query_items))
        df_list = [] # Save
        for i, idlist in enumerate(idchunks):

            query = sess.query(Note, NoteText).join(Note, Note.NoteID == NoteText.NoteID)

            query_filter_list = [Note.PatientLinkID.in_(idlist)]
            if self.InpatientNoteTypeDSC_list is not None:
                query_filter_list.append(Note.InpatientNoteTypeDSC.in_(self.InpatientNoteTypeDSC_list))

            query = query.filter(*query_filter_list)
            query = query.options(Load(Note).load_only(*load_note_cols),
                                  Load(NoteText).load_only(*load_txt_cols))

            # Submit query to db
            print()
            print('Note query {q} of {qn} with {pn} PatientIDs submitted.'.\
                  format(q = i+1, qn = len(idchunks), pn = len(idlist)))
            start_time = time.time()
            query_df = pd.read_sql(query.statement, query.session.bind)

            if len(query_df) > 0:

                empi_df = self.empi_from_patientid(idlist)
                empi_df.rename(columns={'PatientID': 'PatientLinkID'}, inplace=True)
                query_df = query_df.merge(empi_df, on = 'PatientLinkID', how = 'left')
                print('Returned {rows} rows with {nid} unique NoteIDs and {np} PatientIDs. Completed in {t:.1f} minutes.'.\
                      format(rows = query_df.shape[0],
                             nid = len(query_df.NoteID.unique()),
                             np = len(query_df.PatientLinkID.unique()),
                             t=(time.time() - start_time)/60))
                # Add query_df to list
                df_list.append(query_df)

            else:

                print('Query returned 0 rows. Interrupting.')
                break
        sess.close()
        return df_list

    def diagnoses_from_encounterid(self, encounterid_list):
        """
        Queries against Encounter, EncounterDiagnosis and ICDDiagnosis tables
        Args:
            encounterid_list: List of EDW PatientEncounterIDs
        Returns:
            list of pandas data frames with Diagnosis columns.
        """

        # Diagnoses columns for Encounter, EncounterDiagnosis and ICDDiagnosis tables
        load_enc_cols = ['PatientID', 'EncounterTypeDSC', 'DepartmentDSC']
        load_dia_cols = ['LineNBR', 'DiagnosisID']
        load_icd_cols = ['DiagnosisNM', 'CurrentICD10ListTXT']

        # When testing with a small number of patients, we want larger queries
        if self.max_query_items < 100:
            n_ids_per_query = 200
        else:
            n_ids_per_query = self.max_query_items

        # Query to join Note and NoteText tables
        sess = Session()

        # Remove NAs
        enc_list = [id for id in encounterid_list if ~np.isnan(id)]

        # Divide the list of IDs into chunks to make smaller queries
        idchunks = list(self.chunks(enc_list, n_ids_per_query))
        df_list = []
        for i, idlist in enumerate(idchunks):

            # Joins
            query = sess.query(Encounter, EncounterDiagnosis, ICDDiagnosis). \
                outerjoin(EncounterDiagnosis, EncounterDiagnosis.PatientEncounterID == Encounter.PatientEncounterID). \
                outerjoin(ICDDiagnosis, ICDDiagnosis.DiagnosisID == EncounterDiagnosis.DiagnosisID). \
                filter(Encounter.PatientEncounterID.in_(idlist))

            query = query.options(Load(Encounter).load_only(*load_enc_cols),
                                         Load(EncounterDiagnosis).load_only(*load_dia_cols),
                                         Load(ICDDiagnosis).load_only(*load_icd_cols))

            # Submit query
            print()
            print('Diagnosis query {q} of {qn} with {pn} PatientEncounterIDs submitted.'. \
                  format(q = i+1, qn = len(idchunks), pn = len(idlist)))
            start_time = time.time()
            query_df = pd.read_sql(query.statement, query.session.bind)

            # Drop duplicate columns
            query_df = query_df.loc[:, ~query_df.columns.duplicated()]

            print('Returned {nrows} rows with {enid} unique EncounterIDs. Completed in {t:.0f} seconds.'. \
                  format(nrows=query_df.shape[0],
                         enid=len(query_df.PatientEncounterID.unique()),
                         t=time.time() - start_time))

            # Add query_df to list
            df_list.append(query_df)
        sess.close()
        return df_list

    def concatenate_notes(self, query_df):
        """
        Aggregates notes from notes_from_patientid across LineNBR and NoteCSNID
        Args:
            query_df: diagnoses_from_encounterid
        Returns:
            df_concat: data frame with notes columns.
        """
        # Convert NoteCSNID to int so that it can be sorted
        query_df = query_df.astype({'NoteCSNID': 'int64',
                                    'ContactDTS': 'datetime64[ns]'})

        # Filter the rows with the largest NoteCSNID for each NoteID
        df2 = query_df.groupby('NoteID').agg({'NoteCSNID': 'max'}).reset_index()

        # Left join with the original frame
        df3 = df2.merge(right=query_df, on=['NoteID', 'NoteCSNID'], how='left')

        # Sort by LineNBR
        df_sorted = df3.groupby(['NoteID']).apply(
            lambda x: x.sort_values(by=['LineNBR'], ascending=True)).reset_index(drop=True)
        df_sorted_first = df_sorted.groupby(by=['NoteID', 'NoteCSNID']).first().reset_index(drop=False)
        df_sorted_first.drop(columns=['NoteTXT', 'LineNBR'], inplace=True)

        df_concat = df_sorted.groupby('NoteID'). \
            agg({'NoteTXT': lambda x: None if x.isna().values[0] else ''.join(x)}).reset_index(drop=False)

        # Add the other columns from df_sorted (except for NoteTXT)
        df_concat = df_concat.merge(right=df_sorted_first, on=['NoteID'], how='left')

        return df_concat

    def concatenate_diagnoses(self, query_df):
        """
        Aggregates diagnosis columns from diagnoses_from_encounterid across LineNBR
        Args:
            query_df: concatenated notes_from_patientid output
        Returns:
            df_concat: data frames with notes columns.
        """
        # Sort by LineNBR
        # May need to group by PatientID as well? Not if PatientEncounterID is unique
        df_sorted = query_df.groupby(['PatientEncounterID']).apply(
            lambda x: x.sort_values(by=['LineNBR'], ascending=True)).reset_index(drop=True)

        # Concatenate LineNBR
        df_concat = df_sorted.groupby('PatientEncounterID'). \
            agg({'DiagnosisNM': lambda x: None if x.isna().values[0] else list(x),
                 'DiagnosisID': lambda x: None if x.isna().values[0] else list(x),
                 'CurrentICD10ListTXT': lambda x: None if x.isna().values[0] else list(x)}). \
            reset_index(drop=False)

        df_sorted_first = df_sorted.groupby(by='PatientEncounterID').first().reset_index(drop=False)
        df_sorted_first = df_sorted_first[['PatientID', 'PatientEncounterID', 'EncounterTypeDSC', 'DepartmentDSC']]

        # Add the other columns from df_sorted (except for NoteTXT)
        df_concat = df_concat.merge(right=df_sorted_first, on=['PatientEncounterID'], how='left')

        return df_concat
