"""
access clinical data in edw / epic
"""

from enum import Enum
import pyodbc
import pandas as pd
import numpy as np
import datetime as dt
import functools
from collections import defaultdict, namedtuple
import typing
from time import sleep
from tqdm.auto import tqdm
from .utils import force_path


def ids_to_str(id_list):
    if isinstance(id_list, (tuple, list, set, pd.Series, np.ndarray, typing.MappingView)):
        return "(" + ",".join(["'{}'".format(n) for n in id_list]) + ")"
    elif isinstance(id_list, str):
        return id_list
    else:
        print(type(id_list))
        raise NotImplementedError


def concat_notes(df):
    df = df.astype({'NoteCSNID': 'int64', 'LineNBR': 'int64', 'PatientEncounterID': 'object'})
    df_maxCSNID = df[['NoteID', 'NoteCSNID']].groupby('NoteID'). \
        agg({'NoteCSNID': 'max'}).reset_index(drop=False). \
        merge(df, on=['NoteID', 'NoteCSNID'], how='left'). \
        dropna(axis=0, subset=['NoteTXT'])
    df_maxCSNID_sorted = df_maxCSNID.groupby('NoteID').apply(
        lambda x: x.sort_values('LineNBR', ascending=True)).reset_index(drop=True)
    meta_cols = df_maxCSNID_sorted.drop(columns=['NoteTXT', 'LineNBR']). \
        drop_duplicates(subset=['PatientLinkID', 'NoteID', 'NoteCSNID'])
    df_concat = df_maxCSNID_sorted.groupby(['PatientLinkID', 'NoteCSNID', 'NoteID']). \
        agg({'NoteTXT': lambda x: ''.join(x)}).reset_index(drop=False). \
        merge(meta_cols, on=['PatientLinkID', 'NoteID', 'NoteCSNID'], how='left'). \
        rename(columns={'PatientLinkID': 'PatientID'})
    return df_concat


def concat_dx(df, as_string=True):
    # TODO PatientEncounterID should uniquely identify the patient, no need to merge on both
    diagnosis_cols = ['DiagnosisID', 'CurrentICD10ListTXT', 'DiagnosisNM']
    meta_cols = list(set(df.columns).difference(set(diagnosis_cols)))
    meta_cols.remove('LineNBR')

    df_concat_dx = df.dropna(axis=0, subset=diagnosis_cols). \
        astype({'PatientEncounterID': 'object',
                'DiagnosisID': 'int64',
                'LineNBR': 'int64'}). \
        groupby(['PatientID', 'PatientEncounterID']). \
        apply(lambda x: x.sort_values('LineNBR', ascending=True)).reset_index(drop=True). \
        groupby(['PatientID', 'PatientEncounterID']). \
        agg({col: lambda x: list(x) for col in diagnosis_cols}).reset_index(drop=False)

    # flatten ICD10Lists
    df_concat_dx.CurrentICD10ListTXT = df_concat_dx.CurrentICD10ListTXT. \
        apply(lambda x: [item for sublist in x for item in sublist])

    # convert to string
    if as_string:
        df_concat_dx.DiagnosisID = ['|'.join(map(str, l)) for l in df_concat_dx.DiagnosisID]
        df_concat_dx.DiagnosisNM = ['|'.join(map(str, l)) for l in df_concat_dx.DiagnosisNM]
        df_concat_dx.CurrentICD10ListTXT = ['|'.join(map(str, l)) for l in df_concat_dx.CurrentICD10ListTXT]

    # deduplicate on encounterid - don't want duplicates due to slightly different metadata
    df = df[meta_cols].drop_duplicates(subset=['PatientID', 'PatientEncounterID'])
    # now merge back with original df and return
    return df.merge(df_concat_dx, on=['PatientID', 'PatientEncounterID'], how='left')


def expand_list_ICD10(df):
    # expand the lists in the ICD10list
    df_icd = df.loc[~df.CurrentICD10ListTXT.isnull()]
    df_empty = df.loc[df.CurrentICD10ListTXT.isnull()]. \
        rename(columns={'CurrentICD10ListTXT': 'CurrentICD10TXT'})
    id_cols = df_icd.columns.drop('CurrentICD10ListTXT')
    df_icd = df_icd['CurrentICD10ListTXT'].apply(pd.Series). \
        merge(df, left_index=True, right_index=True). \
        drop(['CurrentICD10ListTXT'], axis=1). \
        melt(id_vars=id_cols, value_name='CurrentICD10TXT'). \
        drop('variable', axis=1).dropna(subset=['CurrentICD10TXT'])
    return pd.concat([df_icd, df_empty], ignore_index=True).reset_index(drop=True)


class ExternalIdentity(Enum):
    PMRN = 0
    MGHMRN = 67
    BWHMRN = 69
    EMPI = 140


class Epic:

    def __init__(self,
                 edw_user,
                 edw_password,
                 db,
                 out_dir='.',
                 dataset_name='ehr',
                 date_str=dt.datetime.today().strftime("%m-%d-%Y").replace(" ", "_"),
                 min_age=20,
                 start_date='2001-01-01',
                 end_date=None,
                 pause=5,
                 **kwargs):

        con = pyodbc.connect(f"DSN=phsedw;UID=Partners\\{edw_user};PWD={edw_password}")
        #con = pyodbc.connect(f"DSN=phsedw2;UID=Partners\\{edw_user};PWD={edw_password}")
        self.db = db
        self.con = con
        self.out_dir = force_path(out_dir, require_exists=True)
        self.dataset_name = dataset_name
        if len(dataset_name):
            self.dataset_name += '_'
        self.date_str = date_str
        if min_age is None or min_age < 20:
            print('WARNING: min_age is less than 20. Ensure that requested data is covered by IRB protocol.')
        self.min_age = min_age
        self.demographic_data = pd.DataFrame()
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.pause = pause

    def close(self):
        self.con.close()

    # TODO use chunksize in pd.read_sql_query and iterate over results for very large queries?
    def fetch_iter(self,
                   query_ids,
                   chunk_sizes,
                   start_chunk=None,
                   end_chunk=None,
                   save_format=None,
                   overwrite=False,
                   leave_tqdm=True,
                   *args,
                   **kwargs
                   ):

        # need a subscriptable object for iteration with chunk sizes
        if isinstance(query_ids, (set, typing.MappingView)):
            query_ids = list(query_ids)

        for data_name, chunk_size in chunk_sizes.items():
            # TODO start and end are probably more useful per datatype
            chunk_starts = range(0, len(query_ids), chunk_size)
            # start and end chunk refer to chunk index, not query index
            if start_chunk is None:
                start_chunk = 0
            if end_chunk is None:
                end_chunk = len(chunk_starts) - 1

            # use a named tuple since we are returning a long list of things
            Chunk = namedtuple('Chunk', ['data_name', 'x', 'y', 'step', 'end', 'data'])
            # TODO make tqdm work correctly inside a generator: elapsed time is affected by next fetch call
            with tqdm(chunk_starts, desc=data_name, leave=leave_tqdm) as progress_bar:
                progress_bar.reset()
                progress_bar.refresh()
                # chunks are the steps, e.g. 0 (fetches items 0-100)
                # x is the first item, e.g. 0
                # y is the last item, e.g. 100
                for chunk, x in enumerate(progress_bar):
                    if start_chunk <= chunk <= end_chunk:
                        y = min(x + chunk_size, len(query_ids))
                        id_list = query_ids[x:y]
                        outfile = None
                        if save_format:
                            if save_format == 'parquet' or save_format == 'txt':
                                outfile = self.out_dir / f'{self.dataset_name}{data_name}_' \
                                                         f'{self.date_str}_{x}_{y}.{save_format}'
                            else:
                                raise NotImplementedError
                        if outfile is None or not outfile.exists() or overwrite:
                            df = getattr(self, data_name)(id_list, *args, **kwargs)
                            if save_format == 'parquet':
                                df.to_parquet(outfile)
                            elif save_format == 'txt':
                                df.to_csv(outfile, sep='\t', header=True, index=False)
                            # TODO necessary to pause / better spot to pause?
                            sleep(self.pause)
                            progress_bar.update()
                            progress_bar.refresh()
                            if chunk == end_chunk:
                                progress_bar.close()
                            yield Chunk(data_name, x, y, chunk, end_chunk, df)

    def fetch(self,
              query_ids,
              chunk_sizes,
              start_chunk=None,
              end_chunk=None,
              prefetch_demographics=True,
              save_format=None,
              save_only=False,
              overwrite=False,
              leave_tqdm=True,
              *args,
              **kwargs):
        """
        :param query_ids: list of of ids to fetch
        :param chunk_sizes: dict with requested tables and chunk sizes
        :param start_chunk: chunk to start at (e.g. resuming)
        :param end_chunk: chunk to stop at
        :param prefetch_demographics: fetch demographic information for age filter before doing anything else
        :param save_format: optionally save chunks to disk as they are fetched: 'txt' or 'parquet'
        :param save_only: don't aggregate and return results, only save to disk (make sure to specify save_format)
        :param overwrite: overwrite existing files
        :param leave_tqdm: leave tqdm progress bar after completion
        :return: pandas table with results or dict if multiple tables are fetched
        """
        query_ids = list(query_ids)

        # fetch demographic information for age filtering once at the beginning
        if prefetch_demographics and list(chunk_sizes)[0] is not 'demographics':
            # only retrieve demographics if we are handling PatientIDs
            query_id = query_ids[0]
            if isinstance(query_id, str) and query_id.startswith('Z'):
                self.fetch(query_ids, chunk_sizes={'demographics': 5000}, prefetch_demographics=False, leave_tqdm=False)
            else:
                raise NotImplementedError

        # default is to create pandas df
        all_df = defaultdict(list)
        for chunk in self.fetch_iter(query_ids, chunk_sizes, start_chunk, end_chunk,
                                     save_format, overwrite, leave_tqdm, *args, **kwargs):

            # keep demographic information for age filtering
            if chunk.data_name == 'demographics':
                self.demographic_data = self.demographic_data.append(chunk.data). \
                    drop_duplicates(subset='PatientID'). \
                    reset_index(drop=True)

            if not save_only:
                all_df[chunk.data_name].append(chunk.data)

        # join the fetched dataframes and return
        if not save_only:
            for data_name in all_df:
                all_df[data_name] = pd.concat(all_df[data_name])

            # if dict has only one key, return values directly
            if len(all_df) == 1:
                (k, v), = all_df.items()
                return v
            else:
                return dict(all_df)

    def fetch_all(self, query_ids, start_chunk=None, end_chunk=None, overwrite=None, save_format=None, save_only=False):
        """
        :param query_ids: list of patient ids to fetch default data tables
        :param overwrite: overwrite existing results
        :param start_chunk: chunk to start at (e.g. resuming)
        :param end_chunk: chunk to stop at
        :param save_format: save data with `parquet` and `txt`, default is not to save intermediate files
        :return:
        """
        # TODO the tqdm timings are a bit off, seems like the iterator isn't closed properly

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
            'medications': 1000,
            'labs': 500,
            'vitals': 500,
            #'notes': 100,              # notes take a long time, so retrieve them separately
        }

        return self.fetch(query_ids, chunk_sizes, start_chunk, end_chunk, overwrite, save_format, save_only)

    def _filter_before_after_dts(self, df):
        date_cols = ['ContactDTS', 'ResultDTS', 'RecordedDTS', 'OrderInstantDTS', 'DiagnosisDTS']
        date_cols_df = list(set(df.columns).intersection(set(date_cols)))
        if (len(date_cols_df) == 1) & (any([self.end_date, self.start_date])):
            if self.end_date is None:
                idx = self.start_date <= df[date_cols_df[0]]
            elif self.start_date is None:
                idx = df[date_cols_df[0]] <= self.end_date
            else:
                idx = (self.start_date <= df[date_cols_df[0]]) & (df[date_cols_df[0]] <= self.end_date)
            output = df.loc[idx]
        else:
            output = df
        return output

    def _filter_dates(func):
        @functools.wraps(func)
        def wrap(self, *args, **kwargs):
            df = func(self, *args, **kwargs)
            if isinstance(df, pd.DataFrame) and 'PatientID' in df.columns:
                missing_ids = patient_ids = set(df['PatientID'])
                # retrieve demographic information if missing
                # TODO are there patients without demographic information in EDW?
                #  right now they are silently dropped, which seems safer and should not happen often
                if 'PatientID' in self.demographic_data.columns:
                    missing_ids = patient_ids.difference(self.demographic_data.PatientID)
                if missing_ids:
                    # fetching automatically adds them to self.demographic_data
                    self.fetch(missing_ids,
                               chunk_sizes={'demographics': 5000},
                               prefetch_demographics=False,
                               leave_tqdm=False)

                # add (and later drop) age information used for filtering
                merged_demographics = False
                if 'BirthDTS' not in df.columns:
                    # this drops any patients without demographic information
                    df = df.merge(self.demographic_data[['PatientID', 'BirthDTS', 'Age']], on='PatientID')
                    merged_demographics = True

                # if datatype ends in DTS and is not BirthDTS, then apply age filter
                idx = df.Age >= self.min_age
                for date_col in [col for col in df if col.endswith('DTS') if not col == 'BirthDTS']:
                    # if the date is missing, don't filter it out (often the case for some extra date columns)
                    # alternatively, could only filter based on specific date columns and remove NaT entries
                    idx &= df[date_col].isnull() | ((df[date_col] - df.BirthDTS).dt.days / 365.2425 >= self.min_age)
                # TODO drop age and birth date
                #  it makes the note pulling code more efficient to keep them, since encounters already have it.
                #  but keeps code much more readable to drop them here.
                df = df.loc[idx]
                if merged_demographics:
                    df = df.drop(['Age', 'BirthDTS'], axis=1)
                # only apply date filter if this is a dataframe
                df = self._filter_before_after_dts(df)
            return df
        return wrap

    def patientids_from_external(self, external_patient_list, external_identity=None):
        """
        get epic patient id mappings given external ids.
        :param external_patient_list: list of external patient ids
        :param external_identity: optionally use ehr.ExternalIdentity to specify id type.
        :return:
        """
        query = (
            f"SELECT Epic.Patient.Identity_{self.db}.PatientID, "
            f"PatientIdentityID, IdentityTypeID "
            f"FROM Epic.Patient.Identity_{self.db} "
            f"WHERE Epic.Patient.Identity_{self.db}.PatientIdentityID IN {ids_to_str(external_patient_list)} "
        )
        if external_identity:
            query += f"AND Epic.Patient.Identity_{self.db}.IdentityTypeID = {external_identity.value}"
        mapping_df = pd.read_sql_query(query, self.con, coerce_float=False)
        return self.patientids(mapping_df.PatientID.unique())

    def patientids(self, patient_list, return_query=False):
        """
        get mappings to external ids based on epic patient identifier
        :param patient_list: string of patient ids
        :return:
        """

        if return_query:
            raise NotImplementedError

        # define SQL string for all identity types we want to pull
        id_str = ids_to_str([x.value for x in ExternalIdentity])
        df = pd.read_sql_query(
            f"SELECT Epic.Patient.Identity_{self.db}.PatientID, "
            f"PatientIdentityID, IdentityTypeID "
            f"FROM Epic.Patient.Identity_{self.db} "
            f"WHERE Epic.Patient.Identity_{self.db}.PatientID IN {ids_to_str(patient_list)} "
            f"AND Epic.Patient.Identity_{self.db}.IdentityTypeID IN {id_str}",
            self.con,
            coerce_float=False
        )
        # add descriptions for the external identifiers
        df = df.assign(IdentityDSC=df.IdentityTypeID.apply(lambda x: ExternalIdentity(x).name))
        # pivot to get a wide table
        df_wide = df[['PatientID', 'PatientIdentityID', 'IdentityDSC']].\
            pivot(index='PatientID', columns='IdentityDSC', values='PatientIdentityID').\
            reset_index()
        # remove the index name (IdentityDSC)
        df_wide.columns.name = None
        return df_wide

    def reference_medication(self, return_query=False):
        # TODO this fetches about 150k rows at once
        query = (
            f'SELECT MedicationID, MedicationDSC, '
            f'TherapeuticClassCD, TherapeuticClassDSC, '
            f'PharmaceuticalClassCD, PharmaceuticalClassDSC, PharmaceuticalSubclassCD, PharmaceuticalSubclassDSC, '
            f'SimpleGenericCD, SimpleGenericDSC, GenericNM, GenericProductID, StrengthAMT ' 
            f'FROM Epic.Reference.Medication'
        )

        if return_query:
            return query
        else:
            return pd.read_sql_query(query, self.con, coerce_float=False). \
                drop_duplicates(subset=['MedicationID'])

    #@_filter_dates
    def hematology(self, start_date, n_days=7, return_query=False):
        component_ids = {
            '5200006601': 'HCT',
            '5200006739': 'HGB',
            '5200008673': 'MCV',
            '5200011989': 'RBC',  # TODO ~10% are coded with 5200014701, other codes
            '5210003022': 'WBC',
            '5200012143': 'RDW',  # TODO expanded the panel starting with RDW
            '5200008625': 'MCHC',
            '5200008619': 'MCH',
            '5200010788': 'PLT',
            '5200009906': 'NRBC',
            '5200009325': 'MPV'
        }
        # TODO make this work for lists or single items
        start_date = start_date[0]
        start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d %H:%M:%S')
        end_date = (pd.to_datetime(start_date) +
                    pd.to_timedelta(n_days, unit='D')).strftime('%Y-%m-%d %H:%M:%S')

        query = (
            f"SELECT PatientID, PatientEncounterID, ResultDTS, ComponentID, ResultValueNBR "
            f"FROM Epic.Orders.Result_{self.db} "
            f"WHERE ComponentID IN {ids_to_str(component_ids.keys())} "
            f"AND ResultDateDTS BETWEEN '{start_date}' AND '{end_date}'"
        )
        if return_query:
            return query
        else:
            labs = pd.read_sql_query(query, self.con, coerce_float=False, parse_dates=['ResultDTS'])
            # TODO this drops less than 0.1% of samples, but makes reshaping much more straightforward
            labs = labs.drop_duplicates(subset=['PatientID', 'PatientEncounterID', 'ResultDTS', 'ComponentID'])
            # ensure values are either float or NaN
            labs['ResultValueNBR'] = pd.to_numeric(labs['ResultValueNBR'], errors='coerce')
            labs_wide = labs.set_index(['PatientID', 'PatientEncounterID', 'ResultDTS', 'ComponentID']). \
                unstack(level=-1). \
                reset_index()
            # fix column names
            labs_wide.columns = [str(col) + str(val) for col, val in labs_wide.columns]
            labs_wide = labs_wide.rename(columns={'ResultValueNBR' + k: v for k, v in component_ids.items()})
            return labs_wide

    #@_filter_dates
    def hematology_str(self, start_date, n_days=7, return_query=False):
        component_ids = {
            '5200006601': 'HCT',
            '5200006739': 'HGB',
            '5200008673': 'MCV',
            '5200011989': 'RBC',  # TODO ~10% are coded with 5200014701, other codes
            '5210003022': 'WBC',
            '5200012143': 'RDW',  # TODO expanded the panel starting with RDW
            '5200008625': 'MCHC',
            '5200008619': 'MCH',
            '5200010788': 'PLT',
            '5200009906': 'NRBC',
            '5200009325': 'MPV'
        }
        # TODO make this work for lists or single items
        start_date = start_date[0]
        start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d %H:%M:%S')
        end_date = (pd.to_datetime(start_date) +
                    pd.to_timedelta(n_days, unit='D')).strftime('%Y-%m-%d %H:%M:%S')

        query = (
            f"SELECT PatientID, PatientEncounterID, ResultDTS, ComponentNM, Epic.Orders.Result_{self.db}.ComponentID, ResultValueNBR "
            f"FROM Epic.Orders.Result_{self.db} "
            f"JOIN Epic.Reference.Component "
            f"ON Epic.Reference.Component.ComponentID=Epic.Orders.Result_{self.db}.ComponentID "
            f"WHERE ComponentNM IN {ids_to_str(component_ids.values())} "
            f"AND ResultDateDTS BETWEEN '{start_date}' AND '{end_date}'"
        )
        if return_query:
            return query
        else:
            labs = pd.read_sql_query(query, self.con, coerce_float=False, parse_dates=['ResultDTS'])
            # TODO this drops less than 0.1% of samples, but makes reshaping much more straightforward
            labs = labs.drop_duplicates(subset=['PatientID', 'PatientEncounterID', 'ResultDTS', 'ComponentNM'])
            # ensure values are either float or NaN
            labs['ResultValueNBR'] = pd.to_numeric(labs['ResultValueNBR'], errors='coerce')
            labs_wide = labs.set_index(['PatientID', 'PatientEncounterID', 'ResultDTS', 'ComponentNM']). \
                unstack(level=-1). \
                reset_index()
            # fix column names
            labs_wide.columns = [str(col) + str(val) for col, val in labs_wide.columns]
            labs_wide = labs_wide.rename(columns={'ResultValueNBR' + v: v for k, v in component_ids.items()})
            return labs_wide

    def demographics(self, patient_list, return_query=False):
        query = (
            f"SELECT Epic.Patient.Patient_{self.db}.PatientID, "
            f"SexDSC, PatientRaceDSC, EthnicGroupDSC, PatientStatusDSC, BirthDTS, DeathDTS "
            f"FROM Epic.Patient.Patient_{self.db} "
            f"JOIN Epic.Patient.Race_{self.db} "
            f"ON Epic.Patient.Patient_{self.db}.PatientID=Epic.Patient.Race_{self.db}.PatientID "
            f"WHERE Epic.Patient.Patient_{self.db}.PatientID IN {ids_to_str(patient_list)}"
        )
        if return_query:
            return query
        else:
            df = pd.read_sql_query(query, self.con, coerce_float=False, parse_dates=['BirthDTS', 'DeathDTS']). \
                drop_duplicates(subset='PatientID')
            # now_dt = dt.datetime.now()
            return df.assign(Age=(dt.datetime.now() - df.BirthDTS).dt.days / 365.2425)

    @_filter_dates
    def labs(self, patient_list, return_query=False):
        # TODO Unclear if Female or Male references are used; "Default" section might be unneeded
        query = (
            f"SELECT PatientID, PatientEncounterID, OrderProcedureID, ResultDTS, "
            f"Epic.Reference.Component.ComponentID, ComponentNM, ComponentAbbreviationTXT, ExternalNM, BaseNM, "
            f"ComponentCommonNM, ResultTXT, ResultValueNBR, InReferenceRangeFLG, "
            f"ReferenceRangeLowNBR, ReferenceRangeHighNBR, ReferenceRangeUnitCD, "
            f"Epic.Reference.Component.ComponentTypeDSC, DefaultLowCD, DefaultHighCD, "
            f"DefaultLowFemaleCD, DefaultHighFemaleCD, DefaultUnitCD "
            f"FROM Epic.Orders.Result_{self.db} "
            f"JOIN Epic.Reference.Component "
            f"ON Epic.Reference.Component.ComponentID=Epic.Orders.Result_{self.db}.ComponentID "
            f"WHERE PatientID IN {ids_to_str(patient_list)}"
        )
        if return_query:
            return query
        else:
            # TODO does ComponentID uniquely identify BaseNM?
            # TODO dropping labs with empty ResultTXT
            return pd.read_sql_query(query, self.con, coerce_float=False, parse_dates=['ResultDTS']). \
                dropna(subset=['ResultTXT']). \
                drop_duplicates(subset=["PatientID", "PatientEncounterID", "ResultDTS",
                                        "OrderProcedureID", "ComponentID", "ResultTXT"])

    @_filter_dates
    def vitals(self, patient_list, return_query=False):
        query = (
            f"SELECT PatientID, RecordedDTS, FlowsheetMeasureNM, DisplayNM, "
            f"MeasureTXT, MeasureCommentTXT, "
            f"Epic.Clinical.FlowsheetMeasure_{self.db}.FlowsheetMeasureID "
            f"FROM Epic.Clinical.FlowsheetMeasure_{self.db} "
            f"JOIN Epic.Clinical.FlowsheetRecordLink_{self.db} ON "
            f"Epic.Clinical.FlowsheetRecordLink_{self.db}.FlowsheetDataID=Epic.Clinical.FlowsheetMeasure_{self.db}.FlowsheetDataID "
            f"JOIN Epic.Clinical.FlowsheetGroup_{self.db} ON "
            f"Epic.Clinical.FlowsheetGroup_{self.db}.FlowsheetMeasureID=Epic.Clinical.FlowsheetMeasure_{self.db}.FlowsheetMeasureID "
            f"WHERE Epic.Clinical.FlowsheetMeasure_{self.db}.FlowsheetMeasureID "
            f"IN ('5','6','8','10', '11', '301060','301070') "
            f"AND PatientID IN {ids_to_str(patient_list)}"
        )
        if return_query:
            return query
        else:
            return pd.read_sql_query(query, self.con, coerce_float=False, parse_dates=['RecordedDTS']). \
                drop_duplicates(subset=["PatientID", "RecordedDTS", "FlowsheetMeasureNM"])

    @_filter_dates
    def medications(self, patient_list, return_query=False):
        query = (
            f"SELECT PatientID, PatientEncounterID, OrderID, OriginalMedicationOrderID, "
            f"StartDTS, EndDTS, UpdateDTS, OrderInstantDTS, OrderStartDTS, OrderEndDTS, OrderDiscontinuedDTS, "
            f"MedicationID, MedicationDSC, MedicationDisplayNM, SigTXT, MedicationRouteDSC,"
            f"PrescriptionQuantityNBR, RefillCNT, DiscreteFrequencyDSC, DiscreteDoseAMT, HVDoseUnitDSC,  "
            f"OrderPriorityDSC, OrderStatusDSC, AdditionalInformationOrderStatusDSC, OrderClassDSC, "
            f"ReorderedOrModifiedDSC, MedicationReorderMethodDSC,"
            f"LastAdministeredDoseCommentTXT, MedicationDiscontinueReasonDSC, OrderingModeDSC, "
            f"MedicationPrescribingProviderID, OrderingProviderID, ProviderTypeDSC, ProviderStatusDSC, "
            f"PatientLocationDSC "
            f"FROM Epic.Orders.Medication_{self.db} "
            f"WHERE PatientID IN {ids_to_str(patient_list)}"
        )

        if return_query:
            return query
        else:
            return pd.read_sql_query(query, self.con, coerce_float=False,
                                     parse_dates=['StartDTS', 'EndDTS', 'UpdateDTS', 'OrderInstantDTS', 'OrderStartDTS',
                                                  'OrderEndDTS', 'OrderDiscontinuedDTS']). \
                dropna(subset=['PatientID', 'PatientEncounterID', 'OrderID']). \
                drop_duplicates()

    @_filter_dates
    def tobacco(self, patient_list, return_query=False):
        query = (
            f"SELECT PatientID, ContactDTS, "
            f"TobaccoPacksPerDayCNT, TobaccoUsageYearNBR, TobaccoCommentTXT, TobaccoUserDSC, "
            f"SmokingTobaccoUseDSC, SmokelessTobaccoUserDSC "
            f"FROM Epic.Patient.SocialHistory_{self.db} "
            f"WHERE Epic.Patient.SocialHistory_{self.db}.PatientID IN {ids_to_str(patient_list)}"
        )
        if return_query:
            return query
        else:
            return pd.read_sql_query(query, self.con, coerce_float=False, parse_dates=['ContactDTS']). \
                drop_duplicates(subset=['PatientID', 'ContactDTS'])

    @_filter_dates
    def medicalhx(self, patient_list, return_query=False):

        query = (
            f"SELECT Epic.Patient.MedicalHistory_{self.db}.PatientID, "
            f"Epic.Patient.MedicalHistory_{self.db}.PatientEncounterID, "
            f"Epic.Patient.MedicalHistory_{self.db}.ContactDTS, "
            f"Epic.Reference.DiagnosisCurrentICD10.DiagnosisID, "
            f"Epic.Reference.DiagnosisCurrentICD10.ICD10CD "
            f"FROM Epic.Patient.MedicalHistory_{self.db} "
            f"JOIN Epic.Reference.DiagnosisCurrentICD10 "
            f"ON Epic.Reference.DiagnosisCurrentICD10.DiagnosisID=Epic.Patient.MedicalHistory_{self.db}.DiagnosisID "
            f"WHERE Epic.Patient.MedicalHistory_{self.db}.PatientID IN {ids_to_str(patient_list)}"
        )

        if return_query:
            return query
        else:
            return pd.read_sql_query(query, self.con, coerce_float=False, parse_dates=['ContactDTS']). \
                drop_duplicates(subset=['PatientID', 'PatientEncounterID', 'ICD10CD'])

    # TODO the ContactDTS is NOT the procedure date see issue #29
    @_filter_dates
    def surgicalhx(self, patient_list, return_query=False):
        query = (
            f"SELECT PatientID, PatientEncounterID, Epic.Patient.SurgicalHistory_{self.db}.ProcedureID, "
            f"CommentTXT, Epic.Reference.[Procedure].ProcedureNM, ContactDTS "
            f"FROM Epic.Patient.SurgicalHistory_{self.db} "
            f"JOIN Epic.Reference.[Procedure] "
            f"ON Epic.Reference.[Procedure].ProcedureID= Epic.Patient.SurgicalHistory_{self.db}.ProcedureID "
            f"WHERE Epic.Patient.SurgicalHistory_{self.db}.PatientID IN {ids_to_str(patient_list)}"
        )
        if return_query:
            return query
        else:
            df = pd.read_sql_query(query, self.con, coerce_float=False, parse_dates=['ContactDTS'])
            # replace missing comments with "None" since they are used as primary key
            df["CommentTXT"] = df["CommentTXT"].fillna(value="None")
            return df.drop_duplicates(subset=['PatientID', 'PatientEncounterID', 'ProcedureID', 'CommentTXT'])

    @_filter_dates
    def familyhx(self, patient_list, return_query=False):
        query = (
            f"SELECT PatientEncounterID, PatientID, ContactDTS, MedicalHistoryCD, MedicalHistoryDSC, RelationDSC "
            f"FROM Epic.Patient.FamilyHistory_{self.db} "
            f"WHERE Epic.Patient.FamilyHistory_{self.db}.PatientID IN {ids_to_str(patient_list)}"
        )
        if return_query:
            return query
        else:
            df = pd.read_sql_query(query, self.con, coerce_float=False, parse_dates=['ContactDTS'])
            # drop rows without actual annotation
            df = df.dropna(subset=['MedicalHistoryCD'])
            # replace missing relationship information with 'Unspecified' since they are used as primary key
            df['RelationDSC'] = df['RelationDSC'].fillna(value='Unspecified')
            return df.drop_duplicates(subset=["PatientID", "PatientEncounterID", "MedicalHistoryCD", "RelationDSC"])

    @_filter_dates
    def admitdx(self, patient_list, return_query=False):
        query = (
            f"SELECT Epic.Encounter.PatientEncounter_{self.db}.PatientID, "
            f"Epic.Encounter.PatientEncounter_{self.db}.PatientEncounterID, "
            f"DiagnosisID, AdmitDiagnosisTXT, ContactDTS "
            f"FROM Epic.Encounter.HospitalAdmitDiagnosis_{self.db} "
            f"JOIN Epic.Encounter.PatientEncounter_{self.db} "
            f"ON Epic.Encounter.PatientEncounter_{self.db}.PatientEncounterID="
            f"Epic.Encounter.HospitalAdmitDiagnosis_{self.db}.PatientEncounterID "
            f"WHERE Epic.Encounter.PatientEncounter_{self.db}.PatientID IN {ids_to_str(patient_list)}"
        )
        if return_query:
            return query
        else:
            return pd.read_sql_query(query, self.con, coerce_float=False, parse_dates=['ContactDTS']). \
                dropna(subset=['DiagnosisID']). \
                drop_duplicates(subset=['PatientID', 'PatientEncounterID', 'DiagnosisID'])

    @_filter_dates
    def problemlist(self, patient_list, return_query=False):
        query = (
            f"SELECT PatientID, "
            f"Epic.Reference.DiagnosisCurrentICD10.DiagnosisID, "
            f"ProblemDSC, DiagnosisDTS, ProblemStatusDSC, ICD10CD "
            f"FROM Epic.Patient.ProblemList_{self.db} "
            f"JOIN Epic.Reference.DiagnosisCurrentICD10 "
            f"ON Epic.Reference.DiagnosisCurrentICD10.DiagnosisID=Epic.Patient.ProblemList_{self.db}.DiagnosisID "
            f"WHERE PatientID IN {ids_to_str(patient_list)}"
        )
        if return_query:
            return query
        else:
            return pd.read_sql_query(query, self.con, coerce_float=False, parse_dates=['DiagnosisDTS']). \
                drop_duplicates(subset=["PatientID", "ICD10CD", "DiagnosisDTS"])

    # TODO default is 'string' for notes or 'long' for encounterdx
    #  need to decide how and when to use list columns
    #  i picked safe defaults, but for pandas and parquet we could use the list types directly
    #  the problem is that the fetch iterator could infer that from the result type,
    #  but then it could also be used to fetch with saving and storing in the database.

    @_filter_dates
    def encounterdx(self, id_list, id_type='patient_id', dx_only=True, dx_lists='long', return_query=False):
        """
        :param id_list: list of PatientIDs or PatientEncounterIDs
        :param id_type: 'patient_id' or 'encounter_id'
        :param dx_only: return only rows with DiagnosisID (do not use for notes since it drops ContactDTS)
        :param dx_lists: 'long', 'string' or 'list' for list data like diagnoses and ICD10 codes
        :param return_query: return SQL query instead of result table
        :returns: pd.DataFrame with encounters and diagnosis ids
        """

        # TODO choose whether DISTINCT on the server makes things faster or slower
        #  calculating in python for now, hopefully that keeps the load lower

        query = \
            f"SELECT encounter.PatientEncounterID, " \
            f"encounter.PatientID, " \
            f"encounter.ContactDTS, " \
            f"encounter.EncounterTypeDSC, " \
            f"encounter.DepartmentDSC, " \
            f"DiagnosisNM, CurrentICD10ListTXT, " \
            f"encounterdiagnosis.DiagnosisID, " \
            f"encounterdiagnosis.LineNBR, " \
            f"encounterdiagnosis.UpdateDTS " \
            f"FROM Epic.Encounter.PatientEncounterDiagnosis_{self.db} encounterdiagnosis "
        # Rows with valid DiagnosisID (ICD codes only)
        if dx_only:
            query +=\
                f"JOIN Epic.Encounter.PatientEncounter_{self.db} encounter " \
                f"ON encounter.PatientEncounterID=encounterdiagnosis.PatientEncounterID " \
                f"JOIN Epic.Reference.ICDDiagnosis icddiagnosis " \
                f"ON encounterdiagnosis.DiagnosisID=icddiagnosis.DiagnosisID "
        # All rows
        else:
            query +=\
                f"RIGHT JOIN Epic.Encounter.PatientEncounter_{self.db} encounter " \
                f"ON encounter.PatientEncounterID=encounterdiagnosis.PatientEncounterID " \
                f"LEFT JOIN Epic.Reference.ICDDiagnosis icddiagnosis " \
                f"ON encounterdiagnosis.DiagnosisID=icddiagnosis.DiagnosisID "
        if id_type == 'patient_id':
            query += f"WHERE encounter.PatientID IN {ids_to_str(id_list)}"
        elif id_type == 'encounter_id':
            query += f"WHERE encounter.PatientEncounterID IN {ids_to_str(id_list)}"
        else:
            raise NotImplementedError

        if return_query:
            # TODO need to represent list columns correctly or not create list return values for pqdb
            return query
        else:
            df = pd.read_sql(query, self.con, coerce_float=False, parse_dates=['ContactDTS', 'UpdateDTS']). \
                drop_duplicates(subset=['PatientEncounterID', 'LineNBR'])

            if dx_only:
                df = df.dropna(subset=['CurrentICD10ListTXT'])

            # filter out erroneous encounters
            df = df.loc[~df.EncounterTypeDSC.isin(
                {'Erroneous Encounter', 'Erroneous Telephone Encounter', 'ERRONEOUS ENCOUNTER--DISREGARD'})]

            # Split all strings with multiple ICD10codes into lists
            df['CurrentICD10ListTXT'] = df['CurrentICD10ListTXT']. \
                apply(lambda x: x.split(',') if isinstance(x, str) else None). \
                apply(lambda x: [el.replace(' ', '') for el in x] if isinstance(x, list) else None)
            if dx_lists == 'string':
                df = concat_dx(df, as_string=True)
            elif dx_lists == 'list':
                df = concat_dx(df, as_string=False)
            else:
                df = expand_list_ICD10(df)
            # TODO returning CurrentICD10ListTXT OR CurrentICD10TXT columns - issue #30
            #  this is a problem for downstream methods, so renaming here, but should happen in expand_list_ICD10?
            if "CurrentICD10ListTXT" in df.columns:
                df = df.rename(columns={'CurrentICD10ListTXT': 'CurrentICD10TXT'})
            return df.drop_duplicates(subset=["PatientID", "PatientEncounterID", "CurrentICD10TXT"])

    @_filter_dates
    def notes(self,
              patient_list,
              note_type_list=['Progress Notes', 'Discharge Summaries', 'Discharge Summary', 'H&P'],
              dx_lists='string',
              return_query=False
              ):
        """
        :param patient_list: list of PatientIDs
        :param note_type_list: list of note types, e.g. ['Progress Notes', 'Letter']
        :param return_query: return SQL query instead of result table
        :param dx_lists: 'string' or 'list' for list data like diagnoses and ICD10 codes
        :returns: pd.DataFrame with notes
        """
        if return_query:
            # single large query that does all joins in SQL and no filtering, just to infer schema
            # needs a note_id as argument since it takes too long otherwise
            query = (
                f"SELECT "
                f"Epic.Clinical.Note_{self.db}.PatientID, "
                f"Epic.Clinical.NoteText_{self.db}.NoteCSNID, "
                f"Epic.Clinical.Note_{self.db}.NoteID, "
                f"Epic.Clinical.NoteText_{self.db}.NoteTXT, "
                f"Epic.Clinical.Note_{self.db}.PatientEncounterID, InpatientNoteTypeDSC, LastFiledDTS, CurrentAuthorID, "
                f"Epic.Encounter.PatientEncounter_{self.db}.EncounterTypeDSC, "
                f"Epic.Encounter.PatientEncounter_{self.db}.DepartmentDSC, "
                f"Epic.Encounter.PatientEncounter_{self.db}.ContactDTS, "
                f"Epic.Encounter.PatientEncounter_{self.db}.UpdateDTS, "
                f"Epic.Encounter.PatientEncounterDiagnosis_{self.db}.DiagnosisID, "
                f"Epic.Reference.ICDDiagnosis.CurrentICD10ListTXT, "
                f"Epic.Reference.ICDDiagnosis.DiagnosisNM "
                f"FROM Epic.Clinical.Note_{self.db} "
                f"JOIN Epic.Clinical.NoteText_{self.db} "
                f"ON Epic.Clinical.NoteText_{self.db}.NoteID=Epic.Clinical.Note_{self.db}.NoteID "
                f"JOIN Epic.Encounter.PatientEncounter_{self.db} "
                f"ON Epic.Clinical.Note_{self.db}.PatientEncounterID=Epic.Encounter.PatientEncounter_{self.db}.PatientEncounterID "
                f"JOIN Epic.Encounter.PatientEncounterDiagnosis_{self.db} "
                f"ON Epic.Clinical.Note_{self.db}.PatientEncounterID=Epic.Encounter.PatientEncounterDiagnosis_{self.db}.PatientEncounterID "
                f"JOIN Epic.Reference.ICDDiagnosis "
                f"ON Epic.Encounter.PatientEncounterDiagnosis_{self.db}.DiagnosisID=Epic.Reference.ICDDiagnosis.DiagnosisID "
                # f"WHERE Epic.Clinical.Note_{self.db}.PatientID IN {ids_to_str(patient_list)} "
                f"WHERE Epic.Clinical.Note_{self.db}.NoteID IN {ids_to_str(patient_list)} "
            )
            return query
        else:
            query = (
                f"SELECT Epic.Clinical.Note_{self.db}.NoteID, "
                f"PatientLinkID, PatientEncounterID, InpatientNoteTypeDSC, LastFiledDTS, CurrentAuthorID, "
                f"NoteCSNID, LineNBR, NoteTXT "
                f"FROM Epic.Clinical.Note_{self.db} "
                f"JOIN Epic.Clinical.NoteText_{self.db} "
                f"ON Epic.Clinical.NoteText_{self.db}.NoteID=Epic.Clinical.Note_{self.db}.NoteID "
                f"WHERE PatientLinkID IN {ids_to_str(patient_list)} "
            )

            if isinstance(note_type_list, list) and len(note_type_list) > 0:
                query += f"AND InpatientNoteTypeDSC IN {ids_to_str(note_type_list)}"

            df = pd.read_sql(query, self.con, coerce_float=False, parse_dates=['LastFiledDTS']).drop_duplicates()
            # only fetch encounters if there are any notes
            if df.shape[0] > 0:
                # after concatenating, deduplicate so that there is only one row per note_id
                df_notes = concat_notes(df).drop_duplicates(subset='NoteID')
                encounter_ids = set(df_notes[~df_notes.PatientEncounterID.isnull()].PatientEncounterID.unique())
                df_dx = self.fetch(query_ids=encounter_ids,
                                   id_type='encounter_id',
                                   chunk_sizes={'encounterdx': 5000},
                                   prefetch_demographics=False,
                                   leave_tqdm=False,
                                   dx_only=False,
                                   dx_lists=dx_lists)
                return df_notes.merge(df_dx, on=['PatientID', 'PatientEncounterID'], how='left').drop_duplicates()
            else:
                # return empty dataframe
                return df


def main(edw_user,
         db,
         patient_ids,
         outdir: ("output directory", "option", "o") = '.',
         name: ("name", "option", "n") = 'edw',
         age: ("minimum age", "option", "m") = 20,
         after: ("after date", "option", "a") = '2001-1-1',
         before: ("before date", "option", "b") = None,
         start: ("start chunk", "option", "c") = None,
         end: ("end chunk", "option", "e") = None,
         overwrite: ("overwrite existing", "option", "w") = False,
         save: ("save format", "option", "s") = 'parquet'
         ):

    """Connects to EDW and fetches default data tables for specified patients. Example:
    'edw_pull mj715 phs /mnt/obi0/phi/ehr/note_pull/edw_testdir/test_patientids.txt --outdir .'
    """
    # safely request user password
    from getpass import getpass
    edw_password = getpass('EDW Password:')

    # read in patient ids and fetch default tables
    patient_ids = pd.read_csv(patient_ids, header=None)[0].values
    epic = Epic(edw_user=edw_user, edw_password=edw_password, db=db, out_dir=outdir, dataset_name=name,
                min_age=int(age), start_date=after, end_date=before)
    epic.fetch_all(query_ids=patient_ids, start_chunk=start, end_chunk=end, overwrite=overwrite,
                   save_format=save, save_only=True)
    epic.close()


# TODO console_scripts entry point not working with plac
def run():
    import plac
    plac.call(main)


if __name__ == "__main__":
    run()
