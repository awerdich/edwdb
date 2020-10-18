import urllib
import io
import csv
from sqlalchemy import create_engine, cast
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.dialects.postgresql import JSONB
import numpy as np
import pandas as pd
from random import choice
import string
from .ehr_db_model import *
from .edw import ids_to_str


def _psql_upsert(table, conn, keys, data_iter):
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-sql-method
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = io.StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        # need to choose between keeping existing values or updating with replacement
        tmp_table_name = 'tmp_' + ''.join(choice(string.ascii_uppercase) for _ in range(5))
        # make an empty copy of the target table, but include the constraints
        cur.execute(f'CREATE TABLE {tmp_table_name} ( like {table_name} including all)')
        cur.copy_expert(sql=f'COPY {tmp_table_name} ({columns}) FROM STDIN WITH CSV', file=s_buf)

        # TODO always keeping existing data
        cur.execute(f'INSERT INTO {table_name} SELECT * FROM {tmp_table_name} ON CONFLICT DO NOTHING')

        # if keep_existing:
        #     cur.execute(f'INSERT INTO {table_name} SELECT * FROM {tmp_table_name} ON CONFLICT DO NOTHING')
        # else:
        #     raise NotImplementedError
        #     # TODO this requires specifying violated constraints
        #     # cur.execute(f'INSERT INTO {table_name} SELECT * FROM {tmp_table_name} '
        #     #             f'ON CONFLICT ON CONSTRAINT DO UPDATE')
        cur.execute(f'DROP TABLE {tmp_table_name}')


class EhrDb:

    def __init__(self,
                 user,
                 password,
                 host='obi-cpu8',
                 port='5432',
                 db='ehr_dev'):

        quoted_password = urllib.parse.quote_plus(password)
        db_url = f'postgresql+psycopg2://{user}:{quoted_password}@{host}:{port}/{db}'
        # TODO escape the connection string so all passwords will work
        # db_url = 'postgresql+psycopg2://' + \
        #          urllib.parse.quote_plus(f'{user}:{password}@{host}:{port}/{db}')

        self.user = user
        self.engine = create_engine(db_url, pool_size=1, echo=False)
        self.session = scoped_session(sessionmaker(bind=self.engine))

    def _import_iter(self, import_, edw_iter, keep_existing=True):
        if not keep_existing:
            raise NotImplementedError
        for chunk in edw_iter:
            # refresh the import_session object from db
            self.session.refresh(import_)
            # add import id as column so we can dump the dataframe into the database
            chunk.data['ImportID'] = import_.ImportID
            # this adds data efficiently to the database
            chunk.data.to_sql(chunk.data_name, self.engine, index=False, method=_psql_upsert, if_exists='append')
            # update the import progress
            progress = import_.Progress
            if chunk.data_name not in progress:
                progress[chunk.data_name] = {}
            progress[chunk.data_name][chunk.step] = {'x': chunk.x, 'y': chunk.y}
            import_.Progress = cast(progress, JSONB)  # automatic casting for JSONB isn't working
            # wrap up import session
            self.session.add(import_)
            self.session.commit()
        # return completion state so it can be updated in caller
        # also returns true if there is nothing to iterate over
        return True

    def _load_demographics(self, epic, query_ids=None):
        print('Loading demographic information.')
        query = 'SELECT * FROM demographics'
        if query_ids is not None:
            query += f' WHERE "PatientID" IN {ids_to_str(query_ids)}'
        epic.demographic_data = pd.read_sql(query, self.engine, parse_dates=['BirthDTS', 'DeathDTS'])

    def import_epic(self,
                    name,
                    description,
                    protocol,
                    query_ids,
                    chunk_sizes,
                    epic,
                    keep_existing=True,
                    load_demographics=True
                    ):
        # define a new import
        import_ = Import(
            User=self.user,
            Name=name,
            Description=description,
            Protocol=protocol,
            StartDTS=epic.start_date,
            EndDTS=epic.end_date,
            Status='incomplete',
            ChunkSizes=cast(chunk_sizes, JSONB),
        )
        # store it in the database and refresh to get an import id
        self.session.add(import_)
        self.session.commit()
        self.session.refresh(import_)
        # also store the query
        self.session.bulk_save_objects(
            [Query(QueryID=query_id, ImportID=import_.ImportID) for query_id in query_ids]
        )
        self.session.commit()

        if load_demographics:
            self._load_demographics(epic, query_ids)
        print(f'Started import {import_.ImportID}.')
        for data_name, chunk_size in chunk_sizes.items():
            epic_iter = epic.fetch_iter(query_ids=query_ids, chunk_sizes={data_name: chunk_size})
            completed = self._import_iter(import_, epic_iter, keep_existing=keep_existing)
        if completed:
            import_.Status = 'complete'
            self.session.add(import_)
            self.session.commit()
            print(f'Finished import {import_.ImportID}.')

    def resume_import(self, import_id, epic, keep_existing=True, load_demographics=True):
        # get the Import object from db
        import_ = self.session.query(Import).filter_by(ImportID=import_id).all()[0]
        if import_.Status == 'incomplete':
            print(f'Resuming import {import_.ImportID}.')
            query_ids = pd.read_sql(f'SELECT "QueryID" FROM queries WHERE "ImportID" = {import_id};',
                                    self.engine).QueryID.values
            if load_demographics:
                self._load_demographics(epic, query_ids)
            for data_name, chunk_size in import_.ChunkSizes.items():
                # get the last completed chunk for this datatype and then resume
                start_chunk = 0
                if data_name in import_.Progress:
                    # TODO this does not repeat the last chunk
                    #  but might be safer in case something went wrong on import?
                    start_chunk = max([int(i) for i in import_.Progress[data_name].keys()]) + 1
                # if already completed don't do anything
                completed = start_chunk == np.ceil(len(query_ids) / chunk_size)
                if not completed:
                    epic_iter = epic.fetch_iter(query_ids=query_ids,
                                                start_chunk=start_chunk,
                                                chunk_sizes={data_name: chunk_size})
                    completed = self._import_iter(import_, epic_iter, keep_existing)
            if completed:
                import_.Status = 'complete'
                self.session.add(import_)
                self.session.commit()
                print(f'Finished import {import_.ImportID}.')

    def import_pandas(self,
                      name,
                      description,
                      protocol,
                      import_df,
                      table_name,
                      keep_existing=True,
                      ):
        # define a new import
        import_ = Import(
            User=self.user,
            Name=name,
            Description=description,
            Protocol=protocol,
            Status='incomplete',
        )
        # store it in the database and refresh to get an import id
        self.session.add(import_)
        self.session.commit()
        self.session.refresh(import_)

        print(f'Started import {import_.ImportID}.')
        # add import id as column so we can dump the dataframe into the database
        import_df['ImportID'] = import_.ImportID
        # this adds data efficiently to the database
        import_df.to_sql(table_name, self.engine, index=False, method=_psql_upsert, if_exists='append')

        print(f'Finished import {import_.ImportID}.')
        import_.Status = 'complete'
        self.session.add(import_)
        self.session.commit()

    def close(self):
        self.session.close()
        self.engine.dispose()
