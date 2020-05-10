from sqlalchemy import *
import urllib


def get_engine(user, password, db):
    # db='PHS'
    # EDW_USER = os.environ['EDW_USER']
    # EDW_PASSWORD = os.environ['EDW_PASSWORD']
    epic_connect_str = f'DRIVER=FreeTDS;' \
                  f'SERVER=phsedw.partners.org;' \
                  f'PORT=1433;' \
                  f'DATABASE=Epic;' \
                  f'UID=Partners\\{user};' \
                  f'PWD={password};' \
                  f'TDS_Version=8.0;'
    epic_params = urllib.parse.quote_plus(epic_connect_str)
    return create_engine('mssql+pyodbc:///?odbc_connect=%s' % epic_params, pool_size = 1, echo = False)

def execute_query(engine, query):
    with engine.connect() as conn:
        result = conn.execute(query)
        columns = result.keys()
        data = result.fetchone()
        metadata = result._metadata
    return data, columns, metadata