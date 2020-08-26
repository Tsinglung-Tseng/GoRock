import sqlalchemy
import dotenv
import os
import pandas
import psycopg2
import psycopg2.extras
import contextlib
from dotenv import load_dotenv, find_dotenv


psycopg2.extras.register_uuid()
dotenv.load_dotenv(override=True)


class Database:
    def __init__(self):
        self.db_connection = os.getenv("DB_CONNECTION")

    def read_sql(self, sql, params):
        return pandas.read_sql(sql, self.db_connection, params=params)

    @contextlib.contextmanager
    def cursor(self):
        conn = psycopg2.connect(self.db_connection)
        cursor = conn.cursor()
        yield (conn, cursor)
        cursor.close()
        conn.close()
