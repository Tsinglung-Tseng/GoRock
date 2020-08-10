from contextlib import contextmanager

import psycopg2
from psycopg2.extras import DictCursor
from psycopg2.pool import SimpleConnectionPool


class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls()
        return self._instance[self._cls]


# @Singleton
class PG:
    """
    pg = PG()
    pg.initialize()
    conn = pg.get_connection()
    """

    def __init__(self):
        self.__pool = None

    def initialize(self, **kw):
        if self.__pool is None:
            self.__pool = SimpleConnectionPool(
                host=kw.get("host", "192.168.1.170"),
                port=kw.get("port", 5432),
                user=kw.get("user", "tsinglung"),
                password=kw.get("password", "511kev"),
                dbname=kw.get("dbname", "dlsr"),
                maxconn=kw.get("maxsize", 20),
                minconn=kw.get("minsize", 1),
            )

    def get_connection(self):
        return self.__pool.getconn()

    def return_connection(self, connection):
        return self.__pool.putconn(connection)

    def close_all_connection(self):
        self.__pool.closeall()


@contextmanager
def pg_connection():
    pg = PG()
    pg.initialize()
    conn = pg.get_connection()
    try:
        yield conn
    finally:
        conn.commit()
        pg.putconn(conn)


@contextmanager
def server_side_cursor(cursor_name="cursor_unique_name"):
    """
    usage:
    with server_side_cursor() as cur:
        cur.execute('SELECT 100')
            result = cur.fetchall()
    """
    pg = PG()
    pg.initialize()
    conn = pg.get_connection()
    cursor = conn.cursor(cursor_name, cursor_factory=psycopg2.extras.DictCursor)
    try:
        yield cursor
    finally:
        conn.commit()
        pg.return_connection(conn)


@contextmanager
def simple_cursor():
    """
    usage:
    with simple_cursor() as cur:
        cur.execute('SELECT 100')
            result = cur.fetchall()
    """
    pg = PG()
    pg.initialize()
    conn = pg.get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        yield cursor
    finally:
        conn.commit()
        pg.return_connection(conn)
