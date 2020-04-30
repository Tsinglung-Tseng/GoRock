import json
import psycopg2
from .pool import server_side_cursor, simple_cursor
from ..registry import ConfigType


class Template:
    @staticmethod
    def insert_config(config_type, config):
        return f"""insert INTO {config_type}_config (config) VALUES ('{config}') returning id;"""

    @staticmethod
    def lookup_session_config_by_config(config_type, config):
        return f"""SELECT id FROM {config_type}_config WHERE config='{config}';"""


class SQLRunner:
    @staticmethod
    def insert_config_if_not_exist(config_type, config_content):
        config_content = json.dumps(config_content)
        try:
            with server_side_cursor() as cur:
                cur.execute(Template.lookup_session_config_by_config(config_type, config_content))
                result = cur.fetchone()
        except psycopg2.Error as e:
            with simple_cursor() as cur:
                cur.execute(Template.insert_config(config_type, config_content))
                result = cur.fetchone()
        return result['id']

