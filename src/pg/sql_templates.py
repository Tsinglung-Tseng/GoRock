import json
import psycopg2
from .pool import server_side_cursor, simple_cursor
from ..registry import ConfigType, TableName


class Template:
    @staticmethod
    def insert_config(config_type, config):
        return f"""insert INTO {config_type}_config (config) VALUES ('{config}') returning id;"""

    @staticmethod
    def lookup_session_config_by_config(config_type, config):
        return f"""SELECT id FROM {config_type}_config WHERE config='{config}';"""

    @staticmethod
    def insert_using_mapping(table_name, value_mapping):
        return f"""INSERT INTO {table_name} ({', '.join(value_mapping.keys())}) VALUES ({", ".join(map(str, value_mapping.values()))}) RETURNING *;"""

    @staticmethod
    def select_on_condition(table_name, condition_value_mapping):
        return f"""SELECT * FROM {table_name} WHERE {" AND ".join([k+'='+str(v) for k, v in condition_value_mapping.items()])};"""


class SQLRunner:
    @staticmethod
    def insert_config_if_not_exist(config_type, config_content):
        config_content = json.dumps(config_content)
        try:
            with server_side_cursor() as cur:
                cur.execute(
                    Template.lookup_session_config_by_config(
                        config_type, config_content
                    )
                )
                result = cur.fetchone()
            if result is None:
                with simple_cursor() as cur:
                    cur.execute(Template.insert_config(config_type, config_content))
                    result = cur.fetchone()
        except psycopg2.Error as e:
            with simple_cursor() as cur:
                cur.execute(Template.insert_config(config_type, config_content))
                result = cur.fetchone()
        return result["id"]

    @staticmethod
    def insert_sessoion_log(logger):
        with simple_cursor() as cur:
            cur.execute(Template.insert_using_mapping(TableName.SESSIONLOG, logger.session_process))
            result = cur.fetchone()
        return result  # ['time_stamp']

    @staticmethod
    def create_or_add_cascade_sessoion(logger):
        # with simple_cursor() as cur:
            # cur.execute(Template.select_on_condition(TableName.SESSION, logger.session_reference))
            # result = cur.fetchall()
        # return [dict(row) for row in result][-1]['id']
        with simple_cursor() as cur:
            cur.execute(Template.insert_using_mapping(TableName.SESSION, logger.session_reference))
            result = cur.fetchone()

        # if result is None:
            # with simple_cursor() as cur:
                # cur.execute(Template.)

        return dict(result)['id']

