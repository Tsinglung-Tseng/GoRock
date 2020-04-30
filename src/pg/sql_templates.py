import psycopg2
from .pool import server_side_cursor, simple_cursor
from ..registry import ConfigType


class Template:
    @staticmethod
    def insert_config(config_type, config):
        return f"""insert INTO {config_type}_config (config) VALUES ({config}) returning id; """

    @staticmethod
    def lookup_session_config_by_config(config):
        return f"""SELECT id FROM tf_session_config WHERE config='{{config}}';"""

    # @staticmethod
    # def insert_dataset_config(config):
    # return f"""insert INTO dataset_config (config) VALUES ({config}) returning id; """

    # @staticmethod
    # def insert_model_config(config):
    # return f"""insert INTO model_config (config) VALUES ({config}) returning id; """


class SQLRunner:
    @staticmethod
    def insert_config_if_not_exist_routine(config_type, config_content):
        try:
            with simple_cursor() as cur:
                cur.execute(Template.insert_config(config_type, config_content))
                result = cur.fetchall()
        except psycopg2.Error as e:
            with server_side_cursor() as cur:
                cur.execute(Template.lookup_session_config_by_config(config_content))
                result = cur.fetchall()
        return result
