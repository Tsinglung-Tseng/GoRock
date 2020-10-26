from collections import defaultdict
from ..database import Database
from psycopg2.errors import UniqueViolation
import json


class FuncList:
    def __init__(self, ldata):
        self.ldata = ldata

    def to_list(self):
        return list(self.ldata)

    def map(self, func):
        return FuncList(map(func, self.ldata))

    def filter(self, func):
        return FuncList(filter(func, self.ldata))


class MAC:
    """
    >>> MAC(geometry_mac).to_pair()
    >>> [('/gate/world/daughters/name', 'OpticalSystem'),
         ('/gate/world/daughters/insert', 'box'),
         ...
    """

    def __init__(self, raw_mac, name):
        self.raw_mac = raw_mac
        self.name = name

    def __eq__(self, other):
        """
        >>> m_file = MAC.from_file('/home/zengqinglong/optical_simu/5/jiqun_20mmCrystalWidth/macro/Geometry.mac')
        >>> m_db = MAC.from_database(1)

        >>> m_db == m_file
        True
        """
        return self.to_json() == other.to_json()

    @staticmethod
    def from_file(path):
        return MAC("".join(open(path).readlines()), path.split("/")[-1])

    @staticmethod
    def from_database(mac_id):
        with Database().cursor() as (conn, cur):
            cur.execute("""SELECT * FROM mac WHERE id = %s;""", (str(mac_id),))
            name, mac_json = cur.fetchone()[1:]

        result = []

        def get_nested_keys(j, outer_k=""):
            for key in j.keys():
                if isinstance(j[key], str) or j[key] == "":
                    result.append(("/".join([outer_k, key]), j[key]))
                else:
                    get_nested_keys(j[key], "/".join([outer_k, key]))

        get_nested_keys(mac_json)
        return MAC(
            "\n".join(FuncList(result).map(lambda row: "    ".join(row)).to_list()),
            name,
        )

    def to_json(self):
        def mac_row2json(mac_pair):
            def pair_to_dict(key, value):
                tmp = {}
                tmp[key] = value
                return tmp

            keys = mac_pair[0].split("/")[1:]
            keys.reverse()
            value = mac_pair[1]

            for i in keys:
                value = pair_to_dict(i, value)
            return value

        def parallel_json2nested_json(j):
            result = {}
            result = defaultdict(lambda: [], result)

            for i in j:
                if isinstance(i, str):
                    return i
                else:
                    jkey = list(i.keys())[0]
                    jvalue = list(i.values())[0]

                result[jkey].append(jvalue)

            for k, v in result.items():
                result[k] = parallel_json2nested_json(v)
            return dict(result)

        return parallel_json2nested_json(
            FuncList(self.to_pair().map(mac_row2json).to_list()).to_list()
        )

    def to_pair(self):
        return (
            FuncList(self.raw_mac.split("\n"))
            .filter(lambda i: len(i) != 0)
            .filter(lambda i: i[0] == "/")
            .map(lambda i: i.split())
            .map(lambda i: (i[0], " ".join(i[1:])))
        )

    def dump(self):
        return "\n".join(
            self.to_pair().map(lambda r: "{:<60s}{:<4s}".format(r[0], r[1])).to_list()
        )

    def commit(self):
        try:
            with Database().cursor() as (conn, cur):
                cur.execute(
                    """INSERT INTO mac ("name", "config") VALUES (%s,%s) returning id;""",
                    (self.name, json.dumps(self.to_json())),
                )
                result = cur.fetchall()
                conn.commit()
            return result[0][0]
        except UniqueViolation as e:
            with Database().cursor() as (conn, cur):
                cur.execute(
                    """select id from mac where config=%s;""",
                    (json.dumps(self.to_json()),),
                )
                result = cur.fetchall()
            return result[0][0]
