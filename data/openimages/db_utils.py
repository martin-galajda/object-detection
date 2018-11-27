import sqlite3
import os

def init(db_file_path):
  conn = sqlite3.connect(db_file_path)
  return conn

def load_sql_schema_def(path_to_schema_def = './db-schema.sql'):
  sql_str = ""
  with open(path_to_schema_def, 'r') as file:
    sql_str += os.linesep.join(file.readlines())
  return sql_str

def setup_db(conn, schema_sql):
  conn.executescript(schema_sql)
