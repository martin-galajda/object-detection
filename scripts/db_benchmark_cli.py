import argparse
from data.openimages.constants import BoxableImagesConstants
import time
import sqlite3

def timerfunc(func):
    """
    A timer decorator
    """

    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "The runtime for {func} took {time} seconds to complete"
        print(msg.format(func=func.__name__,
                         time=runtime))
        return value

    return function_timer


def read_sql():
    sql = ''

    read_input = input('Enter your test SQL statement:. End with empty line"\n')

    while read_input.strip() != '':
        sql += '\n' + read_input
        read_input = input()

    return sql

@timerfunc
def execute_sql(db_path: str, sql: str):
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()

    cursor.execute(sql)
    print(f'result = {cursor.fetchone()}')

def main(args: argparse.Namespace):
    sql = read_sql()

    execute_sql(args.path_to_db, sql)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Simple CLI to test times of SQL queries.")

    parser.add_argument('--path_to_db',
                        default=BoxableImagesConstants.PATH_TO_DB_YOLO_V2,
                        type=str,
                        required=False)

    main(parser.parse_args())
