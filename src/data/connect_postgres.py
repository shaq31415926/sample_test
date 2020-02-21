import psycopg2
from src.config import host, port, dbname, user


def connect_postgres():
    """ Connect to the PostgreSQL database server """
    conn = psycopg2.connect(host=host, database=dbname,
                            user=user, port=port)

    return conn

if __name__ == '__main__':
    connect_postgres()



