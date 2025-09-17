from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.dialects.postgresql.psycopg2 import PGDialect_psycopg2, PGExecutionContext_psycopg2
from jaydebeapi import connect


"""
Start dev environment:
pyenv virtualenv 3.11 sp311b
pyenv activate sp311b
pip install sqlalchemy jaydebeapi 
pip list | fgrep -i sqlalchemy
# Test with SqlAlchemy 2.0.43 and 1.4.45 
"""

# Disable psycopg2-specific methods to work with JDBC
PGDialect.paramstyle = 'qmark'
PGDialect_psycopg2.on_connect = lambda self: []
PGDialect_psycopg2._hstore_oids = lambda self, dbapi_conn: None
PGExecutionContext_psycopg2.post_exec = lambda self: None

# JDBC connection parameters
jdbc_driver = "org.postgresql.Driver"
jdbc_url = "jdbc:postgresql://127.0.0.1:54321/spinta"
username = "admin"
password = "admin123"
jar_path = "/home/oa/jdbc/postgresql-42.7.7.jar"


# Create SQLAlchemy engine using JDBC
def get_connection():
    conn = connect(jdbc_driver, jdbc_url, [username, password], jars=[jar_path])
    conn.jconn.setAutoCommit(False)
    return conn


engine = create_engine('postgresql://', creator=get_connection)

# SQLAlchemy Base
Base = declarative_base()


class Product(Base):
    __tablename__ = "product"

    id = Column(Integer, primary_key=True)
    name = Column(String)

# Create tables
# Base.metadata.create_all(engine, checkfirst=True)


# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Example usage: Insert data
# new_product = Product(name='Car', id=1)
# session.add(new_product)
# session.commit()

# Query data
products = session.query(Product).all()
for product in products:
    print(f"ID: {product.id}, Name: {product.name}")

# Close session
session.close()
