from sqlalchemy.engine import create_engine, URL
from sqlalchemy.orm import sessionmaker, DeclarativeBase

import config as cfg

url = URL.create(
    drivername="postgresql+psycopg2",
    username=cfg.POSTGRES_USERNAME,
    password=cfg.POSTGRES_PASSWORD,
    host=cfg.POSTGRES_HOST,
    port=cfg.POSTGRES_PORT,
    database=cfg.POSTGRES_DATABASE,
)

engine = create_engine(url, pool_size=20, max_overflow=0)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

class Base(DeclarativeBase):
    """
    Base class for all models
    """
    pass

def get_db():
    """
    Get a database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
