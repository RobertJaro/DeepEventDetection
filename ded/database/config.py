import pkg_resources
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

from ded.config import resource_dir

Base = declarative_base()
engine = create_engine('sqlite:///' + pkg_resources.resource_filename(resource_dir, "ded.db"), echo=False)
session_factory = sessionmaker(bind=engine)
DEDSession = scoped_session(session_factory)
