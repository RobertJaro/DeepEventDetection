from sqlalchemy import Column, String, Float, Boolean, Table, ForeignKey, DateTime, \
    LargeBinary
from sqlalchemy.orm import relationship, deferred

from ded.database.config import Base, engine

association_table = Table('association', Base.metadata,
                          Column('map_id', String(60), ForeignKey('map.id')),
                          Column('event_id', String(60), ForeignKey('event.id')))


class Event(Base):
    __tablename__ = "event"

    id = Column(String(60), primary_key=True)
    type = Column(String)
    hpc_x = Column(Float)
    hpc_y = Column(Float)
    human = Column(Boolean)
    hpc_bound_x = Column(Float)
    hpc_bound_y = Column(Float)
    tstart = Column(DateTime)
    tend = Column(DateTime)

    def __init__(self, id, type, hpc_x, hpc_y, human, hpc_bound_x, hpc_bound_y, start, end):
        self.id = id
        self.type = type
        self.hpc_x = hpc_x
        self.hpc_y = hpc_y
        self.human = human
        self.hpc_bound_x = hpc_bound_x
        self.hpc_bound_y = hpc_bound_y
        self.tstart = start
        self.tend = end


class Map(Base):
    __tablename__ = "map"

    id = Column(String(60), primary_key=True)
    tstart = Column(DateTime)
    tend = Column(DateTime)
    instrument = Column(String)
    wavelength = Column(Float)
    events = relationship("Event", secondary=association_table)
    path = Column(String)
    data800 = deferred(Column(LargeBinary))
    flag = Column(String, default="TRAIN")


if __name__ == '__main__':
    Base.metadata.create_all(engine)
