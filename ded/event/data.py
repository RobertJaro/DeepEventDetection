import pickle
import random

import numpy as np
import sunpy.map
from astropy import units as u
from sqlalchemy import func
from sqlalchemy.orm import Session
from tensorflow.python.keras.utils import Sequence

from ded.database.config import DEDSession
from ded.database.setup import Map, Event


class DataGenerator(Sequence):

    def __init__(self, size=180, batch_size=32, event_type="FL", flag="TRAIN"):
        data = self._loadDataSet(event_type, flag, size)

        self.batch_size = batch_size
        self.data = data

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        session = DEDSession()

        pos = index * self.batch_size
        batch = np.array(
            [[self._loadData(map_id, session), label] for map_id, label in self.data[pos:pos + self.batch_size]])
        feature_batch = np.array(batch[:, 0].tolist())
        label_batch = np.array(batch[:, 1].tolist())

        session.close()

        return feature_batch, label_batch

    def _loadDataSet(self, event_type, flag, size):
        session: Session = DEDSession()

        event_query = session.query(Map)
        event_query = event_query.filter(Map.path != None, Map.flag == flag)
        event_query = event_query.order_by(func.random())
        event_query = event_query.join(Map.events)
        event_query = event_query.filter(Event.type == event_type)
        event_query = event_query.with_entities(Map.id).distinct(Map.id)
        event_map_ids = event_query.all()
        event_map_ids = list(map(lambda x: x[0], event_map_ids))  # extract results

        none_ids = session.query(Map).filter(Map.path != None, Map.flag == flag,
                                             Map.id.notin_(event_query)).order_by(func.random()).with_entities(
            Map.id).distinct(Map.id).all()
        none_ids = list(map(lambda x: x[0], none_ids))  # extract results
        session.close()

        part_size = int(size / 2)
        data = [[id, 1] for id in event_map_ids[:part_size]] + [[id, 0] for id in none_ids[:part_size]]
        if len(data) < size:
            raise Exception("Not enough entries available!")

        random.shuffle(data)
        return data

    def _loadData(self, map_id, session):
        s_map = session.query(Map).filter(Map.id == map_id).first()
        if s_map.data800:
            return pickle.loads(s_map.data800)
        if s_map.path:
            data = sunpy.map.Map(s_map.path).resample((800, 800) * u.pixel).data.tolist()
            s_map.data800 = pickle.dumps(data)
            session.commit()
            return data
