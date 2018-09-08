import pickle
import random

import numpy as np
import sunpy.map
from astropy import units as u
from pkg_resources import resource_filename
from sqlalchemy import func, not_
from sqlalchemy.orm import Session
from tensorflow.python.keras.utils import Sequence

from ded.config import resource_dir
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
        feature_batch = np.array(batch[:, 0].tolist(), dtype=np.float16)
        label_batch = np.array(batch[:, 1].tolist(), dtype=np.float16)

        session.close()

        return feature_batch, label_batch

    def _loadDataSet(self, event_type, flag, size):
        session: Session = DEDSession()
        part_size = int(size / 2)

        event_filter = Map.events.any(Event.type == event_type)

        query = session.query(Map).filter(Map.path != None, Map.flag == flag, event_filter).order_by(
            func.random()).with_entities(Map.id)
        result = query.limit(part_size).all()
        event_map_ids = list(map(lambda x: x[0], result))

        query = session.query(Map).filter(Map.path != None, Map.flag == flag, not_(event_filter)).order_by(
            func.random()).with_entities(Map.id)
        result = query.limit(part_size).all()
        none_ids = list(map(lambda x: x[0], result))

        session.close()

        data = [[id, [0, 1]] for id in event_map_ids] + [[id, [1, 0]] for id in none_ids]
        if len(data) < size:
            raise Exception("Not enough entries available!")

        random.shuffle(data)
        return data

    def _loadData(self, map_id, session):
        s_map = session.query(Map).filter(Map.id == map_id).first()
        if s_map.data800:
            return prepareData(pickle.loads(s_map.data800))
        if s_map.path:
            data = sunpy.map.Map(s_map.path).resample((800, 800) * u.pixel).data.tolist()
            s_map.data800 = pickle.dumps(data)
            session.commit()
            return prepareData(data)


def prepareData(data):
    shift = 1
    data = (np.array(data) - shift)
    np.nan_to_num(data, copy=False)
    return np.reshape(data, (800, 800, 1)).tolist()


if __name__ == '__main__':
    gen = DataGenerator(400, 50)
    for i in range(len(gen)):
        data, label = gen[i]
        pickle.dump(data, open(resource_filename(resource_dir, "data_batch{}.pickle".format(i)), "wb"))
        pickle.dump(label, open(resource_filename(resource_dir, "label_batch{}.pickle".format(i)), "wb"))
