import random
from threading import Thread

import astropy.units as u
import numpy as np
import sunpy
from sqlalchemy import func
from sqlalchemy.orm import Session
from sunpy.net import Fido, attrs

from ded.database.config import DEDSession
from ded.database.setup import Map


def loadData(s_map: Map):
    query = Fido.search(attrs.Time(s_map.tstart.isoformat(), s_map.tstart.isoformat()),
                        attrs.Instrument(s_map.instrument), attrs.Wavelength(s_map.wavelength * u.AA))
    query = query[0, 0]
    if not s_map.id.startswith(list(query.responses)[0][0].fileid):
        raise Exception("ERROR: invalid search result encountered!")
    return Fido.fetch(query, progress=True)


def download(event_types, amount=100, flag="TRAIN"):
    session: Session = DEDSession()
    part_size = int(amount / len(event_types))

    maps = []
    for type in event_types:
        result = session.query(Map).order_by(func.random()).filter_by(path=None, flag=flag).join(Map.events).filter_by(
            type=type).all()[:part_size]
        if len(result) < part_size:
            raise Exception("Not enough entries available to fetch!")
        maps.extend(result)

    random.shuffle(maps)
    session.close()

    for chunck in np.split(np.array(maps), amount / 50):
        Thread(target=requestData, args=[[s_map.id for s_map in chunck]]).start()


def requestData(maps):
    session: Session = DEDSession()
    for id in maps:
        try:
            s_map = session.query(Map).filter(Map.id == id).first()
            paths = loadData(s_map)
            if len(paths) == 0:
                continue
                # session.query(Map).filter(Map.id == id).delete()
            else:
                s_map.path = paths[0]
            session.commit()
        except Exception as ex:
            print(ex)
    session.close()


if __name__ == '__main__':
    sunpy.config.set("downloads", "download_dir", "D:\\UNI\\DeepLearning\\SolarData\\DB")

    download(event_types=["FL", "AR", "CH", "ER"], amount=1000, flag="TRAIN")
