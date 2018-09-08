import astropy.units as u
import sunpy
from dateutil import parser
from sqlalchemy import or_, and_, select
from sqlalchemy.orm import Session
from sunpy.net import hek, Fido, attrs

from ded.config import instrument, wavelength
from ded.database.config import DEDSession
from ded.database.setup import Event, Map, association_table

client = hek.HEKClient()


def loadEvents(tstart, tend, types, session):
    hek_result = []
    for type in types:
        res = client.search(hek.attrs.Time(tstart, tend), hek.attrs.EventType(type))
        res = [entry for entry in res if
               entry['obs_instrument'] == instrument and entry["obs_meanwavel"] == wavelength * 1e-8]
        if len(res) == 0:
            raise Exception("No Data found for classification type " + type)
        hek_result.extend(res)

    for entry in hek_result:
        event = Event(entry["kb_archivid"],
                      entry["event_type"],
                      entry["hpc_x"],
                      entry["hpc_y"],
                      entry["frm_humanflag"] == "true",
                      0,
                      0,
                      parser.parse(entry["event_starttime"]),
                      parser.parse(entry["event_endtime"]))
        session.merge(event)
    session.commit()


def loadMaps(tstart, tend, session):
    vso_results = Fido.search(attrs.Time(start=tstart, end=tend),
                              attrs.Wavelength(wavemin=wavelength * u.AA),
                              attrs.Instrument(instrument))
    for i in range(vso_results.file_num):
        query = vso_results[0, i]
        entry = list(query.responses)[0][0]
        entry.time.end = entry.time.end.replace("240000", "000000") # workaround
        s_map = Map(id=entry.fileid,
                    tstart=parser.parse(entry.time.start),
                    tend=parser.parse(entry.time.end),
                    instrument=instrument,
                    wavelength=wavelength)
        session.merge(s_map)
    session.commit()


def mapEvents(session):
    sel_query = select([Map.id, Event.id]).where(or_(
        and_(Event.tstart < Map.tstart, Map.tstart < Event.tend),
        and_(Event.tstart < Map.tend, Map.tend < Event.tend),
        and_(Map.tstart < Event.tstart, Event.tstart < Map.tend)))
    in_query = association_table.insert().from_select(["map_id", "event_id"], sel_query)

    session.execute(in_query)
    session.commit()


if __name__ == '__main__':
    session: Session = DEDSession()

    tstart = '2018/04/01 00:00:00'
    tend = '2018/06/01 00:00:00'

    loadEvents(tstart, tend, ["FL"], session)#, "AR", "CH", "ER"
    loadMaps(tstart, tend, session)
    mapEvents(session)

    session.close()
