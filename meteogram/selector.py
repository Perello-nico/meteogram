# %%
from __future__ import annotations

import xarray as xr
import pandas as pd
import numpy as np
import pytz
from pyproj import CRS
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from shapely.geometry import Point
from datetime import datetime
from pyproj import Transformer


# %% #####################################################################
# OBJECTS ################################################################
##########################################################################

class Event():
    """
    Store an event that is collection of time range and geometry
    in standard reference system (WGS84 - EPSG:4326) and UTC timezone.
    """

    def __init__(
        self,
        time_from: datetime | str,
        time_to: datetime | str,
        geometry: Point,
        epsg: int = 4326
    ):

        # convert to datetime if string
        if isinstance(time_from, str):
            self.time_from = datetime.strptime(time_from, '%Y%m%d%H%M')
        else:
            self.time_from = time_from
        if isinstance(time_to, str):
            self.time_to = datetime.strptime(time_to, '%Y%m%d%H%M')
        else:
            self.time_to = time_to

        # convert to the reference tz
        if self.time_from.tzinfo is None:
            self.time_from = self.time_from.replace(tzinfo=pytz.UTC)
        else:
            self.time_from = self.time_from.astimezone(pytz.utc)
    
        if self.time_to.tzinfo is None:
            self.time_to = self.time_to.replace(tzinfo=pytz.UTC)
        else:    
            self.time_to = self.time_to.astimezone(pytz.utc)

        if self.time_from > self.time_to:
            raise ValueError('time_from must be before time_to')

        # convert geometry to EPSG 4326
        if epsg != 4326:
            self.geometry = convert_geometry(geometry, epsg, 4326)
        else:
            self.geometry = geometry


@dataclass
class EventsCollection():
    collection: Dict[int | str, Event]
    n_events: int = 0
    id_list: List[int | str] = field(default_factory=list)
    key_id: str = 'id'

    def __post_init__(self):
        # get number of elements in collection
        self.n_events = len(self.collection)
        # get list of keys
        self.id_list = list(self.collection.keys())

    @classmethod
    def from_csv(
        cls,
        path: str,
        time_from: Optional[datetime | str] = None,
        time_to: Optional[datetime | str] = None,
        key_id: str = 'id',
        key_xcoord: str = 'longitude',
        key_ycoord: str = 'latitude',
        key_time_from: str = 'time_from',
        key_time_to: str = 'time_to',
        epsg: int = 4326
    ) -> EventsCollection:
        # open dataset
        df = pd.read_csv(path)
        # check if latitue and longitude columns are present
        if key_id not in df.columns or key_xcoord not in df.columns or key_ycoord not in df.columns:
            raise ValueError(f'Columns {key_id}, {key_xcoord} and {key_ycoord} must be present in the dataset')
        # check uniqueness of id
        if len(df[key_id].unique()) != len(df):
            raise ValueError('id field must be unique') 
        # add time information
        if time_from is None:
            # check if time_from column are present
            if key_time_from not in df.columns:
                raise ValueError(f'If time_from is not provided, column {key_time_from} must be present in the dataset')
        else:
            # since I provide time information, I force it to the dataset
            df[key_time_from] = time_from
        if time_to is None:
            # check if time_to column are present
            if key_time_to not in df.columns:
                raise ValueError(f'If time_to is not provided, column {key_time_to} must be present in the dataset')
        else:
            # since I provide time information, I force it to the dataset
            df[key_time_to] = time_to
        # create the collection of events
        collection = dict()
        for _, row in df.iterrows():
            id = row[key_id]
            x = row[key_xcoord]
            y = row[key_ycoord]
            # create Point geometry
            geom = Point(x, y)
            collection[id] = Event(
                time_from=row[key_time_from],
                time_to=row[key_time_to],
                geometry=geom,
                epsg=epsg
            )
        return cls(collection=collection, key_id=key_id)


# %% #################################################################
# FUNCTIONS
######################################################################

def convert_geometry(
    geometry: Point,
    epsg_in: int,
    epsg_out: int
) -> Point:
    """Convert geometry to specific EPSG"""
    crs_in = CRS.from_epsg(epsg_in)
    crs_out = CRS.from_epsg(epsg_out)
    transformer = Transformer.from_crs(crs_in, crs_out, always_xy=True)
    x_out, y_out = transformer.transform(geometry.x, geometry.y)
    return Point(x_out, y_out)


def distance(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float
) -> np.ndarray:
    "Distances in km between lat/lon -> HAVERSINE FORMULA"
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat/2.0)**2) + \
        np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2.0)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def select_data_event(
    dataset: xr.Dataset,
    event: Event,
    radius_km: float = -1
) -> pd.DataFrame:
    """Data selection."""
    # Time selection
    time_from_select = event.time_from.replace(tzinfo=None)
    time_to_select = event.time_to.replace(tzinfo=None)
    times_ok = dataset.time.sel(time=slice(time_from_select,
                                           time_to_select)).values
    times_idx = [list(dataset.time.values).index(tt) for tt in times_ok]
    # Space selection - create a grid of latitude and longitude
    latitudes = dataset.latitude.values
    longitudes = dataset.longitude.values
    if len(latitudes.shape) == 1:
        latitudes, longitudes = np.meshgrid(latitudes, longitudes,
                                            indexing='ij')
    # Calculate distances between event centroid and all points in the grid
    dist_km = distance(longitudes, latitudes,
                       event.geometry.x, event.geometry.y)
    if radius_km == -1:  # METHOD: nearest
        radius_km = np.min(dist_km)
    # Find points within the specified max_distance_km
    mask = np.where(dist_km <= radius_km, True, False)
    if mask.sum() == 0:
        raise ValueError(f"No points found.")
    # Gather the corresponding data for these points
    variables = [var for var in dataset.data_vars]
    shape_out = dataset[variables[0]].values[:, mask][times_idx].shape
    # selection
    times = np.repeat(times_ok, shape_out[1])
    lats = np.tile(latitudes[mask], shape_out[0])
    lons = np.tile(longitudes[mask], shape_out[0])
    dists = np.tile(dist_km[mask], shape_out[0])
    result = pd.DataFrame({
        'time': times,
        'latitude': lats,
        'longitude': lons,
        'distance_km': dists
    })
    for var in variables:
        values = dataset[var].values
        vals = values[:, mask][times_idx]
        vals = vals.flatten()
        result[var] = vals
    return result


def select_data(
    dataset: xr.Dataset,
    events: EventsCollection,
    radius_km: float = -1
) -> pd.DataFrame:
    # search for data
    df_result = pd.DataFrame()
    for id, event in events.collection.items():
        df_tmp = select_data_event(dataset, event, radius_km)
        df_tmp.insert(0, events.key_id, id)
        df_result = pd.concat([df_result, df_tmp],
                              ignore_index=True)
    return df_result



# %%
