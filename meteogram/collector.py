# %%
from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from typing import Any, List, Optional, Dict
import logging
from datetime import datetime, timedelta
import pytz
import numpy as np
import pandas as pd
import xarray as xr
from drops2 import coverages
from .settings import MAX_ITER_DROPS, LOGGER

# %% ###################################################
# OBJECTS
########################################################

@dataclass
class Variable():
    id: str
    level: str | int
    label: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        # if another name is not given, use the id as name
        if self.label is None:
            self.label = self.id

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Variable':
        return cls(
            id=data['id'],
            level=data['level'],
            label=data.get('label'),
            description=data.get('description')
        )
    
    @classmethod
    def from_yaml(
        cls,
        path: str,
    ) -> 'Variable':
        with open(path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return cls.from_dict(data)

    def get_label(self) -> Optional[str]:
        return self.label

    def get_attrs(self) -> Dict[str, str | int | None]:
        return {
            'id': self.id,
            'level': self.level,
            'label': self.label,
            'description': self.description if self.description is not None else '-'
        }


@dataclass
class Model():
    id: str
    variables: List[Variable]
    forecast_hours: int  # hours
    analysis_hours: int  # hours
    runs: Optional[List[int]] = None  # hour
    label: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        # add the name of the model
        if self.label is None:
            self.label = self.id

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Model':
        variables = [
            Variable(**variable_data)
            for variable_data in data['variables']
        ]
        return cls(
            id=data['id'],
            label=data.get('label'),
            description=data.get('description'),
            variables=variables,
            forecast_hours=data['forecast_hours'],
            analysis_hours=data['analysis_hours'],
            runs=data.get('runs')
        )

    @classmethod
    def from_yaml(
        cls,
        path: str,
    ) -> 'Model':
        with open(path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return cls.from_dict(data)

    def get_label(self) -> Optional[str]:
        return self.label

    def get_runs(self) -> List[int]:
        if self.runs is None:
            runs = []
        else:
            runs = [int(rr) for rr in self.runs]
        return runs

    def get_forecast_hours(self) -> timedelta:
        return timedelta(hours=self.forecast_hours)

    def get_analysis_hours(self) -> timedelta:
        return timedelta(hours=self.analysis_hours)

    def get_variables(self) -> List[Variable]:
        return self.variables

    def get_attrs(self) -> Dict[str, str | int | List[int] | None]:
        return {
            'id': self.id,
            'label': self.label,
            'description': self.description if self.description is not None else '-',
            'forecast_hours': self.forecast_hours,
            'analysis_hours': self.analysis_hours,
            'runs': self.runs if self.runs is not None else '-'
        }

# %% ###################################################
# FUNCTIONS
########################################################

# %% HELPER FUNCTIONS

def round_to_closest(date: datetime, unit: str = 'hour') -> datetime:
    unit = unit.lower()
    round_map = {
        'hour': ('hours', datetime(date.year, date.month, date.day, date.hour)),
        'minute': ('minutes', date.replace(second=0, microsecond=0)),
        'second': ('seconds', date.replace(microsecond=0)),
    }
    if unit not in round_map:
        raise ValueError("round_unit must be one of: 'hour', 'minute', 'second'")

    freq, floor_date = round_map[unit]
    delta_seconds = (date - floor_date).total_seconds()
    threshold_seconds = {
        'hour': 1800,
        'minute': 30,
        'second': 0.5,
    }[unit]

    if delta_seconds >= threshold_seconds:
        floor_date = floor_date + pd.Timedelta(1, unit=freq)  # type: ignore
    return floor_date


def round_time_coord(data: xr.Dataset, unit: str = 'hour') -> xr.Dataset:
    unit = unit.lower()
    freq_map = {
        'hour': 'h',
        'minute': 'min',
        'second': 's',
    }
    if unit not in freq_map:
        raise ValueError("round_unit must be one of: 'hour', 'minute', 'second'")

    rounded_time = pd.to_datetime(data['time'].values, format='mixed').round(freq_map[unit])
    return data.assign_coords(time=rounded_time.to_numpy())


def insist(
    func,
    n_trials: int = MAX_ITER_DROPS,
    action_name: str | None = None,
    logger: logging.Logger | None = None,
    **kwargs
):
    """Try to execute a function for n_trials times."""
    logger = logger or LOGGER
    action_name = action_name or getattr(func, '__name__', 'unknown_action')
    success = False
    n = 0
    result = None
    while (n < n_trials) and (not success):
        n += 1
        try:
            logger.debug(
                '%s | attempt %s/%s',
                action_name,
                n,
                n_trials
            )
            result = func(**kwargs)
            success = True
        except Exception as exc:
            if n < n_trials:
                logger.warning('%s failed (%s/%s): %s', action_name, n, n_trials, exc)
            elif logger.isEnabledFor(logging.DEBUG):
                logger.exception('%s failed after %s attempts', action_name, n_trials)
            continue
    if not success:
        logger.error('%s failed after %s attempts', action_name, n_trials)
    return result, success


# %% MAIN FUNCTIONS

def get_model_dates(
    model: Model,
    time_from: datetime | str,
    time_to: datetime | str,
    all_variables: bool = True,
    logger: logging.Logger | None = None
) -> pd.DataFrame:
    """
    Get the available dates for a model in a given time range.
    """
    logger = logger or LOGGER

    # transform time to datetime
    if isinstance(time_from, str):
        time_from = datetime.strptime(time_from, '%Y%m%d%H%M')
    if isinstance(time_to, str):
        time_to = datetime.strptime(time_to, '%Y%m%d%H%M')
    # check that time_from is before time_to
    if time_from > time_to:
        raise ValueError('time_from must be before time_to')
    # put the time in UTC > needed for donwload
    tz = pytz.utc
    if time_from.tzinfo is None:
        time_from_sel = time_from.replace(tzinfo=tz)
    else:
        time_from_sel = time_from.astimezone(tz)
    # time to
    if time_to.tzinfo is None:
        time_to_sel = time_to.replace(tzinfo=tz)
    else:
        time_to_sel = time_to.astimezone(tz)

    # define the time for search
    start_time = (time_from_sel - model.get_forecast_hours()).strftime('%Y%m%d%H%M')
    stop_time = (time_to_sel + model.get_analysis_hours()).strftime('%Y%m%d%H%M')
    logger.info('Checking dates for model:%s in [%s, %s]', model.id, start_time, stop_time)
    all_dates, check = insist(
                        coverages.get_dates,
                        logger=logger,
                        action_name=f'get_dates model:{model.id}',
                        data_id=model.id,
                        date_from=start_time,
                        date_to=stop_time)

    if not check:
        logger.error('Unable to extract reference dates')
        return pd.DataFrame()
    if len(all_dates) == 0:  # type: ignore
        logger.warning('No reference dates available')
        return pd.DataFrame()

    # order the dates
    all_dates.sort(reverse=True)  # type: ignore 
    # clean the dates according to the runs
    runs = model.get_runs()
    if len(runs) == 0:
        all_dates_ok = all_dates.copy()  # type: ignore
    else:
        all_dates_ok = []
        for date in all_dates:  # type: ignore
            if date.hour in runs:
                all_dates_ok.append(date)
        logger.debug('Filtered dates with runs:%s > %s kept', runs, len(all_dates_ok))

    # get data for each variable
    info_all = pd.DataFrame()
    for variable in model.get_variables():
        logger.debug('Checking timeline for variable:%s', variable.id)
        info_var = pd.DataFrame()
        for date in all_dates_ok:  # type: ignore
            logger.debug('Extracting timeline for date_ref:%s', date)
            timeline, check = insist(
                                coverages.get_timeline,
                                logger=logger,
                                action_name=(
                                    'get_timeline '
                                    f'model:{model.id} variable:{variable.id} date_ref:{date}'
                                ),
                                data_id=model.id,
                                date_ref=date,
                                variable=variable.id,
                                level=variable.level)
            if not check:
                logger.error('Timeline extraction failed')
            if timeline is None:
                logger.warning('Empty timeline returned')
                continue

            # select timeline that you need
            timeline = np.array(timeline)
            timeline_ok = timeline[np.where(
                            (timeline >= time_from_sel) &
                            (timeline <= time_to_sel))]
            if len(timeline_ok) > 0:
                df = pd.DataFrame({'time': timeline_ok})
                df['date_ref'] = date
                info_var = pd.concat([info_var, df])
                logger.debug('Extracted %s times', len(timeline_ok))

        if len(info_var) == 0:
            logger.warning(
                'No valid timeline found for variable:%s', variable.id)

        if len(info_var) > 0:
            info_var['variable_id'] = variable.id
            info_all = pd.concat([info_all, info_var])

    if all_variables and len(info_all)>0:
        logger.debug('all_variables=True',
        '> Filtering timelines to keep only those with all variables available')        # take only the date_ref-time that contains all variables
        date_ref_times = info_all.groupby(['date_ref', 'time']).size()
        date_ref_times = date_ref_times[date_ref_times == len(model.get_variables())]
        info_all = info_all[info_all.set_index(['date_ref', 'time']).index.isin(date_ref_times.index)]

    if len(info_all) == 0:
        logger.warning('No timelines collected for model:%s', model.id)
        return pd.DataFrame()

    cols = ['date_ref', 'time', 'variable_id']
    info_all = info_all[cols].sort_values(cols)
    return info_all


def collect_data(
    model: Model,
    time_from: datetime | str,
    time_to: datetime | str,
    round_time: bool = True,
    round_unit: str = 'hour',
    all_variables: bool = True,
    only_last_run: bool = True,
    logger: logging.Logger | None = None
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Get model data"""
    logger = logger or LOGGER
    logger.info('Collecting %s', model.id)
    columns_metadata = ['time', 'variable_id', 'model_id', 'date_ref']

    # get dates ref
    df_dates = get_model_dates(
        model=model,
        time_from=time_from,
        time_to=time_to,
        logger=logger,
        all_variables=all_variables
    )
    if df_dates.empty:
        logger.warning('Skipping %s: no dates selected', model.id)
        return xr.Dataset(), pd.DataFrame(columns=columns_metadata)

    if only_last_run:
        # among the different date_ref for a time step, take only the most recent one
        df_dates = df_dates.sort_values('date_ref', ascending=False).drop_duplicates(subset=['time', 'variable_id'], keep='first')
        logger.debug('only_last_run=True > keeping only the most recent date_ref for each time step and variable')

    # take available variables
    vars_ok = df_dates['variable_id'].unique().tolist()
    if all_variables:
        # we are sure that all time steps have all variables
        # check that all variables are available
        if len(vars_ok) < len(model.get_variables()):
            logger.error('all_variables=True',
                'Not all variables have timelines for model=%s. '
                'Expected: %s, found: %s. Skipping model.',
                model.id,
                [variable.id for variable in model.get_variables()],
                vars_ok
            )
            return xr.Dataset(), pd.DataFrame(columns=columns_metadata)

    # get the data
    df_dates = df_dates.set_index(['variable_id', 'time'])
    data = xr.Dataset()
    metadata = pd.DataFrame(columns=columns_metadata)

    for variable in model.get_variables():
        logger.info('Extracting data for %s', variable.id)
        if variable.id not in vars_ok:
            logger.warning('Skipping variable:%s > no timeline', variable.id)
            continue
        data_var = xr.Dataset()
        df_dates_var = df_dates.xs(variable.id, level='variable_id')
        times_var = df_dates_var.index.get_level_values('time').unique()
        # order the times from oldest to most recent for the variable
        times_var = times_var.sort_values()

        # selection of data for each time step
        for tt in times_var:
            time_sel = tt.to_pydatetime()
            # order the date_ref from most recent to oldest for the time step
            dates = df_dates_var.loc[tt:tt]['date_ref'].sort_values(ascending=False).tolist()
            search_date = True
            logger.debug('Searching date_ref for time %s', tt)
            while len(dates) > 0 and search_date:
                date = dates.pop(0)
                date = date.to_pydatetime().replace(tzinfo=None)
                data_tmp, check = insist(
                    coverages.get_data,
                    logger=logger,
                    action_name=(
                        'get_data '
                        f'model:{model.id} variable:{variable.id} '
                        f'time:{tt} date_ref:{date}'
                    ),
                    data_id=model.id,
                    date_ref=date,
                    variable=variable.id,
                    level=variable.level,
                    date_selected=time_sel
                )
                if not check:
                    logger.error('Data extraction failed')
                if data_tmp is None:
                    logger.warning('No data returned')
                    continue
                search_date = False
                if round_time:
                    data_tmp = round_time_coord(data_tmp, unit=round_unit)
                # add info date_ref
                data_tmp = data_tmp.assign_coords(
                    date_ref=('time', [np.datetime64(date, 'ns')])
                )
                # save
                if len(data_var.data_vars) == 0:
                    data_var = data_tmp
                else:
                    data_var = xr.concat([data_var, data_tmp],
                                         dim='time')
                logger.info('time %s extracted from date_ref: %s', tt, date)
                # add the metadata
                df_metadata = pd.DataFrame({
                    'time': data_tmp['time'].values,
                    'variable_id': variable.id,
                    'model_id': model.id,
                    'date_ref': data_tmp['date_ref'].values
                })
                metadata = pd.concat([metadata, df_metadata], ignore_index=True)
            if search_date:
                logger.warning('No usable data found for time:%s', tt)

        if len(data_var.data_vars) == 0:
            logger.warning('Variable %s produced no data', variable.id)
            if all_variables:
                logger.warning('Skipping model %s > all_variables=True', model.id)
                return xr.Dataset(), pd.DataFrame(columns=columns_metadata)
            continue

        # add the attributes of the variable to the DataArray itself
        data_var[variable.id].attrs.update(variable.get_attrs())
        # merge all together
        data = xr.merge([data, data_var], compat='no_conflicts')
        logger.debug('Merged %s:%s', model.id, variable.id)
    data.attrs.update(model.get_attrs())
    metadata = metadata.sort_values(columns_metadata)

    # get how many time steps
    ntimes = len(data['time'])
    logger.info('Extracted data for %s: %s variables and %s times', model.id, len(data.data_vars), ntimes)
    return data, metadata


# %%
if __name__ == '__main__':
    model = Model(
        id='ICON_LAMI',
        variables=[
            Variable(id='2t', level='2.0', label='T', description='Temperature at 2m'),
            Variable(id='rh', level='2.0', label='RH', description='Relative Humidity at 2m')
        ],
        forecast_hours=72,
        analysis_hours=0,
        description='Test ICON'
    )

    time_from = '202604231800'
    time_to = '202604261800'

    df_dates = get_model_dates(model, time_from, time_to)

    data, metadata = collect_data(model, time_from, time_to)
# %%
