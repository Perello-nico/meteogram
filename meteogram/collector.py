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
from .settings import MAX_ITER_DROPS

LOGGER = logging.getLogger(__name__)


# %% ###################################################
# OBJECTS
########################################################

@dataclass
class Variable():
    id: str
    level: str | int
    name: Optional[str] = None

    def __post_init__(self) -> None:
        # if another name is not given, use the id as name
        if self.name is None:
            self.name = self.id

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Variable':
        return cls(
            id=data['id'],
            name=data.get('name'),
            level=data['level']
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
        return self.name

    def get_attrs(self) -> Dict[str, str | int | None]:
        return {
            'id': self.id,
            'name': self.name,
            'level': self.level
        }


@dataclass
class Model():
    id: str
    variables: List[Variable]
    forecast: int  # hours
    analysis: int  # hours
    runs: Optional[List[int]] = None  # hour
    name: Optional[str] = None

    def __post_init__(self) -> None:
        # add the name of the model
        if self.name is None:
            self.name = self.id

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Model':
        variables = [
            Variable(**variable_data)
            for variable_data in data['variables']
        ]
        return cls(
            id=data['id'],
            name=data.get('name'),
            variables=variables,
            forecast=data['forecast'],
            analysis=data['analysis'],
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
        return self.name

    def get_runs(self) -> List[int]:
        if self.runs is None:
            runs = []
        else:
            runs = [int(rr) for rr in self.runs]
        return runs

    def get_forecast(self) -> timedelta:
        return timedelta(hours=self.forecast)

    def get_analysis(self) -> timedelta:
        return timedelta(hours=self.analysis)

    def get_variables(self) -> List[Variable]:
        return self.variables

    def get_attrs(self) -> Dict[str, str | int | List[int] | None]:
        return {
            'id': self.id,
            'name': self.name,
            'forecast': self.forecast,
            'analysis': self.analysis,
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
    logger: logging.Logger | None = None,
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
    start_time = (time_from_sel - model.get_forecast()
                  ).strftime('%Y%m%d%H%M')
    stop_time = (time_to_sel + model.get_analysis()
                 ).strftime('%Y%m%d%H%M')
    logger.info('Checking dates for %s [%s, %s]', model.id, start_time, stop_time)
    all_dates, check = insist(
                        coverages.get_dates,
                        logger=logger,
                        action_name=f'get_dates model={model.id}',
                        data_id=model.id,
                        date_from=start_time,
                        date_to=stop_time)
    if not check:
        logger.error(
            'Unable to extract reference dates for model=%s',
            model.id
        )
        return pd.DataFrame()
    if len(all_dates) == 0:  # type: ignore
        logger.warning(
            'No reference dates available for model=%s',
            model.id
        )
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
        logger.debug('Filtered dates for %s with runs=%s: %s kept', model.id, runs, len(all_dates_ok))
    info_all = pd.DataFrame()
    for variable in model.get_variables():
        logger.debug('Checking timeline for %s:%s', model.id, variable.id)
        info_var = pd.DataFrame()
        for date in all_dates_ok:  # type: ignore
            timeline, check = insist(
                                coverages.get_timeline,
                                logger=logger,
                                action_name=(
                                    'get_timeline '
                                    f'model={model.id} variable={variable.id} '
                                    f'date_ref={date}'
                                ),
                                data_id=model.id,
                                date_ref=date,
                                variable=variable.id,
                                level=variable.level)
            if not check:
                logger.error(
                    'Timeline extraction failed for model=%s variable=%s '
                    'level=%s date_ref=%s',
                    model.id,
                    variable.id,
                    variable.level,
                    date
                )
            if timeline is None:
                logger.warning(
                    'Empty timeline returned for model=%s variable=%s '
                    'level=%s date_ref=%s',
                    model.id,
                    variable.id,
                    variable.level,
                    date
                )
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
        if len(info_var) == 0:
            logger.warning(
                'No valid timeline found in range for model=%s variable=%s',
                model.id,
                variable.id
            )
        if len(info_var) > 0:
            info_var['variable_id'] = variable.id
            info_all = pd.concat([info_all, info_var])
    if len(info_all) == 0:
        logger.warning(
            'No timelines were collected for model=%s',
            model.id
        )
        return pd.DataFrame()
    cols = ['variable_id', 'time', 'date_ref']
    info_all = info_all[cols].sort_values(cols)
    info_all = info_all.set_index(['variable_id', 'time'])
    logger.info('Found %s timeline entries for %s', len(info_all), model.id)
    return info_all


def get_model_data(
    model: Model,
    time_from: datetime | str,
    time_to: datetime | str,
    round_time: bool = False,
    round_unit: str = 'hour',
    logger: logging.Logger | None = None,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Get model data"""
    logger = logger or LOGGER
    logger.info('Collecting %s', model.id)
    # get dates ref
    df_dates = get_model_dates(
        model=model,
        time_from=time_from,
        time_to=time_to,
        logger=logger
    )
    if df_dates.empty:
        logger.warning('Skipping %s: no dates selected', model.id)
        return xr.Dataset(), pd.DataFrame(
            columns=['variable', 'model_id', 'variable_id', 'time', 'date_ref']
        )
    vars_ok = df_dates.index.get_level_values('variable_id').unique().tolist()
    # get the data
    data = xr.Dataset()
    metadata = pd.DataFrame(
        columns=['variable', 'model_id', 'variable_id', 'time', 'date_ref']
    )
    for variable in model.get_variables():
        if variable.id not in vars_ok:
            logger.warning('Skipping %s:%s, no timeline', model.id, variable.id)
            continue
        logger.info('Downloading %s: %s', model.id, variable.id)
        data_var = xr.Dataset()
        df_dates_var = df_dates.xs(variable.id, level='variable_id')
        times_var = df_dates_var.index.get_level_values('time').unique()
        # selection of data for each time step
        for tt in times_var:
            time_sel = tt.to_pydatetime()
            dates = df_dates_var.loc[tt:tt]['date_ref'].sort_values(ascending=False).tolist()
            search_date = True
            logger.info('Selecting date for %s:%s at %s', model.id, variable.id, tt)
            while len(dates) > 0 and search_date:
                date = dates.pop(0)
                date = date.to_pydatetime().replace(tzinfo=None)
                data_tmp, check = insist(
                    coverages.get_data,
                    logger=logger,
                    action_name=(
                        'get_data '
                        f'model={model.id} variable={variable.id} '
                        f'time={tt} date_ref={date}'
                    ),
                    data_id=model.id,
                    date_ref=date,
                    variable=variable.id,
                    level=variable.level,
                    date_selected=time_sel
                )
                if not check:
                    logger.error(
                        'Data extraction failed for model=%s variable=%s '
                        'time=%s date_ref=%s',
                        model.id,
                        variable.id,
                        tt,
                        date
                    )
                if data_tmp is None:
                    logger.warning(
                        'No data returned for model=%s variable=%s '
                        'time=%s date_ref=%s',
                        model.id,
                        variable.id,
                        tt,
                        date
                    )
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
                # add the metadata
                df_metadata = pd.DataFrame({
                    'variable': variable.name,
                    'model_id': model.id,
                    'variable_id': variable.id,
                    'time': data_tmp['time'].values,
                    'date_ref': data_tmp['date_ref'].values
                })
                metadata = pd.concat([metadata, df_metadata], ignore_index=True)
            if search_date:
                logger.warning(
                    'No usable data found for model=%s variable=%s time=%s',
                    model.id,
                    variable.id,
                    tt
                )
        if len(data_var.data_vars) == 0:
            logger.warning('Variable %s:%s produced no data', model.id, variable.id)
            continue
        # substitute with the variable name
        data_var = data_var.rename_vars(
                    {variable.id: variable.get_label()}
                    )
        # add the attributes of the variable to the DataArray itself
        var_name = variable.get_label()
        if var_name is not None:
            data_var[var_name].attrs.update(variable.get_attrs())
        # merge all together
        data = xr.merge([data, data_var], compat='no_conflicts')
        logger.debug('Merged %s:%s', model.id, variable.id)
    data.attrs.update(model.get_attrs())
    logger.info('Done %s: %s variables', model.id, len(data.data_vars))
    return data, metadata


def collect_data(
    time_from: datetime | str,
    time_to: datetime | str,
    models: List[dict[str, Any]],
    path_save: str,
    logger: logging.Logger | None = None,
):
    logger = logger or LOGGER

    # information of the time range for the name of the output files
    time_from_str = datetime.strftime(
        pd.to_datetime(time_from).to_pydatetime(),
        '%Y%m%d%H%M'
    )
    time_to_str = datetime.strftime(
        pd.to_datetime(time_to).to_pydatetime(),
        '%Y%m%d%H%M'
    )

    if len(models) == 0:
        raise ValueError('No models to collect.')

    for model_cfg in models:
        model = Model.from_dict(model_cfg)
        try:
            data, metadata = get_model_data(
                model=model,
                time_from=time_from,
                time_to=time_to,
                logger=logger
            )
            if len(data.data_vars) == 0:
                logger.warning('No data for %s; writing empty outputs', model.id)
            file_stub = f'{model.id}_{time_from_str}_{time_to_str}'
            path_data = os.path.join(path_save, f'{file_stub}.nc')
            path_metadata = os.path.join(path_save, f'{file_stub}_metadata.csv')
            logger.debug('Writing NetCDF: %s', path_data)
            data.to_netcdf(path_data)
            logger.debug('Writing metadata: %s', path_metadata)
            metadata.to_csv(path_metadata, index=False)
            logger.info('Saved %s', model.id)
        except Exception:
            logger.exception('Unhandled error while processing model=%s', model.id)

    logger.info('Collection completed')
