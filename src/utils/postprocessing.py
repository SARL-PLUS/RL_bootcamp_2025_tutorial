from typing import List, Union
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import pandas as pd
import re

def get_tensorboard_record(run_dir: Path) -> EventAccumulator:
    tb_record_path = next(iter(run_dir.joinpath("tensorboard").rglob("events.out.tfevents.*")))
    ea = EventAccumulator(tb_record_path.as_posix())
    ea.Reload()
    return ea


def resolve_tags(obj: Union[EventAccumulator, pd.DataFrame], prefix) -> List[str]: 
    
    def get_tags(obj):
        if isinstance(obj, EventAccumulator):
            return obj.Tags()["scalars"]
        elif isinstance(obj, pd.DataFrame):
            return list(obj.columns)
        else:
            raise TypeError("Object must be EventAccumulator or DataFrame")

    tags = [tag for tag in get_tags(obj) if tag.startswith(prefix)]
    return sorted(tags)

def get_synced_traces(
    ea: EventAccumulator, tags: Union[str, List[str]]) -> List[np.ndarray]:
    df = pd.DataFrame()
    
    if isinstance(tags, str):
        tags = [tags]
    
    for tag in tags:

        if "steps" in df.columns:
            assert np.array_equal(df["steps"], np.asarray([scalar.step for scalar in ea.Scalars(tag)])),\
                "Unable to convert asynchronus data into one data frame"        
        else:
            df["steps"] = np.asarray([scalar.step for scalar in ea.Scalars(tag)])
            #df["steps"] = pd.to_numeric(df["steps"])

        df[tag] = np.asarray([scalar.value for scalar in ea.Scalars(tag)])
            
    df = df.sort_values(by="steps")
            
    return df


def get_async_traces(ea: EventAccumulator, tags: Union[str, List[str]]) -> List[pd.DataFrame]:
    return [get_synced_traces(ea=ea, tags=tag) for tag in tags]

