"""Compact panel storage and bounded-size hierarchical lookups."""
import numpy as np
import pandas as pd

ID_COLUMNS = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']


def compact_panel(frame):
    """Return a shallow frame with compact columns, without copying all blocks."""
    frame = frame.copy(deep=False)
    for col in ID_COLUMNS:
        if col in frame and not isinstance(frame[col].dtype, pd.CategoricalDtype):
            frame[col] = frame[col].astype('category')
    for col in frame.select_dtypes(include=['float64']).columns:
        frame[col] = frame[col].astype('float32')
    for col in frame.select_dtypes(include=['int64']).columns:
        frame[col] = pd.to_numeric(frame[col], downcast='integer')
    return frame


def order_panel(frame):
    """Avoid sorting/copying an already id/date ordered wide feature table."""
    ids = (frame.id.cat.codes.to_numpy() if isinstance(frame.id.dtype, pd.CategoricalDtype)
           else pd.factorize(frame.id, sort=True)[0])
    dates = frame.date.to_numpy()
    for start in range(1, len(frame), 250_000):
        stop = min(start + 250_000, len(frame))
        before, after = ids[start-1:stop-1], ids[start:stop]
        if ((after < before).any() or
                ((after == before) & (dates[start:stop] < dates[start-1:stop-1])).any()):
            return frame.sort_values(['id', 'date'])
    return frame


def add_lookup_columns(frame, lookup, keys, columns, chunk_size=250_000):
    """Add one float32 column at a time; never merge/copy the wide left table."""
    lookup = lookup.set_index(keys)
    if not lookup.index.is_unique:
        raise ValueError(f'Duplicate lookup keys: {keys}')
    # One integer indexer, reused for all feature columns. No full-size MultiIndex.
    positions = np.empty(len(frame), dtype=np.int32)
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        part = frame.iloc[start:stop]
        index = (pd.MultiIndex.from_frame(part[keys]) if len(keys) > 1
                 else pd.Index(part[keys[0]]))
        positions[start:stop] = lookup.index.get_indexer(index)
    for col in columns:
        source = lookup[col].to_numpy(dtype=np.float32)
        values = np.empty(len(frame), dtype=np.float32)
        for start in range(0, len(frame), chunk_size):
            stop = min(start + chunk_size, len(frame))
            selected = positions[start:stop]
            values[start:stop] = source[np.maximum(selected, 0)]
            values[start:stop][selected < 0] = np.nan
        frame[col] = values
    return frame
