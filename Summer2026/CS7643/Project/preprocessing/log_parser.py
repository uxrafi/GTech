"""parse structured HDFS traces into per-block event sequences.

the raw HDFS.log is parsed upstream by LogHub (Drain) into Event_traces.csv,
where each block's ordered event-template ids live in the Features column as
"[E5,E22,E5,...]". this module turns that file into (block_id, [event_ids]).
"""
import re
from typing import List, Tuple

import pandas as pd

# event-template token, e.g. E5, E26
EVENT_RE = re.compile(r"E\d+")


def parse_features(features: str) -> List[str]:
    """extract ordered event-template ids from one Features cell."""
    if pd.isna(features):
        return []
    return EVENT_RE.findall(str(features))


def parse_event_traces(path: str) -> List[Tuple[str, List[str]]]:
    """read Event_traces.csv -> [(block_id, [event_ids]), ...] in file order."""
    df = pd.read_csv(path, usecols=["BlockId", "Features"])
    sequences = df["Features"].map(parse_features)
    return list(zip(df["BlockId"].tolist(), sequences.tolist()))


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/Event_traces.csv"
    traces = parse_event_traces(src)
    lengths = [len(seq) for _, seq in traces]
    print(f"parsed {len(traces)} blocks from {src}")
    print(f"length min/mean/max: {min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)}")
    print(f"first block: {traces[0][0]} -> {traces[0][1][:12]}...")
