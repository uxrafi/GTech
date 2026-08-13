# HDFS Data Exploration Summary

Source: LogHub HDFS_v1. Raw `HDFS.log` (11,175,629 lines) is parsed upstream by
Drain into `Event_traces.csv`, which groups events by block into ordered
event-template sequences. Ground-truth labels come from `anomaly_label.csv`.

## Table 1: Data Statistics

| Statistic | Value |
|---|---|
| Total blocks (sequences) | 575,061 |
| Total event occurrences | 11,175,629 |
| Unique event templates | 29 |
| Normal blocks | 558,223 (97.07%) |
| Anomalous blocks | 16,838 (2.93%) |
| Sequence length (min) | 2 |
| Sequence length (median) | 19 |
| Sequence length (mean) | 19.4 |
| Sequence length (max) | 298 |

## Class imbalance

The anomaly rate is 2.93%. This is preserved through the pipeline (no
undersampling) and handled downstream in the loss (`pos_weight` in BCE).
Accuracy is a vanity metric here; an all-normal predictor scores 97%. The real
scoreboard is precision / recall / F1 / AUC on the anomaly class.

## Sequence length and the choice of T

Lengths are right-skewed: median 19, mean 19.4, max 298. Padding/truncating to a
fixed T = 50 truncates only 55 of 575,061 sequences (0.0096%), so almost no
signal is lost at T = 50. For contrast, T = 20 would truncate 24.6%. T is left as
a tunable hyperparameter for the paper (Table 2); T = 50 is the default.

## Event distribution

29 unique templates (matching DeepLog's reported count for HDFS). The 10 most
frequent events:

| Event | Count | Template |
|---|---|---|
| E5 | 1,723,232 | Receiving block |
| E26 | 1,719,741 | addStoredBlock: blockMap updated |
| E11 | 1,706,679 | PacketResponder terminating |
| E9 | 1,706,514 | Received block of size |
| E21 | 1,402,047 | Deleting block file |
| E23 | 1,396,174 | delete: added to invalidSet |
| E22 | 575,061 | allocateBlock |
| E3 | 428,726 | Served block to |
| E4 | 356,207 | Got exception while serving |
| E2 | 120,036 | Verification succeeded for |

E22 (block allocation) appears exactly once per block, as expected.

## Qualitative: normal vs anomalous

Normal sequences follow a stable allocate, receive, respond, replicate story
(e.g. `E5 E22 E5 E11 E9 E26 ...`). Anomalous blocks are distinguished mainly by rare
error and control events that almost never occur in normal executions: E20 ("Unexpected
error trying to delete block"), E27/E28 (redundant or orphaned addStoredBlock requests),
and E29 (replication timeout). Note that "Got exception while serving" (E4) is not a
reliable anomaly signal here: it is slightly more common in normal blocks, so whether
event order adds signal beyond event frequency is a question for the model comparison,
not a foregone conclusion.

## Data contract (produced by the pipeline)

`data/sequences/`: `X_{train,val,test}.npy` (int32 [N, 50], right-padded with 0),
`y_*.npy` (int8, 0=normal 1=anomaly), `len_*.npy` (int32, true clipped lengths),
`vocab.json` (`<PAD>`=0, `<UNK>`=1, E1..E29 → 2..30; vocab_size 31),
`metadata.json`, `splits.json`. Stratified 80/10/10 split, seed 42; the 2.93%
anomaly ratio holds in every split; one block = one sequence, so there is no
cross-split leakage.
