import pandas as pd
import numpy as np

from edflow.util import pp2mkdtable


STAGES = [
        'Start',
        'frames',
        'estimate',
        'track',
        'csv+npz',
        'crops',
        'Masks',
        'Flow',
        'Done'
        ]


ERROR = 'Error'


def loader(log_file):
    df = pd.read_csv(log_file, delimiter=': ', names=['Stage', 'Video', 'Message'])

    return df.to_numpy()


def get_stages(logs):
    videos_in_log = np.unique(logs[:, 1])

    stages_per_vid = {}

    for vid in videos_in_log:
        has_errors = len(logs[0]) == 3
        if has_errors:
            stages = logs[logs[:, 1] == vid][:, [0, 2]]
        else:
            stages = logs[logs[:, 1] == vid][:, [0]]

        stages_per_vid[vid] = stages

    return stages_per_vid


def display_status(stages_per_vid, stages=STAGES, error=ERROR):
    to_print = {}
    finished = []
    for vid, completed in stages_per_vid.items():
        seen_stages = np.array(completed)[:, 0]
        if error in seen_stages:
            to_print[vid] = completed[0, 1]
        elif stages[-1] in seen_stages:
            finished += [vid]
            continue
        else:
            def sort_key(v):
                v = v[0]
                if ' - skipped' in v:
                    element = v[:len(v) - len(' - skipped')]
                else:
                    element = v
                return stages.index(element)
            to_print[vid] = sorted(completed, key=sort_key)[-1][0]

    to_print = pp2mkdtable(to_print)
    print(to_print)
    print('Finished {} videos'.format(len(finished)))
    return to_print


if __name__ == '__main__':
    import argparse
    import os

    A = argparse.ArgumentParser()

    A.add_argument('-l', '--log', type=str, help='Gimme some logs',
                   default=None)
    A.add_argument('-r', '--root', type=str,
                   help='Here should be a log. (Either use -l or -p)',
                   default=None)

    args = A.parse_args()

    logfile = args.log
    log_root = args.root

    if logfile is None:
        if log_root is None:
            raise ValueError('Must Specify either --log or --root')
        logfiles = [o for o in os.listdir(log_root) if 'pipeline_log' in o]
        logfile = [o for o in logfiles if o[0] != '.'][0]
        logfile = os.path.join(log_root, logfile)

        print('Displaying status from the following logfile:\n{}'
              .format(logfile))

    logs = loader(logfile)
    stages = get_stages(logs)
    status = display_status(stages)
