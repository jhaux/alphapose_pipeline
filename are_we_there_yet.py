'''Compare the log against all videos to check how much is already finished.'''
import os
import re
import time
import datetime
from estimate_and_track import listfiles


def walklevel(some_dir, level=1):
    all_files = []

    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):

        for fname in dirs + files:
            all_files += [os.path.join(root, fname)]

        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

    return all_files


def checker(logfile, root, continuous=False):
    '''Args:
        logfile (str): Where the log lives
        root (str): Where the videos live.
    '''

    with open(logfile, 'r') as lfile:
        data = lfile.readlines()

    done_videos = [v.strip('Done/crops: ') for v in data if 'Done/crops: ' in v]

    regex = re.compile('.+seq$')
    all_videos = [v for v in walklevel(root, 1) if regex.match(v) is not None]

    frac = len(done_videos) / len(all_videos)
    print('{}% - done: {}, total: {}'.format(100 * frac, len(done_videos), len(all_videos)))

    if continuous:
        from scipy.stats import linregress
        t0 = time.time()
        n_done_start = len(done_videos)

        fracs = []
        ts = []
        while True:
            with open(logfile, 'r') as lfile:
                data = lfile.readlines()

            done_videos = [v.strip('Done/crops: ') for v in data if 'Done/crops: ' in v]

            frac = len(done_videos) / len(all_videos)
            print('{}% - done: {}, total: {}'.format(100 * frac, len(done_videos), len(all_videos)))

            not_done = len(all_videos) - len(done_videos)

            delta_v = len(done_videos) - n_done_start
            if delta_v != 0:
                rate = (time.time() - t0) / (delta_v)

                eta = not_done * rate
                t_final = datetime.timedelta(eta)
                print('ETA: {}'.format(t_final))
            else:
                print('ETA: ?')

            time.sleep(5)



if __name__ == '__main__':
    checker('./estimate_and_track_log.txt', '/export/scratch/jhaux/Data/olympic_sports_new/', True)
