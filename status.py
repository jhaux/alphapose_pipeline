from multiperson.aggregated_dataset import find_videos
from tqdm import tqdm
import os


def done_parts(videos):

    frames = 0
    poses = 0
    tracks = 0
    labels = 0
    crops = 0
    masks = 0
    done_vids = 0

    for vid in videos:
        vid_ = vid.replace(' ', '\ ')
        done = True
        if os.path.exists(vid+'_frames'):
            frames += 1
        else:
            done = False

        pose_file = vid+'_track/alphapose-results.json'
        if os.path.exists(pose_file):
            poses += 1
        else:
            done = False
        
        track_file = vid+'_track/alphapose-forvis-tracked.json'
        if os.path.exists(track_file):
            tracks += 1
        else:
            done = False

        lpath = os.path.join(vid + '_track', 'per_person_labels.npz')
        if os.path.exists(lpath):
            labels += 1
        else:
            done = False

        success_file = os.path.join(vid+'_crops', '.Success')
        if os.path.exists(success_file):
            crops += 1
        else:
            done = False

        success_file = os.path.join(vid+'_masks', '.Success')
        if os.path.exists(success_file):
            masks += 1
        else:
            done = False


        if done:
            done_vids += 1

    return {'frames': frames,
            'poses': poses,
            'tracks': tracks,
            'labels': labels,
            'crops': crops,
            'masks': masks,
            'done': done_vids,
            'n_vids': len(videos)}

def status_of_video(video):
    status = {}

    vid = video

    done = True
    if os.path.exists(vid+'_frames'):
        status['frames'] = True
    else:
        status['frames'] = False
        done = False

    pose_file = vid+'_track/alphapose-results.json'
    if os.path.exists(pose_file):
        status['poses'] = True
    else:
        status['poses'] = False
        done = False
    
    track_file = vid+'_track/alphapose-forvis-tracked.json'
    if os.path.exists(track_file):
        status['tracks'] = True
    else:
        status['tracks'] = False
        done = False

    lpath = os.path.join(vid + '_track', 'per_person_labels.npz')
    if os.path.exists(lpath):
        status['labels'] = True
    else:
        status['labels'] = False
        done = False

    success_file = os.path.join(vid+'_crops', '.Success')
    if os.path.exists(success_file):
        status['crops'] = True
    else:
        status['crops'] = False
        done = False

    success_file = os.path.join(vid+'_masks', '.Success')
    if os.path.exists(success_file):
        status['masks'] = True
    else:
        status['masks'] = False
        done = False

    status['done'] = done

    return status


def status(root, ext):
    print('Pipeline Status')
    print(root)
    videos = find_videos(root, ext)

    results = done_parts(videos)

    n = results['n_vids']
    for k, v in results.items():
        if k == 'n_vids':
            continue
        p = v / n * 100
        print('{}:\t{}/{} = {:3.3f}%'.format(k, v, n, p))


if __name__ == '__main__':
    import argparse

    A = argparse.ArgumentParser()
    
    A.add_argument('--root',
                   type=str,
                   default='/export/scratch/jhaux/Data/olympic_test/')
    
    A.add_argument('--ext',
                   type=str,
                   nargs='+',
                   default='mp4')

    args = A.parse_args()
    root = args.root
    ext = args.ext

    status(root, ext)
