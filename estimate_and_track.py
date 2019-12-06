from tqdm import tqdm, trange
import argparse
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import re
from PIL import Image
import sys
import traceback
import json

from edflow.data.util import get_support, adjust_support
from multiperson.multiperson_dataset import MultiPersonDataset
from status import status_of_video


PARTS = [
        '',
        'frames',
        'poses',
        'tracks',
        'labels',
        'crops',
        'masks',
        'done'
        ]

DONE = 'done'


def test_ending(string, tests=[], mode='or'):
    if mode == 'or':
        for test in tests:
            if string[-len(test):] == test:
                return True
        return False

    elif mode == 'and':
        for test in tests:
            if string[-len(test):] != test:
                return False
        return True

    else:
        raise ValueError('Unrecognized mode. Must be one of `or`, '
                         + '`and` but is {}'.format(mode))


def listfiles(folder):
    ret_list = []
    for root, folders, files in os.walk(folder):
        new_folders = []
        for f in folders:
            if not test_ending(f, ['_frames', '_masks', '_crops', '_track']):
                new_folders += [f]
        folders[:] = new_folders
        for filename in folders + files:
            ret_list += [os.path.join(root, filename)]

    return  ret_list


def e_and_t(vgplf):
    videos, gpu, parts, logfile, force = vgplf

    for vid in videos:
        print(vid)
        vid_ = vid.replace(' ', '\ ')
        try:
            with open(logfile, 'a+') as f:
                f.write('Start: {}\n'.format(vid))
            if 1 in parts:
                if not os.path.exists(vid+'_frames') or 1 in force:
                    os.system('bash make_frames.sh {} {}'.format(vid_, gpu))
                    with open(logfile, 'a+') as f:
                        f.write('frames: {}\n'.format(vid))
                else:
                    with open(logfile, 'a+') as f:
                        f.write('frames - skipped: {}\n'.format(vid))

            if 2 in parts:
                pose_file = vid+'_track/alphapose-results.json'
                if not os.path.exists(pose_file) or 2 in force:
                    os.system('bash estimate_olympic {} {}'.format(vid_, gpu))
                    with open(logfile, 'a+') as f:
                        f.write('estimate: {}\n'.format(vid))
                else:
                    with open(logfile, 'a+') as f:
                        f.write('estimate - skipped: {}\n'.format(vid))
            
            if 3 in parts:
                track_file = vid+'_track/alphapose-forvis-tracked.json'
                if not os.path.exists(track_file) or 3 in force:
                    os.system('bash track_olympic.sh {} {}'.format(vid_, gpu))
                    with open(logfile, 'a+') as f:
                        f.write('track: {}\n'.format(vid))
                else:
                    with open(logfile, 'a+') as f:
                        f.write('track - skipped: {}\n'.format(vid))

            if 4 in parts:
                lpath = os.path.join(vid + '_track', 'per_person_labels.npz')
                if not os.path.exists(lpath) or 4 in force:
                    make_csv_and_npz(vid)
                    with open(logfile, 'a+') as f:
                        f.write('csv+npz: {}\n'.format(vid))
                else:
                    with open(logfile, 'a+') as f:
                        f.write('csv+npz - skipped: {}\n'.format(vid))

            if 5 in parts:
                success_file = os.path.join(vid+'_crops', '.Success')
                if not os.path.exists(success_file) or 5 in force:
                    make_crops(vid)
                    with open(logfile, 'a+') as f:
                        f.write('crops: {}\n'.format(vid))
                else:
                    with open(logfile, 'a+') as f:
                        f.write('crops - skipped: {}\n'.format(vid))

            if 6 in parts:
                success_file = os.path.join(vid+'_masks', '.Success')
                if not os.path.exists(success_file) or 5 in force:
                    os.system('bash make_masks.sh {} {}'.format(vid_, gpu))
                    with open(logfile, 'a+') as f:
                        f.write('Masks: {}\n'.format(vid))
                else:
                    with open(logfile, 'a+') as f:
                        f.write('Masks - skipped: {}\n'.format(vid))

            if 7 in parts:
                make_flows(vid)
                with open(logfile, 'a+') as f:
                    f.write('Flow: {}\n'.format(vid))

            with open(logfile, 'a+') as f:
                f.write('Done: {}\n'.format(vid))
        except Exception as e:
            with open(logfile, 'a+') as f:
                f.write('Error: {} -*- {}\n'.format(vid, str(e).replace(': ', '; ')))

            if not isinstance(e, FileNotFoundError):
                traceback.print_exc()
            continue


def extract_lines(tracking_data):
    ''' Converts dict of list of persons to list of persons with frame
    annotation.
    
    Args:
        tracking_data (dict): ``frame: [{idx: 1, ...}, {...}, ...]``
    '''

    linear = []
    for i, k in enumerate(sorted(tracking_data.keys())):
        for data in tracking_data[k]:
            example = {'orig': k, 'fid': i}
            example.update(data)
            linear += [example]

    sorted_linear = sorted(linear, key=lambda e: [e['idx'], e['fid']])

    last_id_change = 0
    last_id = None
    last_fid = -1
    for example in sorted_linear:
        ex_id = example['idx']
        if last_id != ex_id or last_fid != example['fid'] - 1:
            last_id_change = example['fid']

        seq_idx = example['fid'] - last_id_change
        example['sid'] = seq_idx
        last_id = ex_id
        last_fid = example['fid']

    return sorted_linear


def prepare_keypoints(kps_raw):
    '''Converts kps of form ``[x, y, c, x, y, c, ...]`` to 
    ``[[x, y, c], [x, y, c], ...]``'''

    x = kps_raw[::3]
    y = kps_raw[1::3]
    c = kps_raw[2::3]

    return np.stack([x, y, c], axis=-1)

def square_bbox(prepared_kps, pad=0.35, kind='percent'):
    if not kind in ['percent', 'abs']:
        raise ValueError('`kind` must be one of [`percent`, `abs`], but is {}'
                         .format(kind))

    x = prepared_kps[:, 0]
    y = prepared_kps[:, 1]

    minx, maxx = x.min(), x.max()
    miny, maxy = y.min(), y.max()

    wx = maxx - minx
    wy = maxy - miny
    w = max(wx, wy)

    centerx = minx + wx / 2.
    centery = miny + wy / 2.

    if pad is not None and pad != 0:
        if kind == 'percent':
            w = (1 + pad) * w
        else:
            w += pad

    bbox = np.array([centerx - w/2., centery - w/2., w, w])

    return bbox


def get_kps_rel(kps_abs, bbox):
    kps_rel = np.copy(kps_abs)
    kps_rel[:, :2] = kps_rel[:, :2] - bbox[:2]

    kps_rel[:, :2] = kps_rel[:, :2] / bbox[2:]

    return kps_rel


def add_lines_to_csv(data_frame, track_dir, frame_dir, root, kp_in_csv=True):
    json_name = os.path.join(root,
                             track_dir,
                             'alphapose-forvis-tracked.json')

    with open(json_name, 'r') as json_file:
        tracking_data = json.load(json_file)

    all_kps_abs = []
    all_kps_rel = []
    all_boxes = []

    raw_lines = extract_lines(tracking_data)
    for j, line in enumerate(tqdm(raw_lines, 'L')):
        kps_abs = prepare_keypoints(line['keypoints'])
        bbox = square_bbox(kps_abs)
        kps_rel = get_kps_rel(kps_abs, bbox)

        frame_root = os.path.join(root, frame_dir, line['orig'])

        vid = os.path.join(root, frame_dir[:-7])
        pid = line['idx']
        fid = line['fid']
        sid = line['sid']

        if kp_in_csv:
            data_frame = data_frame.append(
                    {
                        'frame_path': frame_root,
                        'video_path': vid,
                        'frame_idx': fid,
                        'sequence_idx': sid,
                        'person_id': pid,
                        'keypoints_abs': kps_abs,
                        'bbox': bbox,
                        'keypoints_rel': kps_rel
                        },
                    ignore_index=True  # append with incremental index
                    )
        else:
            all_kps_abs += [kps_abs]
            all_kps_rel += [kps_rel]
            all_boxes += [bbox]

            data_frame = data_frame.append(
                    {
                        'frame_path': frame_root,
                        'video_path': vid,
                        'frame_idx': fid,
                        'sequence_idx': sid,
                        'person_id': pid,
                        },
                    ignore_index=True  # append with incremental index
                    )

    if not kp_in_csv:
        return data_frame, np.stack(all_kps_abs), np.stack(all_kps_rel), np.stack(all_boxes)
    else:
        return data_frame


def make_csv_and_npz(video):
    '''Writes a csv containing all frame paths, with person id etc and a .npz
    containing all keypoints of each person as well as the bounding boxes around
    those keypoints with the keypoints relative to that bounding box.
    '''

    data_frame = pd.DataFrame(columns=
            [
                'frame_path',
                'video_path',
                'frame_idx',
                'sequence_idx',
                'person_id'
                ]
            )

    root = os.path.dirname(video)
    data_frame, kps_abs, kps_rel, boxes = add_lines_to_csv(data_frame,
                                                           video+'_track',
                                                           video+'_frames',
                                                           root,
                                                           False)

    csv_name = os.path.join(video + '_track', 'per_person_content.csv')
    data_frame.to_csv(csv_name, sep=';', index=False)

    lpath = os.path.join(video + '_track', 'per_person_labels.npz')
    labels = {'keypoints_abs': kps_abs,
              'keypoints_rel': kps_rel,
              'bbox': boxes
             }
    np.savez(lpath, **labels)


def crop(image, box):
    '''Arguments:
        image (np.ndarray or PIL.Image): Image to crop.
        box (list): Box specifying ``[x, y, width, height]``
        points (np.ndarray): Optional set of points in image coordinate, which
        are translated to box coordinates. Shape: ``[(*), 2]``.

    Returns:
        np.ndarray: Cropped image with shape ``[W, H, C]`` and same support
            as :attr:`image`.

        If points is not None:
            np.ndarray: The translated point coordinates.
    '''

    is_image = True
    if not isinstance(image, Image.Image):
        in_support = get_support(image)
        image = adjust_support(image, '0->255')
        image = Image.fromarray(image)
        is_image = False

    box[2:] = box[:2] + box[2:]

    image = image.crop(box)

    if not is_image:
        image = adjust_support(np.array(image), in_support)

    return image


def make_crops(video):
    crop_path = video + '_crops'
    os.makedirs(crop_path, exist_ok=True)

    data_root = video + '_track'
    print(data_root)
    MP = MultiPersonDataset(data_root)

    for i in trange(len(MP), desc='Crop'):
        example = MP[i]
        im_path = example['frame_path']
        box = example['bbox']
        pid = example['person_id']
        sid = example['sequence_idx']
        fid = example['frame_idx']

        crop_im = crop(Image.open(im_path), box)

        savepath = '{:0>5}-p{:0>3}-s{:0>3}-f{:0>3}.png'.format(
                i,
                pid,
                sid,
                fid
                )
        savepath = os.path.join(crop_path, savepath)
        if i == 0:
            print(savepath)
        crop_im.save(savepath, 'PNG')
    with open(os.path.join(crop_path, '.Success'), 'w+') as sf:
        sf.write('We did it!')


def make_flows(video):
    '''Estimate the flow between sets of frames.'''

    pass


if __name__ == '__main__':
    from datetime import datetime
    from multiperson.aggregated_dataset import find_videos

    A = argparse.ArgumentParser()
    
    A.add_argument('--root',
                   type=str,
                   default='/export/scratch/jhaux/Data/olympic_test/')
    
    # A.add_argument('--nw',
    #                type=int,
    #                default=10)

    A.add_argument('--p', type=int, nargs='+', default=list(range(10)),
                   help='Which parts to do')
    A.add_argument('--f', type=int, nargs='*', default=[],
                   help='Which parts to force and not skip. '
                        'Only considers parts specified by `--p`.')
    
    A.add_argument('--gpus',
                   type=int,
                   nargs='+',
                   default=list(range(5)))
    
    A.add_argument('--per_gpu',
                   type=int,
                   default=2)

    A.add_argument('--ext',
                   type=str,
                   nargs='+',
                   default='mp4')

    A.add_argument('--vids',
                   type=str,
                   nargs='+',
                   default=None)

    args = A.parse_args()
    root = args.root
    gpus = args.gpus
    per_gpu = args.per_gpu
    force = args.f
    
    nw = len(gpus) * per_gpu
    
    if args.vids is None:
        all_videos = find_videos(root, args.ext)
    else:
        all_videos = args.vids

    videos = []
    for v in all_videos:
        any_not_done = False
        status = status_of_video(v)
        for p in args.p:
            key = PARTS[p]
            if not status[key]:
                videos += [v]
                continue

    vid_indices_per_gpu = np.array_split(videos, len(gpus))

    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y_%H:%M:%S")
    logfile = os.path.join(root, 'pipeline_log.{}.txt'.format(date_time))

    old_logs = [o for o in os.listdir(root) if 'pipeline_log' in o]
    for o in old_logs:
        if o[0] != '.':
            print(o)
            src = os.path.join(root, o)
            dst = os.path.join(root, '.'+o)
            os.rename(src, dst)
    
    with mp.Pool(nw) as pool:
        args_ = []
        for gpu, indices in zip(gpus, vid_indices_per_gpu):
            sub_indices = np.array_split(indices, per_gpu)
            for si in sub_indices:
                args_ += [(si, gpu, args.p, logfile, force)]
    
        print(args_)
        
        tqdm(pool.map(e_and_t, args_), total=len(videos))
