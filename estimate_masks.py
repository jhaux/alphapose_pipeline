from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from multiperson.multiperson_dataset import MultiPersonDataset
from PIL import Image


def make_masks(video):
    '''Estimate masks and match them given the keypoints of the current
    person.
    '''

    M = LazyMaskRCNN()

    mask_path = video + '_masks'
    os.makedirs(mask_path, exist_ok=True)

    data_root = video + '_track'
    print(data_root)
    MP = MultiPersonDataset(data_root)

    crops_path = video + '_crops'

    for i in tqdm(range(len(MP)), desc='M'):
        example = MP[i]
        im_path = example['frame_path']
        box = example['bbox']
        pid = example['person_id']
        sid = example['sequence_idx']
        fid = example['frame_idx']
        kps = example['keypoints_rel']

        cropname = '{:0>5}-p{:0>3}-s{:0>3}-f{:0>3}.png'.format(
                i,
                pid,
                sid,
                fid
                )
        croppath = os.path.join(crops_path, cropname)
        crop_im = np.array(Image.open(croppath))

        masks, scores = M.get_person_mask(crop_im)

        # Scale keypoints to pixel coordinates of crop
        kps[..., :2] = kps[..., :2] * box[2:]

        if len(masks) == 0:
            continue

        savepath = '{:0>5}-p{:0>3}-s{:0>3}-f{:0>3}.png'.format(
                i,
                pid,
                sid,
                fid,
                )
        savepath = os.path.join(mask_path, savepath)

        if i < 10:
            d = savepath.replace('.png', 'debug.png')
        else:
            d = None
        mask, mscore = filter_masks(masks, scores, kps, debug=d)

        if i == 0:
            print(savepath)

        mask_im = Image.fromarray(mask.astype(np.uint8)*255, mode='L')
        mask_im = mask_im.convert('1')
        mask_im.save(savepath, 'PNG')

    with open(os.path.join(mask_path, '.Success'), 'w+') as sf:
        sf.write('We did it!')


def filter_masks(masks, scores, keypoints, debug=False):
    match_scores = []
    match_lists = []
    for m, s in zip(masks, scores):
        matches = []
        for x, y, c in keypoints:
            v = m[int(y), int(x)]
            matches += [v]
        match_scores += [sum(matches)]
        match_lists += [matches]

    win_idx = np.argmax(match_scores)

    if debug:
        import matplotlib.pyplot as plt

        f, AX = plt.subplots(1, len(masks))

        if len(masks) == 1:
            AX = [AX]

        for idx, [m, ax, ml] in enumerate(zip(masks, AX, match_lists)):
            ax.imshow(m)
            colors = ['green' if s else 'red' for s in ml]
            ax.scatter(keypoints[..., 0], keypoints[..., 1], c=colors)
            if idx == win_idx:
                ax.set_title('Winner')
        f.savefig(debug)
        plt.close()
    return masks[win_idx], scores[win_idx]


class LazyMaskRCNN(object):
    model = None
    def __init__(self, root='/export/home/jhaux/work_projects/Mask_RCNN'):
        self.root = root
        if LazyMaskRCNN.model is None:
            self.init(self.root)

    def get_person_mask(self, image):

        results = LazyMaskRCNN.model.detect([image])[0]
        class_ids = results['class_ids']
        masks = results['masks'].transpose(2, 0, 1)
        scores = results['scores']

        # Only consider persons masks
        selection = class_ids == 1

        masks = masks[selection]
        scores = scores[selection]

        return masks, scores


    def init(self, root='/export/home/jhaux/work_projects/Mask_RCNN'):
        sys.path.append(root)  # To find local version of the library
        from mrcnn import utils
        import mrcnn.model as modellib
        # Import COCO config
        sys.path.append(os.path.join(root, "samples/coco/"))  # To find local version
        import coco
        
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(root, "logs")
        
        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(root, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        
        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        LazyMaskRCNN.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        
        # Load weights trained on MS-COCO
        LazyMaskRCNN.model.load_weights(COCO_MODEL_PATH, by_name=True)


if __name__ == '__main__':
    import argparse

    A = argparse.ArgumentParser()
    A.add_argument('-v', '--video', type=str, help='Path to video')

    args = A.parse_args()

    make_masks(args.video)
