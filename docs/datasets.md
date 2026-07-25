# Datasets

AffKernel is evaluated on two public affordance-segmentation benchmarks.
**Neither is redistributed by this repository.** Download each from its original
source and honour the terms and citation requests of its authors.

---

## IIT-AFF

8,835 real-world images, 10 object classes, 9 affordance classes, split
6,184 train / 2,651 test. Introduced by Nguyen et al. (IROS 2017).

- Project page: <https://sites.google.com/site/iitaffdataset/>
- Pascal-VOC-formatted distribution (the layout this code expects) is linked
  from the AffordanceNet repository: <https://github.com/nqanh/affordance-net>

### Licence

**The IIT-AFF authors state no licence and no redistribution terms.** They ask
that you cite the original paper. Because no licence is granted, do not mirror
or re-host the data; download it from the authors' own links. If you need
clarification on permitted use, contact the dataset authors directly.

### Acquisition

Download the Pascal-VOC-formatted archive linked from the AffordanceNet
repository (Google Drive or OneDrive) and extract it so the tree matches the
layout below. The configuration key `root` in
`configs/dataset/iit_detection.yml` defaults to `./dataset/iit/data`.

```
dataset/iit/data/
├── VOCdevkit2012/
│   └── VOC2012/
│       ├── JPEGImages/          8835 x <id>.jpg
│       ├── Annotations/         8835 x <id>.xml   (VOC boxes + class names)
│       └── ImageSets/
│           └── Main/
│               ├── train.txt    6184 image ids
│               └── test.txt     2651 image ids
└── cache/
    ├── GTsegmask_VOC_2012_train/    <id>_<k>_segmask.sm
    └── GTsegmask_VOC_2012_test/     <id>_<k>_segmask.sm
```

The archive also ships `imagenet_models/VGG16.v2.caffemodel` (553 MB). That is
a leftover from the original AffordanceNet Caffe pipeline. **AffKernel never
reads it** and you can safely delete it.

### Annotation formats

**Boxes** come from the VOC XML files: `size`, then one `object` block per
instance with `name`, `bndbox`, and `difficult`. IIT-AFF sets no `difficult`
flags in practice.

**Affordance masks** are the `.sm` files: one file per object instance, named
`<image_id>_<instance_index>_segmask.sm` with `instance_index` starting at 1.
Instances are discovered by probing `_1_`, `_2_`, ... until a file is missing,
so **`.sm` ordering must align positionally with the `object` order in the XML**.

A `.sm` file is a **raw Python pickle of a `numpy.ndarray` of `uint8`**, the
same height and width as its JPEG, where each pixel holds an affordance class
id and 0 means background. The `.sm` extension is inherited from the
AffordanceNet distribution; it is not a bespoke binary format.

> **Security note.** The loader calls `pickle.load` on these files. Unpickling
> executes arbitrary code, so only load `.sm` files obtained from the official
> distribution. If you redistribute a derived dataset internally, prefer
> converting the masks to `.npy` or PNG first.

### Classes

Object classes, in label order:

```
__background__, bowl, tvm, pan, hammer, knife, cup, drill, racket, spatula, bottle
```

Affordance classes, where the index is the pixel value inside a `.sm` mask:

```
__background__, contain, cut, display, engine, grasp, hit, pound, support, w-grasp
```

### No validation split

IIT-AFF ships only `train` and `test`. There is no validation split, so this
codebase does **not** select a checkpoint on the test set. The epoch is fixed a
priori: the last epoch, using EMA weights. Any protocol that picks the best
epoch by test score is not comparable to the numbers reported here.

### Citation

```bibtex
@inproceedings{nguyen2017object,
  title     = {Object-Based Affordances Detection with Convolutional Neural
               Networks and Dense Conditional Random Fields},
  author    = {Nguyen, Anh and Kanoulas, Dimitrios and Caldwell, Darwin G. and
               Tsagarakis, Nikos G.},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and
               Systems (IROS)},
  year      = {2017}
}
```

---

## UMD Part-Affordance

28,843 labelled RGB-D frames of 105 tools on a turntable, 17 tool categories
and 7 affordance classes. Introduced by Myers et al. (ICRA 2015).

- Project page: <https://users.umiacs.umd.edu/~amyers/part-affordance-dataset/>

### Acquisition

Download and extract so the tree matches the layout below. The configuration
key `root` in `configs/dataset/umd_affordance.yml` defaults to `./dataset/umd`.

```
dataset/umd/part-affordance-dataset/
├── category_split.txt      novel-instance split (1 = train, 2 = test)
├── novel_split.txt         novel-category split
├── tool_categories.txt     tool name -> category id (1..17)
└── tools/
    └── <tool_name>/
        ├── <tool>_<frame:08d>_rgb.jpg
        ├── <tool>_<frame:08d>_depth.png          (not used)
        ├── <tool>_<frame:08d>_label.mat          (used)
        └── <tool>_<frame:08d>_label_rank.mat     (not used)
```

Only `*_rgb.jpg` and `*_label.mat` are read. `scipy.io.loadmat(...)["gt_label"]`
yields a 480x640 `uint8` single-label affordance map, 0 = background.

### Boxes are derived, not annotated

UMD ships no bounding-box annotations. Each frame contains exactly one tool, so
this codebase derives a single box per frame as the tight bounding box of the
non-zero affordance pixels. Frames whose label map is empty contribute zero
instances. This is a documented design choice; note it when comparing against
methods that use a different box convention.

### Splits

The paper's protocol uses `category_split.txt` (the novel-**instance** split):
14,823 train / 14,020 test frames at `frame_stride: 1`. `novel_split.txt`
provides the harder novel-**category** split.

### Classes

17 tool categories (`num_classes: 18` including background) and 7 affordances
(`num_affordance_classes: 8` including background):

```
grasp, cut, scoop, contain, pound, support, w-grasp
```

### Citation

```bibtex
@inproceedings{myers2015affordance,
  title     = {Affordance Detection of Tool Parts from Geometric Features},
  author    = {Myers, Austin and Teo, Ching L. and Ferm{\"u}ller, Cornelia and
               Aloimonos, Yiannis},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2015}
}
```

---

## Verifying a dataset install

Both datasets are optional for running the test suite: dataset-dependent tests
skip automatically when the data is absent.

```bash
# runs everything that does not need data on disk
python -m unittest discover -s tests -v

# once IIT-AFF is in place, these stop skipping
python -m unittest tests.test_iit_dataset -v

# once UMD is in place
python -m unittest tests.test_umd_dataset -v
```

To inspect the ground truth itself (class balance, instance counts, mask area
statistics) without training anything:

```bash
python tools/gt_anatomy.py -c configs/rtdetr/rtdetr_r50vd_6x_iit_v3_stride2_deepsup.yml
```
