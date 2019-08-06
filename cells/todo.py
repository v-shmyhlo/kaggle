# lr_search base: 0.301
# lr_search acc1: 0.301
# lr_search acc2: 0.715
# lr_search acc5: 1.696
# lr_search acc10: 1.944

# SOFTMAX: metric: 0.8451, temp: 0.0108
# SIGMOID: metric: 0.8443, temp: 0.0174
# metric: 0.9137, temp: 0.0085

# base:                  metric: 0.8451, temp: 0.0108
# tta base:              metric: 0.8441, temp: 0.0026
# tta site avg:          metric: 0.9112, temp: 0.0010
# tta site avg low temp: metric: 0.9118, temp: 0.0011
# tta4 full:             metric: 0.9143, temp: 0.0002
# tta4 full sigmoid:     metric: 0.9137, temp: 0.0240
# tta4 full logits:      metric: 0.9134, temp: 0.0001

# crop-norm:             metric: 0.8602, temp: 0.0063

# crop-norm-la:          metric: 0.8586, temp: 0.0008
# crop-norm-la-avg:      metric: 0.9210, temp: 0.0233

# 1589, 3525


# TODO: predict group
# TODO: penalize errors not in group
# TODO: lsep within plate
# TODO: freeze BN
# TODO: exponential moving average of their parameters with a decay rate of 0.999
# TODO: use mixmatch-like distribution sharpening
# TODO: iteratively  optimize single batch on pseudo-labels
# TODO: fix 0.9 -> 0.92
# TODO: add tta for thresh search
# TODO: learn normalization
# TODO: learn image to extract from experiment images to normalize
# TODO: learn features to extract from experiment features to normalize
# TODO: mixup
# TODO: train longer
# TODO: plot outliers/error analysis
# TODO: better visualizations
# TODO: check train_emb.py Sampler corner cases
# TODO: check train_lsep.py Sampler corner cases
# TODO: normalize last layer (embeddings) by experiment
# TODO: use tta for oof thresh search, especially for PL generation
# TODO: reduce amount of pseudolabeled data with each epoch
# TODO: pl soft probs
# TODO: pl+mixup
# TODO: do not use softmax with lsep
# TODO: hinge loss
# TODO: lsep loss
# TODO: lsep within experiment
# TODO: train longer
# TODO: train second cycle
# TODO: use LAP to compute error/loss for update
# TODO: use LAP for OHEM
# TODO: 1cycle decay to zero?
# TODO: precompute ref embeddings (at each epoch) for ref models
# TODO: train longer
# TODO: b1,b2,b3
# TODO: pseudo labeling
# TODO: train on all data (with controls)
# TODO: compute stats/fold splits using train.csv + train_controls.csv
# TODO: better kfold split (mean image of each experiment)
# TODO: normalize by plate stats
# TODO: pretrain on huge dataset
# TODO: lookahead larger step size
# TODO: SEARCH THRESHOLD USING AVERAGE OF 2 SITES
# TODO: delta with ref + arcface
# TODO: try to learn diverse model
# TODO: CutMix on sites
# TODO: CutMix on all data
# TODO: progressive jitter
# TODO: use all controll files (train_controls.csv)
# TODO: start from hard aug and lower with each epoch
# TODO: continute from best cp
# TODO: huge batch rmsprop
# TODO: other losses, soft-f1, lsep, focal
# TODO: check last metric learning experiment loss plot
# TODO: even wider temp search space
# TODO: metric inverted/cyclic weight progress
# TODO: hints from cells telegram channel
# TODO: remove id_code and other data asserts
# TODO: wider temp search space for metric learning models
# TODO: progressive jitter
# TODO: metric learning and additional feats
# TODO: cyclic arcface weight
# TODO: always check relative plots
# TODO: metric learning separate branch
# TODO: triplet loss with all lessons learned from arcface
# TODO: label smoothing
# TODO: ref metric learning
# TODO: ref subtract on logits
# TODO: make ref and image close in space and ref and other image - far in space
# TODO: ref difference method with classification of both ref and image
# TODO: ref difference method with classification of both difference and output
# TODO: compute stats on all controls
# TODO: finetune with arcface
# TODO: knn classifier
# TODO: training less epochs
# TODO: check if sites can be stitched
# TODO: metric per type
# TODO: metric per experiment
# TODO: onecycle remove plat area
# TODO: random stitch sites
# TODO: distort aspect ratio
# TODO: well info
# TODO: autoaugment
# TODO: Population Based Training
# TODO: model diverse split
# TODO: well coding
# TODO: learn normalization per plate
# TODO: TVN (Typical Variation Normalization)
# TODO: normalize by ref (?)
# TODO: your network will need to evaluate your current image compared to each control or maybe a selection of it.
# TODO: random scale
# TODO: cyclic, clr, cawr
# TODO: correct tensorboard visualization
# TODO: correct color jitter
# TODO: visualize!!!
# TODO: speedup eval
# TODO: mix features for same class (multiple layers)
# TODO: relevant literature
# TODO: resized crop
# TODO: initialize the kernel properly in order to keep approximately the same variance that the original model had.
# TODO: learn closer to negative control and further to other batches
# TODO: compute norm stats only from controls
# TODO: batch/plate effects
# TODO: k shot learning
# TODO: https://github.com/recursionpharma/rxrx1-utils
# TODO: 2 images as control
# TODO: https://data.broadinstitute.org/bbbc/image_sets.html
# TODO: https://github.com/awesomedata/awesome-public-datasets#biology
# TODO: smarter split: https://www.kaggle.com/mariakesa/pcaforvisualizingbatcheffects
# TODO: different heads for different cell types
# TODO: mix sites

# TODO: concat pool
# TODO: gem pool


# TODO: parallel temp search
# TODO: eval site selection
# TODO: check predictions/targets name
# TODO: more fc layers for arcface
# TODO: pseudo labeling
# TODO: greedy assignment


# TODO: domain adaptation
# TODO: user other images stats
# TODO: other cyclic (1cycle) impl
# TODO: https://www.rxrx.ai/
# TODO: batch effects
# TODO: context modelling notes in rxrx
# TODO: generalization notes in rxrx
