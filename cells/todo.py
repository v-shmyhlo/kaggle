# TODO: brightness, contrast
# TODO: pl + mixup
# TODO: disable channel reweight
# TODO: pick temp search range
# TODO: mixmatch over predicted labels
# TODO: contrast
# TODO: ohem on samples outside of group
# TODO: check all boolean indexing
# TODO: predict group
# TODO: penalize errors not in group
# TODO: freeze BN
# TODO: iteratively unfreeze layers
# TODO: exponential moving average of their parameters with a decay rate of 0.999
# TODO: use mixmatch-like distribution sharpening for temp search
# TODO: iteratively optimize single batch on pseudo-labels
# TODO: learn normalization
# TODO: learn image to extract from experiment images to normalize
# TODO: learn features to extract from experiment features to normalize
# TODO: train longer
# TODO: plot outliers/error analysis
# TODO: better visualizations
# TODO: check train_emb.py Sampler corner cases
# TODO: check train_lsep.py Sampler corner cases
# TODO: pl + label smoothing
# TODO: normalize last layer (embeddings) by experiment
# TODO: train longer
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
# TODO: try to learn diverse model
# TODO: use all controll files (train_controls.csv)
# TODO: start from hard aug and lower with each epoch
# TODO: even wider temp search space
# TODO: hints from cells telegram channel
# TODO: cyclic arcface weight
# TODO: triplet loss with all lessons learned from arcface
# TODO: deterministic eval
# TODO: label smoothing
# TODO: make ref and image close in space and ref and other image - far in space
# TODO: knn classifier
# TODO: well info
# TODO: autoaugment
# TODO: Population Based Training
# TODO: model diverse split
# TODO: well coding
# TODO: TVN (Typical Variation Normalization)
# TODO: normalize by ref (?)
# TODO: your network will need to evaluate your current image compared to each control or maybe a selection of it.
# TODO: speedup eval
# TODO: mix features for same class (multiple layers)
# TODO: relevant literature
# TODO: initialize the kernel properly in order to keep approximately the same variance that the original model had.
# TODO: batch/plate effects
# TODO: k shot learning
# TODO: https://github.com/recursionpharma/rxrx1-utils
# TODO: 2 images as control
# TODO: https://data.broadinstitute.org/bbbc/image_sets.html
# TODO: https://github.com/awesomedata/awesome-public-datasets#biology
# TODO: smarter split: https://www.kaggle.com/mariakesa/pcaforvisualizingbatcheffects
# TODO: different heads for different cell types
# TODO: parallel temp search
# TODO: eval site selection
# TODO: check predictions/targets name
# TODO: greedy assignment
# TODO: user other images stats
# TODO: https://www.rxrx.ai/
# TODO: batch effects
# TODO: context modelling notes in rxrx
# TODO: generalization notes in rxrx
