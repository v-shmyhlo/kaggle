# TODO: https://arxiv.org/abs/1906.07282
# TODO: find lr for la(k=0.5)
# TODO: ensemble different kfold split methods
# TODO: https://www.embopress.org/doi/pdf/10.15252/msb.20177551
# TODO: step 5 ewa and la
# TODO: pow-style temp on logits
# TODO: ewa mom 0.5
# TODO: https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/
# TODO: SWA
# TODO: Oleg's transform
# TODO: per image/per experiment correction
# TODO: CAM
# TODO: another padding mode
# TODO: brightness, contrast
# TODO: pl + mixup
# TODO: disable channel reweight
# TODO: pick temp search range
# TODO: mixmatch over predicted labels
# TODO: reweight experiment loss by metric
# TODO: ohem on samples outside of group
# TODO: check all boolean indexing
# TODO: ohem experiments with bad accuracy
# TODO: predict group
# TODO: penalize errors not in group
# TODO: iteratively unfreeze layers
# TODO: exponential moving average of their parameters with a decay rate of 0.999
# TODO: use mixmatch-like distribution sharpening for temp search
# TODO: learn image to extract from experiment images to normalize
# TODO: learn features to extract from experiment features to normalize
# TODO: plot outliers/error analysis
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
# TODO: train on all data (with controls)
# TODO: compute stats/fold splits using train.csv + train_controls.csv
# TODO: better kfold split (mean image of each experiment OR more representative images)
# TODO: pretrain on huge dataset
# TODO: try to learn diverse model
# TODO: use all controll files (train_controls.csv)
# TODO: even wider temp search space
# TODO: hints from cells telegram channel
# TODO: deterministic eval
# TODO: knn classifier
# TODO: Population Based Training
# TODO: model diverse split
# TODO: TVN (Typical Variation Normalization)
# TODO: normalize by ref (?)
# TODO: your network will need to evaluate your current image compared to each control or maybe a selection of it.
# TODO: speedup eval
# TODO: mix features for same class (multiple layers)
# TODO: relevant literature
# TODO: batch/plate effects
# TODO: k shot learning
# TODO: https://github.com/recursionpharma/rxrx1-utils
# TODO: https://data.broadinstitute.org/bbbc/image_sets.html
# TODO: https://github.com/awesomedata/awesome-public-datasets#biology
# TODO: smarter split: https://www.kaggle.com/mariakesa/pcaforvisualizingbatcheffects
# TODO: parallel temp search
# TODO: check predictions/targets name
# TODO: greedy assignment
# TODO: user other images stats
# TODO: https://www.rxrx.ai/
# TODO: batch effects
# TODO: context modelling notes in rxrx
# TODO: generalization notes in rxrx
# TODO: mixmatch in train mode
