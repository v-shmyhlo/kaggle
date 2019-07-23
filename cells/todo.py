# lr_search base: 0.301
# lr_search acc1: 0.301
# lr_search acc2: 0.715
# lr_search acc5: 1.696
# lr_search acc10: 1.944

# SOFTMAX: metric: 0.8451, temp: 0.0108
# SIGMOID: metric: 0.8443, temp: 0.0174
# metric: 0.9137, temp: 0.0085


# TODO: infer with sigmoid
# TODO: remove id_code and other data asserts
# TODO: valid rotate
# TODO: wider temp search space for metric learning models
# TODO: metric learning and additional feats
# TODO: huge batch
# TODO: prog arc 0.2
# TODO: prog arc 1.0
# TODO: cyclic arcface weight
# TODO: progressive arcface
# TODO: stabilize arcface
# TODO: progressive resize for lr search
# TODO: always check relative plots
# TODO: metric learning separate branch
# TODO: triplet loss with all lessons learned from arcface
# TODO: log temp search metric
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
# TODO: fix rotation order
# TODO: metric per type
# TODO: metric per experiment
# TODO: onecycle remove plat area
# TODO: no cycle beta
# TODO: b2/b3
# TODO: shift / scale
# TODO: random stitch sites
# TODO: distort aspect ratio
# TODO: well info
# TODO: cutout
# TODO: autoaugment
# TODO: Population Based Training
# TODO: binarize image?
# TODO: do not cycle beta
# TODO: model diverse split
# TODO: well coding
# TODO: learn normalization per plate
# TODO: triplet loss
# TODO: TVN (Typical Variation Normalization)
# TODO: normalize by ref (?)
# TODO: label smoothing
# TODO: diverse model
# TODO: your network will need to evaluate your current image compared to each control or maybe a selection of it.
# TODO: random scale
# TODO: cyclic, clr, cawr
# TODO: correct tensorboard visualization
# TODO: correct color jitter
# TODO: visualize!!!
# TODO: speedup eval
# TODO: mix features for same class (multiple layers)
# TODO: relevant literature
# TODO: gradacum
# TODO: resized crop
# TODO: initialize the kernel properly in order to keep approximately the same variance that the original model had.
# TODO: learn closer to negative control and further to other batches
# TODO: batch/plate effects
# TODO: k shot learning
# TODO: https://github.com/recursionpharma/rxrx1-utils
# TODO: 2 images as control
# TODO: https://data.broadinstitute.org/bbbc/image_sets.html
# TODO: https://github.com/awesomedata/awesome-public-datasets#biology
# TODO: use all controls for training
# TODO: smarter split: https://www.kaggle.com/mariakesa/pcaforvisualizingbatcheffects
# TODO: different heads for different cell types
# TODO: mix sites
# TODO: concat pool
# TODO: greedy assign
# TODO: gem pool
# TODO: deep supervision
# TODO: parallel temp search
# TODO: eval site selection
# TODO: check predictions/targets name
# TODO: more fc layers for arcface
# TODO: pseudo labeling
# TODO: greedy assignment
# TODO: better minimum for lr
# TODO: eval with tta?
# TODO: tta
# TODO: val tta (sites)
# TODO: lr schedules
# TODO: allow shuffle of plate refs within experiment
# TODO: domain adaptation
# TODO: user other images stats
# TODO: cutout
# TODO: other cyclic (1cycle) impl
# TODO: focal
# TODO: https://www.rxrx.ai/
# TODO: batch effects
# TODO: context modelling notes in rxrx
# TODO: generalization notes in rxrx
