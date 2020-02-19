from config2 import Config as C

config = C(
    seed=42,
    epochs=10,
    model=C(
        type='efficientnet-b1'),
    train=C(
        batch_size=128,
        label_smoothing=0.1,
        self_distillation=C(
            start_epoch=100,
            target_weight=0.5,
            pred_ewa=2 / 3),
        optimizer=C(
            type='sgd',
            lr=0.14,
            momentum=0.9,
            weight_decay=1e-4,
            lookahead=C(
                lr=0.5,
                steps=5)),
        scheduler=C(
            type='coswarm')),
    eval=C(
        batch_size=128))
