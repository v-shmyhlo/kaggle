from config2 import Config as C

config = C(
    seed=42,
    epochs=10,
    model=C(),
    train=C(
        batch_size=128,
        optimizer=C(
            type='sgd',
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4,
            lookahead=C(
                lr=0.5,
                steps=5)),
        scheduler=C(
            type='cosine')),
    eval=C(
        batch_size=128))
