import optax

def adam(lr):
    optimizer = optax.adam(lr)
    return optimizer

def adam_clipping(lr, clip=1.0):
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip),
        optax.adam(lr)
    )
    return optimizer