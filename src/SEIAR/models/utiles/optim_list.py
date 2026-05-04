import optax

def adam(lr):
    return optax.adam(lr)

def adam_clipping(lr, clip=1.0):
    return optax.chain(
        optax.clip_by_global_norm(clip),
        optax.adam(lr)
    )