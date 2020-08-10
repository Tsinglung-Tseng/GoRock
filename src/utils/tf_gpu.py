import tensorflow as tf


def USEGPU(USING):
    USING = 0
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
        try:
            tf.config.experimental.set_visible_devices(physical_devices[USING], "GPU")
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(physical_devices),
                "Physical GPUs,",
                len(logical_gpus),
                "Logical GPU",
            )
            tf.config.experimental.set_memory_growth(physical_devices[USING], True)
        except RuntimeError as e:
            print(e)

    try:
        tf.config.experimental.set_memory_growth(physical_devices[USING], True)
    except:
        pass
