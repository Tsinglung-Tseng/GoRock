import tensorflow as tf


def USEGPU(USING):
    USING = 0
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
        try:
            physical_devices = tf.config.list_physical_devices("GPU")
            tf.config.set_visible_devices(physical_devices[USING], "GPU")
            tf.config.experimental.set_memory_growth(physical_devices[USING], True)
            print(len(physical_devices), "Physical GPUs, using", USING)
            tf.config.experimental.set_memory_growth(physical_devices[USING], True)
        except RuntimeError as e:
            print(e)
