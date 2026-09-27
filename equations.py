import numpy as np

PARAMETER_COUNT = 1_039_968
KERNEL_LAUNCHES = 17


def flops(image_size, batch):
    return 17_714.0 * batch * image_size**2 + 313_344.0 * batch


def memory(image_size, batch):
    return 4.0 * (PARAMETER_COUNT + 26.0 * batch * image_size**2 + 868.0 * batch)


def latency(image_size, batch, theta):
    memory_time = bytes_moved(image_size, batch) / theta["bandwidth_bytes_per_s"]
    compute_time = flops(image_size, batch) / theta["throughput_flops_per_s"]
    return KERNEL_LAUNCHES * theta["tau_s"] + np.maximum(memory_time, compute_time)


def energy(image_size, batch, theta_energy):
    return (
        theta_energy["idle_power_w"]
        * latency(image_size, batch, theta_energy["latency"])
        + theta_energy["joules_per_flop"] * flops(image_size, batch)
        + theta_energy["joules_per_byte"] * bytes_moved(image_size, batch)
    )


def bytes_moved(image_size, batch):
    return 4.0 * (91.0 * batch * image_size**2 + 2_148.0 * batch + PARAMETER_COUNT)
