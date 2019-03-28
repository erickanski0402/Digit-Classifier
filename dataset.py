import numpy as np

def prepare_dataset(path):
    data_file = open(path, "r")
    set = []

    for line in data_file.readlines():
        example = line.split(',')
        for i in range(len(example)):
            example[i] = int(example[i])
        set.append(example)

    data_file.close()
    return set

def zero_out_targets(outputs):
    return np.zeros(outputs) + 0.01

def set_desired_target(outputs, target):
    targets = zero_out_targets(outputs)
    targets[target] = 0.99
    return targets

def scale_training_set(training_set):
    return (np.array(training_set[1:]) / 255.0 * 0.99) + 0.01
