import numpy as np
import matplotlib.pyplot as plt
import dataset as ds

training_set = ds.prepare_dataset("./mnist_dataset/mnist_train_100.csv")
testing_set = ds.prepare_dataset("./mnist_dataset/mnist_test_10.csv")

# image_array = np.array(training_set[0][1:]).reshape((28,28))
# plt.imshow(image_array, cmap='Greys', interpolation='None')
# plt.waitforbuttonpress()
#   Visualization of single digit in 28x28 grid of grayscale pixels

scaled_training_set = ds.scale_training_set(training_set)

output_nodes = 10
targets = ds.set_desired_target(output_nodes, training_set[0][0])
print(targets)
