import numpy as np
import matplotlib.pyplot as plt
import dataset as ds
import neural_network as network

# Define the number of nodes for each layer
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
# Define the learning rate for the network
learning_rate = 0.3
# Initialize the neural network
nn = network.neural_network(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Get the list of training data from mnist_train_100.csv
training_data_list = ds.getMnistData("./mnist_dataset/mnist_train_100.csv")


# Trains the network on every single training example once
for record in training_data_list:
    # Split each value in the record into a new list
    all_values = record.split(',')
    # Convert the list of record values into a normalized list of floats
    inputs = ds.convertToNormalizedFloatArray(all_values)
    # Define target array with the desired target value assigned to 0.99
    targets = ds.getTargets(output_nodes, all_values)
    # Train the network based on the single example just prepared
    nn.train(inputs, targets)
    pass


# Get the test data from mnist_test_10.csv
test_data_list = ds.getMnistData("./mnist_dataset/mnist_test_10.csv")

# Define the scorecard for gauging success
scorecard = []

# Tests the trained network against the testing set
for record in test_data_list:
    # Split each value in the record into a new list
    all_values = record.split(',')

    # Get and print the correct label from the record
    correct_label = int(all_values[0])
    print(correct_label, "correct label")

    # Convert the record list into a normalized list of floats
    inputs = ds.convertToNormalizedFloatArray(all_values)
    # Run the feed-forward algorithm to determine what the neural network 'thinks'
    #   the output should be
    outputs = nn.query(inputs)
    # (Meaning the highest value of the list of outputs)
    label = np.argmax(outputs)

    # Append a good value (1) or a bad value (0) based on the neural networks success rate
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

# Print the scorecard and its relative success rate
print(scorecard)
scorecard_array = np.asarray(scorecard)
print("performance = ", scorecard_array.sum() / len(scorecard_array))
