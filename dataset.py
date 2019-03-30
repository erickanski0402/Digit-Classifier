import numpy as np

def getMnistData(path):
    # Open filestream at the file path provided
    data_file = open(path)
    # Reads the whole file into a list
    data_list = data_file.readlines()
    # Close datastream
    data_file.close()
    # Return list of values gotten from the file
    return data_list

def convertToNormalizedFloatArray(all_values):
    # Convert the list of strings into an array of floats (omitting the target
    #   value at the beginning of the list) and normalize the values from 0-255
    #   to 0-1
    return (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

def getTargets(output_nodes, all_values):
    # Define target list with each value as low as possible (0.01)
    targets = np.zeros(output_nodes) + 0.01
    # Set the index of the desired target to be as high as possible (0.99)
    targets[int(all_values[0])] = 0.99
    # return the list of targets
    return targets
