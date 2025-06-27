#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "src/ann.h"

int main(void) {
  MNISTSample *train_data = malloc(TRAIN_SIZE * sizeof(MNISTSample));
  MNISTSample *test_data = malloc(TEST_SIZE * sizeof(MNISTSample));
  load_mnist("MNIST/train-images-idx3-ubyte", "MNIST/train-labels-idx1-ubyte",
             train_data, TRAIN_SIZE);
  load_mnist("MNIST/t10k-images-idx3-ubyte", "MNIST/t10k-labels-idx1-ubyte",
             test_data, TEST_SIZE);

  NeuralNetwork nn;
  init_network(&nn);
  // pretrained file has 92.96% accuracy using:
  // load_network(&nn, "model_data/pretrained.bin");
  // validate_network(&nn, test_data, TEST_SIZE);

  train_network(&nn, train_data, TRAIN_SIZE);
  validate_network(&nn, test_data, TEST_SIZE);
  save_network(&nn, "model_data/model_parameters.bin");

  free(train_data);
  free(test_data);
  return 0;
}