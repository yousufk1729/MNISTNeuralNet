#ifndef ANN_H
#define ANN_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 16
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
#define EPOCHS 25
#define LEARNING_RATE 0.025
#define BATCH_SIZE 20

typedef struct {
  // Weights/biases
  double W1[INPUT_SIZE][HIDDEN_SIZE];
  double b1[HIDDEN_SIZE];
  double W2[HIDDEN_SIZE][OUTPUT_SIZE];
  double b2[OUTPUT_SIZE];

  // Activations/pre-activations
  double a0[INPUT_SIZE];
  double z1[HIDDEN_SIZE];
  double a1[HIDDEN_SIZE];
  double z2[OUTPUT_SIZE];
  double a2[OUTPUT_SIZE];

  // a2 needs to match this!
  double target[OUTPUT_SIZE];

  // Gradient accumulators for mini-batch
  double dW1[INPUT_SIZE][HIDDEN_SIZE];
  double db1[HIDDEN_SIZE];
  double dW2[HIDDEN_SIZE][OUTPUT_SIZE];
  double db2[OUTPUT_SIZE];
} NeuralNetwork;

typedef struct {
  unsigned char image[INPUT_SIZE];
  unsigned char label;
} MNISTSample;

// MNIST Data
unsigned int reverse_int(unsigned int i);
void load_mnist(const char *image_file, const char *label_file,
                MNISTSample *samples, int num_samples);

// Model parameters
void init_network(NeuralNetwork *nn);

// Training/testing
void train_network(NeuralNetwork *nn, MNISTSample *train_data, int train_size);
void validate_network(NeuralNetwork *nn, MNISTSample *test_data, int test_size);

// SGD
void forward_pass(NeuralNetwork *nn, double *input);
void backward_pass(NeuralNetwork *nn, int target_label);
void apply_gradients(NeuralNetwork *nn);
double relu(double x);
double relu_derivative(double x);
void normalize_input(unsigned char *raw_input, double *normalized_input);
void shuffle_data(MNISTSample *data, int size);
void create_one_hot(int label, double *one_hot);
void softmax(double *input, double *output, int size);
int argmax(double *array, int size);
double mse_loss(double *predicted, double *target);

// Saving model parameters
void load_network(NeuralNetwork *nn, const char *filename);
void save_network(NeuralNetwork *nn, const char *filename);

#endif