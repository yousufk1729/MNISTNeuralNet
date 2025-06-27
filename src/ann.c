#include "ann.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

unsigned int reverse_int(unsigned int i) {
  unsigned char c1 = (unsigned char)(i & 0xFF);
  unsigned char c2 = (unsigned char)((i >> 8) & 0xFF);
  unsigned char c3 = (unsigned char)((i >> 16) & 0xFF);
  unsigned char c4 = (unsigned char)((i >> 24) & 0xFF);

  return ((unsigned int)c1 << 24) | ((unsigned int)c2 << 16) |
         ((unsigned int)c3 << 8) | (unsigned int)c4;
}

void load_mnist(const char *image_file, const char *label_file,
                MNISTSample *samples, int num_samples) {
  FILE *img_fp = fopen(image_file, "rb");
  FILE *lbl_fp = fopen(label_file, "rb");

  unsigned int magic, num_images, rows, cols;
  fread(&magic, sizeof(magic), 1, img_fp);
  fread(&num_images, sizeof(num_images), 1, img_fp);
  fread(&rows, sizeof(rows), 1, img_fp);
  fread(&cols, sizeof(cols), 1, img_fp);

  magic = reverse_int(magic);
  num_images = reverse_int(num_images);
  rows = reverse_int(rows);
  cols = reverse_int(cols);

  unsigned int lbl_magic, num_labels;
  fread(&lbl_magic, sizeof(lbl_magic), 1, lbl_fp);
  fread(&num_labels, sizeof(num_labels), 1, lbl_fp);

  lbl_magic = reverse_int(lbl_magic);
  num_labels = reverse_int(num_labels);

  for (int i = 0; i < num_samples; ++i) {
    fread(samples[i].image, sizeof(unsigned char), INPUT_SIZE, img_fp);
    fread(&samples[i].label, sizeof(unsigned char), 1, lbl_fp);
  }

  fclose(img_fp);
  fclose(lbl_fp);
}

void init_network(NeuralNetwork *nn) {
  srand((unsigned int)time(NULL));

  // He uniform: weights = u(-sqrt(6/fan_in), sqrt(6/fan_in))
  double w1_scale = sqrt(6.0 / INPUT_SIZE);
  double w2_scale = sqrt(6.0 / HIDDEN_SIZE);

  for (int i = 0; i < INPUT_SIZE; ++i) {
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
      nn->W1[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * w1_scale;
    }
  }

  for (int i = 0; i < HIDDEN_SIZE; ++i) {
    nn->b1[i] = 0.0;
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
      nn->W2[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * w2_scale;
    }
  }

  for (int i = 0; i < OUTPUT_SIZE; ++i) {
    nn->b2[i] = 0.0;
  }
}

void train_network(NeuralNetwork *nn, MNISTSample *train_data, int train_size) {
  printf("\nTraining...\n");
  printf("Batch Size: %d\n", BATCH_SIZE);
  printf("Learning Rate: %.3f\n", LEARNING_RATE);
  printf("Epochs: %d\n", EPOCHS);

  int num_batches = train_size / BATCH_SIZE;

  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    printf("Epoch %d/%d\n", epoch + 1, EPOCHS);
    shuffle_data(train_data, train_size);

    double epoch_loss = 0.0;
    int correct = 0;

    for (int batch = 0; batch < num_batches; ++batch) {
      memset(nn->dW1, 0, sizeof(nn->dW1));
      memset(nn->db1, 0, sizeof(nn->db1));
      memset(nn->dW2, 0, sizeof(nn->dW2));
      memset(nn->db2, 0, sizeof(nn->db2));

      for (int i = batch * BATCH_SIZE; i < batch * BATCH_SIZE + BATCH_SIZE;
           ++i) {
        double normalized_input[INPUT_SIZE];
        normalize_input(train_data[i].image, normalized_input);
        forward_pass(nn, normalized_input);

        double target_one_hot[OUTPUT_SIZE];
        create_one_hot(train_data[i].label, target_one_hot);
        epoch_loss += mse_loss(nn->a2, target_one_hot);

        int predicted = argmax(nn->a2, OUTPUT_SIZE);
        if (predicted == train_data[i].label) {
          correct++;
        }

        backward_pass(nn, train_data[i].label);
      }

      apply_gradients(nn);

      if ((batch + 1) % 1000 == 0) {
        printf("  Processed %d/%d batches\n", batch + 1, num_batches);
      }
    }

    printf("Training Accuracy: %.2f%% (%d/%d), MSE Loss: %.6f\n",
           (double)correct / train_size * 100, correct, train_size,
           epoch_loss / train_size);
  }
}

void validate_network(NeuralNetwork *nn, MNISTSample *test_data,
                      int test_size) {
  int correct = 0;
  double total_loss = 0.0;

  for (int i = 0; i < test_size; ++i) {
    double normalized_input[INPUT_SIZE];
    normalize_input(test_data[i].image, normalized_input);
    forward_pass(nn, normalized_input);

    double target_one_hot[OUTPUT_SIZE];
    create_one_hot(test_data[i].label, target_one_hot);
    total_loss += mse_loss(nn->a2, target_one_hot);

    int predicted = argmax(nn->a2, OUTPUT_SIZE);
    if (predicted == test_data[i].label) {
      correct++;
    }
  }

  printf("Validation Accuracy: %.2f%% (%d/%d), MSE Loss: %.6f\n",
         (double)correct / test_size * 100, correct, test_size,
         total_loss / test_size);
}

void forward_pass(NeuralNetwork *nn, double *input) {
  // Taken from
  // https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras
  // Copy input into a0
  memcpy(nn->a0, input, INPUT_SIZE * sizeof(double));

  // z1 = W1^T * a0 + b1
  // a1 = ReLU(z1)
  for (int j = 0; j < HIDDEN_SIZE; ++j) {
    nn->z1[j] = nn->b1[j];
    for (int i = 0; i < INPUT_SIZE; ++i) {
      nn->z1[j] += nn->a0[i] * nn->W1[i][j];
    }
    nn->a1[j] = relu(nn->z1[j]);
  }

  // z2 = W2^T * a1 + b2
  // a2 = softmax(z2)
  for (int k = 0; k < OUTPUT_SIZE; ++k) {
    nn->z2[k] = nn->b2[k];
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
      nn->z2[k] += nn->a1[j] * nn->W2[j][k];
    }
  }
  softmax(nn->z2, nn->a2, OUTPUT_SIZE);
}

void backward_pass(NeuralNetwork *nn, int target_label) {
  // Taken from
  // https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras
  create_one_hot(target_label, nn->target);

  // Output layer gradients (layer 2)
  // dZ[2] = A[2] - Y (for softmax with MSE loss)
  double dz2[OUTPUT_SIZE];
  for (int k = 0; k < OUTPUT_SIZE; k++) {
    dz2[k] = (nn->a2[k] - nn->target[k]);
  }

  // dW[2] = (1/m) * A[1]^T * dZ[2]
  // Accumulate weight gradients for output layer
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    for (int k = 0; k < OUTPUT_SIZE; k++) {
      nn->dW2[j][k] += nn->a1[j] * dz2[k];  // A[1] (hidden activations) * dZ[2]
    }
  }

  // db[2] = (1/m) * sum(dZ[2])
  // Accumulate bias gradients for output layer
  for (int k = 0; k < OUTPUT_SIZE; k++) {
    nn->db2[k] += dz2[k];
  }

  // Hidden layer gradients (layer 1)
  // dZ[1] = W[2]^T * dZ[2] * g'(Z[1]) where g'(Z[1]) is ReLU derivative
  double dz1[HIDDEN_SIZE];
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    // Calculate W[2]^T * dZ[2]: backpropagate error from output to hidden layer
    double sum = 0.0;
    for (int k = 0; k < OUTPUT_SIZE; k++) {
      sum += nn->W2[j][k] * dz2[k];  // W2[j][k] * dz2[k]
    }
    // Multiply by ReLU derivative: g'(z1[j])
    dz1[j] = sum * relu_derivative(nn->z1[j]);
  }

  // dW[1] = (1/m) * A[0]^T * dZ[1]
  // Accumulate weight gradients for hidden layer
  for (int i = 0; i < INPUT_SIZE; i++) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      nn->dW1[i][j] += nn->a0[i] * dz1[j];  // A[0] (input) * dZ[1]
    }
  }

  // db[1] = (1/m) * sum(dZ[1])
  // Accumulate bias gradients for hidden layer
  for (int j = 0; j < HIDDEN_SIZE; ++j) {
    nn->db1[j] += dz1[j];
  }
}

void apply_gradients(NeuralNetwork *nn) {
  // Taken from
  // https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras
  for (int i = 0; i < INPUT_SIZE; ++i) {
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
      nn->W1[i][j] -= LEARNING_RATE * nn->dW1[i][j];
    }
  }

  for (int j = 0; j < HIDDEN_SIZE; ++j) {
    nn->b1[j] -= LEARNING_RATE * nn->db1[j];
  }

  for (int j = 0; j < HIDDEN_SIZE; ++j) {
    for (int k = 0; k < OUTPUT_SIZE; ++k) {
      nn->W2[j][k] -= LEARNING_RATE * nn->dW2[j][k];
    }
  }

  for (int k = 0; k < OUTPUT_SIZE; ++k) {
    nn->b2[k] -= LEARNING_RATE * nn->db2[k];
  }
}

double relu(double x) { return x > 0 ? x : 0; }

double relu_derivative(double x) { return x > 0 ? 1 : 0; }

void normalize_input(unsigned char *raw_input, double *normalized_input) {
  for (int i = 0; i < INPUT_SIZE; ++i) {
    normalized_input[i] = raw_input[i] / 255.0;
  }
}

void shuffle_data(MNISTSample *data, int size) {
  for (int i = size - 1; i > 0; --i) {
    int j = rand() % (i + 1);
    MNISTSample temp = data[i];
    data[i] = data[j];
    data[j] = temp;
  }
}

void create_one_hot(int label, double *one_hot) {
  for (int i = 0; i < OUTPUT_SIZE; ++i) {
    one_hot[i] = (i == label) ? 1.0 : 0.0;
  }
}

void softmax(double *input, double *output, int size) {
  double max_val = input[0];
  for (int i = 1; i < size; ++i) {
    if (input[i] > max_val) max_val = input[i];
  }

  double sum = 0.0;
  for (int i = 0; i < size; ++i) {
    output[i] = exp(input[i] - max_val);
    sum += output[i];
  }

  for (int i = 0; i < size; ++i) {
    output[i] /= sum;
  }
}

int argmax(double *array, int size) {
  int max_idx = 0;
  for (int i = 1; i < size; ++i) {
    if (array[i] > array[max_idx]) {
      max_idx = i;
    }
  }
  return max_idx;
}

double mse_loss(double *predicted, double *target) {
  double sum = 0.0;
  for (int i = 0; i < OUTPUT_SIZE; ++i) {
    double diff = predicted[i] - target[i];
    sum += diff * diff;
  }
  return sum / OUTPUT_SIZE;
}

void load_network(NeuralNetwork *nn, const char *filename) {
  FILE *fp = fopen(filename, "rb");
  fread(nn->W1, sizeof(double), INPUT_SIZE * HIDDEN_SIZE, fp);
  fread(nn->b1, sizeof(double), HIDDEN_SIZE, fp);
  fread(nn->W2, sizeof(double), HIDDEN_SIZE * OUTPUT_SIZE, fp);
  fread(nn->b2, sizeof(double), OUTPUT_SIZE, fp);
  fclose(fp);
}

void save_network(NeuralNetwork *nn, const char *filename) {
  FILE *fp = fopen(filename, "wb");
  fwrite(nn->W1, sizeof(double), INPUT_SIZE * HIDDEN_SIZE, fp);
  fwrite(nn->b1, sizeof(double), HIDDEN_SIZE, fp);
  fwrite(nn->W2, sizeof(double), HIDDEN_SIZE * OUTPUT_SIZE, fp);
  fwrite(nn->b2, sizeof(double), OUTPUT_SIZE, fp);
  fclose(fp);
}
