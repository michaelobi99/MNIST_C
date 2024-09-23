#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define HIDDEN_SIZE 256
#define LEARNING_RATE 0.001f
#define MOMENTUM 0.9f
#define L2_LAMBDA 0.0001f
#define EPOCHS 20
#define BATCH_SIZE 64
#define IMAGE_SIZE 28
#define TRAIN_SPLIT 0.8
#define MNIST_IMAGES_MAGIC_NUMBER 2051
#define MNIST_LABELS_MAGIC_NUMBER 2049

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

typedef struct {
	float* weights, * biases, * weight_momentum, * bias_momentum;
	int input_size, output_size;
} Layer;

typedef struct {
	unsigned char* images, * labels;
	int nImages, nLabels;
} InputData;

float relu(float x) {
	return x > 0 ? x : 0;
}

float relu_derivative(float x) {
	return x > 0 ? 1 : 0;
}

void softmax(float* input, float* output, int size) {
	float sum = 0;
	int i;
	for (i = 0; i < size; ++i) {
		output[i] = expf(input[i]);
		sum += output[i];
	}
	for (i = 0; i < size; ++i)
		output[i] /= sum;
}

int main() {

}