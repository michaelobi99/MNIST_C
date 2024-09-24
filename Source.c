#define _CRT_SECURE_NO_WARNINGS

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

typedef struct
{
	Layer hidden, output;
} Network;

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

void init_layer(Layer* layer, int in_size, int out_size) {
	int n = in_size * out_size;
	int i;
	float scale = sqrt(2.0f / in_size);
	layer->input_size = in_size;
	layer->output_size = out_size;
	layer->weights = malloc(n * sizeof(float));
	layer->weight_momentum = calloc(n, sizeof(float));
	layer->biases = calloc(out_size, sizeof(float));
	layer->bias_momentum = calloc(out_size, sizeof(float));

	for (i = 0; i < n; ++i) {
		layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
	}
}

//swap endianness of a 32 bit integer
int __builtin_swap32(int number) {
	number = ((number & 0x000000FF) << 24) | ((number & 0x0000FF00) << 8) | ((number & 0x00FF0000) >> 8) | ((number & 0xFF000000) >> 24);
	return number;
}

void read_mnist_images(const char* filename, unsigned char** images, int* nImages) {
	FILE* file = fopen(filename, "rb");
	int magic_number = 0, rows = 0, cols = 0;
	if (!file) exit(1);


	fread(&magic_number, sizeof(int), 1, file);
	magic_number = __builtin_swap32(magic_number);
	if (magic_number != MNIST_IMAGES_MAGIC_NUMBER)
		exit(1);
	fread(nImages, sizeof(int), 1, file);
	*nImages = __builtin_swap32(*nImages);

	fread(&rows, sizeof(int), 1, file);
	fread(&cols, sizeof(int), 1, file);

	rows = __builtin_swap32(rows);
	cols = __builtin_swap32(cols);
	*images = malloc((*nImages) * rows * cols, file);
	fread(*images, sizeof(unsigned char), (*nImages) * rows * cols, file);
	fclose(file);
	printf("got here\n");

}

void read_mnist_labels(const char* filename, unsigned char** labels, int* nLabels) {
	FILE* file = fopen(filename, "rb");
	int magic_number = 0;
	if (!file) exit(1);


	fread(&magic_number, sizeof(int), 1, file);
	magic_number = __builtin_swap32(magic_number);
	if (magic_number != MNIST_LABELS_MAGIC_NUMBER)
		exit(1);

	fread(nLabels, sizeof(int), 1, file);
	*nLabels = __builtin_swap32(*nLabels);
	*labels = malloc(*nLabels);
	fread(*labels, sizeof(unsigned char), *nLabels, file);
	fclose(file);
	printf("got here\n");

}

int main() {
	Network net;
	InputData data = { 0 };
	int epoch, i, j, k, correct;
	float learning_rate = LEARNING_RATE;
	float img[INPUT_SIZE];
	float total_loss;
	float hidden_output[HIDDEN_SIZE];
	float final_output[OUTPUT_SIZE];

	srand(time(NULL));

	init_layer(&net.hidden, INPUT_SIZE, HIDDEN_SIZE);
	init_layer(&net.output, HIDDEN_SIZE, OUTPUT_SIZE);

	//read MNIST Images

	read_mnist_images(TRAIN_IMG_PATH, &data.images, &data.nImages);
	read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &data.nLabels);
}