#include <stdio.h>

// Function to calculate binomial coefficient
int binomial_coefficient(int n, int k) {
    int res = 1;
    if (k > n - k)
        k = n - k;
    for (int i = 0; i < k; ++i) {
        res *= (n - i);
        res /= (i + 1);
    }
    return res;
}

// Function to calculate the convolution of two arrays
void convolution(double *a, int size_a, double *b, int size_b, double *result) {
    // Assuming size_a >= size_b
    int size_result = size_a;
    for (int i = 0; i < size_result; ++i) {
        result[i] = 0.0;
        for (int j = 0; j <= i && j < size_b; ++j) {
            result[i] += a[i - j] * b[j];
        }
    }
}

int main() {
    int n = 3; // Example value for n
    int m = 2; // Example value for m
    double Dx = 1.0; // Example value for Dx
    double Dy = 1.0; // Example value for Dy
    double theta = 45.0; // Example value for theta
    double G0 = 1.0; // Example value for G0

    // Calculate the size of the arrays
    int size_a = n + 1;
    int size_b = m + 1;
    int size_result = size_a + size_b - 1;

    // Allocate memory for the arrays
    double a[size_a];
    double b[size_b];
    double result[size_result];

    // Calculate the arrays
    for (int k = 0; k <= n; ++k) {
        a[k] = binomial_coefficient(n, k) * pow(Dx * cos(theta), k) * pow(Dy * sin(theta), n - k);
    }

    for (int i = 0; i <= m; ++i) {
        b[i] = binomial_coefficient(m, i) * pow(-Dx * sin(theta), i) * pow(Dy * cos(theta), m - i);
    }

    // Calculate the convolution
    convolution(a, size_a, b, size_b, result);

    // Scale the result by G0
    for (int i = 0; i < size_result; ++i) {
        result[i] *= G0;
    }

    // Print the result
    for (int i = 0; i < size_result; ++i) {
        printf("%f\n", result[i]);
    }

    return 0;
}
