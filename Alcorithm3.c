void steeringFilter(float* space_filt, float* R, int nxs, int nOrthogonal, int nSpat_filters, int nTemp_filters, int nFrames, int L) {
    for (int x = 0; x < nxs; x++) {
        for (int oo = 0; oo < nOrthogonal; oo++) {
            for (int sf = 0; sf < nSpat_filters; sf++) {
                for (int tf = 0; tf < nTemp_filters; tf++) {
                    for (int fr = 0; fr <= nFrames - L; fr++) {
                        // Calculate index in space_filt
                        int index = (((oo * nSpat_filters + sf) * nTemp_filters + tf) * nFrames + fr) * nxs + x;
                        float I = space_filt[index];

                        // Perform the operation [G^x tensor I]
                        // For simplicity, assume G^x is a scalar value
                        float Gx = 1.0f; // Example scalar value for G^x

                        float result = Gx * I; // Perform element-wise multiplication

                        // Calculate index in R
                        int index_R = (((x * nOrthogonal + oo) * nSpat_filters + sf) * nTemp_filters + tf) * nFrames + fr;
                        R[index_R] = result;
                    }
                }
            }
        }
    }
}

// Calculate the tensor product of two matrices
void tensor_product(float* A, int rows_A, int cols_A, float* B, int rows_B, int cols_B, float* C) {
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_A; j++) {
            for (int k = 0; k < rows_B; k++) {
                for (int l = 0; l < cols_B; l++) {
                    int row = i * rows_B + k;
                    int col = j * cols_B + l;
                    C[row * (cols_A * cols_B) + col] = A[i * cols_A + j] * B[k * cols_B + l];
                }
            }
        }
    }
}
