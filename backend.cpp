#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>
#include <omp.h>

extern "C" {

    // --- Memory & Utils ---
    // We rely on Python to manage memory allocation to keep C++ stateless and simple.
    // Random initialization (Xavier/He)
    void init_weights(float* data, int size, int fan_in, int fan_out,int seed) {
        std::mt19937 gen(seed);
        // He Initialization
        float limit = sqrt(2.0f / (float)fan_in);
        std::normal_distribution<float> dist(0.0, limit);
        for (int i = 0; i < size; i++) {
            data[i] = dist(gen);
        }
    }

    // --- Layers (Forward & Backward) ---

    // 1. Fully Connected (Linear)
    // Forward: Y = X @ W + B
    // X: (B, I), W: (I, O), Output: (B, O)
    void linear_forward(float* x, float* w, float* b, float* out, int batch, int in_feat, int out_feat) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < out_feat; j++) {
                float sum = (b) ? b[j] : 0.0f;
                for (int k = 0; k < in_feat; k++) {
                    sum += x[i * in_feat + k] * w[k * out_feat + j];
                }
                out[i * out_feat + j] = sum;
            }
        }
    }

    // Backward: dL/dX = dL/dY @ W.T, dL/dW = X.T @ dL/dY
    void linear_backward(float* x, float* w, float* dout, float* dx, float* dw, float* db, int batch, int in_feat, int out_feat) {
        // dW
        #pragma omp parallel for collapse(2)
        for (int r = 0; r < in_feat; r++) {
            for (int c = 0; c < out_feat; c++) {
                float sum = 0.0f;
                for (int i = 0; i < batch; i++) {
                    sum += x[i * in_feat + r] * dout[i * out_feat + c];
                }
                dw[r * out_feat + c] += sum; // Accumulate gradients
            }
        }
        // dB
        #pragma omp parallel for
        for (int c = 0; c < out_feat; c++) {
            float sum = 0.0f;
            for (int i = 0; i < batch; i++) {
                sum += dout[i * out_feat + c];
            }
            db[c] += sum;
        }
        // dX
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < batch; i++) {
            for (int k = 0; k < in_feat; k++) {
                float sum = 0.0f;
                for (int j = 0; j < out_feat; j++) {
                    sum += dout[i * out_feat + j] * w[k * out_feat + j];
                }
                dx[i * in_feat + k] = sum;
            }
        }
    }

    // 2. Convolution 2D (Naive implementation to satisfy "from scratch")
    // Input: (N, C, H, W)
    void conv2d_forward(float* x, float* w, float* b, float* out, 
                        int N, int C_in, int H, int W, 
                        int C_out, int K, int S, int P) {
        
        int H_out = (H + 2 * P - K) / S + 1;
        int W_out = (W + 2 * P - K) / S + 1;

        #pragma omp parallel for collapse(2)
        for (int n = 0; n < N; n++) {
            for (int c_out = 0; c_out < C_out; c_out++) {
                for (int h_out = 0; h_out < H_out; h_out++) {
                    for (int w_out = 0; w_out < W_out; w_out++) {
                        
                        float sum = (b) ? b[c_out] : 0.0f;
                        int h_start = h_out * S - P;
                        int w_start = w_out * S - P;

                        for (int c_in = 0; c_in < C_in; c_in++) {
                            for (int i = 0; i < K; i++) {
                                for (int j = 0; j < K; j++) {
                                    int h_in = h_start + i;
                                    int w_in = w_start + j;
                                    
                                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                        // Indexing logic
                                        int x_idx = ((n * C_in + c_in) * H + h_in) * W + w_in;
                                        int w_idx = ((c_out * C_in + c_in) * K + i) * K + j;
                                        sum += x[x_idx] * w[w_idx];
                                    }
                                }
                            }
                        }
                        int out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                        out[out_idx] = sum;
                    }
                }
            }
        }
    }

    // --- Add this after conv2d_forward ---

    // Backward: dL/dX, dL/dW, dL/db
    void conv2d_backward(float* x, float* w, float* dout, float* dx, float* dw, float* db,
                         int N, int C_in, int H, int W, 
                         int C_out, int K, int S, int P) {
        
        int H_out = (H + 2 * P - K) / S + 1;
        int W_out = (W + 2 * P - K) / S + 1;

        // 1. Gradient w.r.t Bias (db)
        #pragma omp parallel for
        for (int c_out = 0; c_out < C_out; c_out++) {
            float sum = 0.0f;
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H_out; h++) {
                    for (int w_out = 0; w_out < W_out; w_out++) {
                        int out_idx = ((n * C_out + c_out) * H_out + h) * W_out + w_out;
                        sum += dout[out_idx];
                    }
                }
            }
            db[c_out] += sum;
        }

        // 2. Gradient w.r.t Weights (dw)
        #pragma omp parallel for collapse(2)
        for (int c_out = 0; c_out < C_out; c_out++) {
            for (int c_in = 0; c_in < C_in; c_in++) {
                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        float sum = 0.0f;
                        for (int n = 0; n < N; n++) {
                            for (int h_out = 0; h_out < H_out; h_out++) {
                                for (int w_out = 0; w_out < W_out; w_out++) {
                                    int h_in = h_out * S - P + kh;
                                    int w_in = w_out * S - P + kw;
                                    
                                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                        int x_idx = ((n * C_in + c_in) * H + h_in) * W + w_in;
                                        int out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                                        sum += x[x_idx] * dout[out_idx];
                                    }
                                }
                            }
                        }
                        int w_idx = ((c_out * C_in + c_in) * K + kh) * K + kw;
                        dw[w_idx] += sum;
                    }
                }
            }
        }

        // 3. Gradient w.r.t Input (dx)
        // Initialize dx to 0 first (done in Python usually, but ensures safety)
        // Note: We parallelize over N and C_in to avoid race conditions on dx
        #pragma omp parallel for collapse(2)
        for (int n = 0; n < N; n++) {
            for (int c_in = 0; c_in < C_in; c_in++) {
                for (int h_in = 0; h_in < H; h_in++) {
                    for (int w_in = 0; w_in < W; w_in++) {
                        
                        float sum = 0.0f;
                        for (int c_out = 0; c_out < C_out; c_out++) {
                            for (int kh = 0; kh < K; kh++) {
                                for (int kw = 0; kw < K; kw++) {
                                    int h_out_scaled = h_in + P - kh;
                                    int w_out_scaled = w_in + P - kw;
                                    
                                    if (h_out_scaled % S == 0 && w_out_scaled % S == 0) {
                                        int h_out = h_out_scaled / S;
                                        int w_out = w_out_scaled / S;
                                        
                                        if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                                            int w_idx = ((c_out * C_in + c_in) * K + kh) * K + kw;
                                            int out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                                            sum += w[w_idx] * dout[out_idx];
                                        }
                                    }
                                }
                            }
                        }
                        int x_idx = ((n * C_in + c_in) * H + h_in) * W + w_in;
                        dx[x_idx] = sum;
                    }
                }
            }
        }
    }


    // 3. Max Pooling
    void maxpool_forward(float* x, float* out, int* mask, int N, int C, int H, int W, int K, int S) {
        
        int H_out = (H - K) / S + 1;
        int W_out = (W - K) / S + 1;

        #pragma omp parallel for collapse(3)
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int h_out = 0; h_out < H_out; h_out++) {
                    for (int w_out = 0; w_out < W_out; w_out++) {
                        
                        int h_start = h_out * S;
                        int w_start = w_out * S;
                        float max_val = -1e9;
                        int max_idx = -1;

                        for (int i = 0; i < K; i++) {
                            for (int j = 0; j < K; j++) {
                                int h_in = h_start + i;
                                int w_in = w_start + j;
                                int idx = ((n * C + c) * H + h_in) * W + w_in;
                                
                                if (x[idx] > max_val) {
                                    max_val = x[idx];
                                    max_idx = idx;
                                }
                            }
                        }
                        
                        int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
                        out[out_idx] = max_val;
                        mask[out_idx] = max_idx; // Save index for backward pass
                    }
                }
            }
        }
    }

    void maxpool_backward(float* dout, float* dx, int* mask, int size) {
        // Initialize dx to 0 first (done in Python)
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            int idx = mask[i];
            #pragma omp atomic
            dx[idx] += dout[i];
        }
    }

    // 4. ReLU
    void relu_forward(float* x, float* out, int size) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            out[i] = (x[i] > 0) ? x[i] : 0.0f;
        }
    }

    void relu_backward(float* x, float* dout, float* dx, int size) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            dx[i] = (x[i] > 0) ? dout[i] : 0.0f;
        }
    }

    // 5. Cross Entropy Loss + Softmax (Combined for numerical stability)
    float cross_entropy_loss(float* logits, float* targets, float* dx, int batch, int classes) {
        // targets are one-hot encoded or probabilities
        float total_loss = 0.0f;

        // No parallel here to avoid race condition on total_loss easily, logic is light
        for (int i = 0; i < batch; i++) {
            // 1. Max trick for stability
            float max_val = -1e9;
            for (int j = 0; j < classes; j++) max_val = std::max(max_val, logits[i * classes + j]);

            // 2. Compute Softmax denominator
            float sum_exp = 0.0f;
            for (int j = 0; j < classes; j++) sum_exp += std::exp(logits[i * classes + j] - max_val);

            // 3. Compute Loss & Gradient
            for (int j = 0; j < classes; j++) {
                float prob = std::exp(logits[i * classes + j] - max_val) / sum_exp;
                if (targets[i * classes + j] > 0.5f) { // Assuming 1-hot
                    total_loss -= std::log(prob + 1e-7f);
                }
                // Gradient of CE + Softmax w.r.t logits is (prob - target)
                dx[i * classes + j] = (prob - targets[i * classes + j]) / batch; 
            }
        }
        return total_loss / batch;
    }
    
    // 6. Optimizer (SGD)
    void sgd_step(float* param, float* grad, int size, float lr) {
        #pragma omp parallel for
        for(int i=0; i<size; i++) {
            param[i] -= lr * grad[i];
        }
    }
}