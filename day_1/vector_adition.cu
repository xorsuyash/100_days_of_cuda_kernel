#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addVectors(float *a, float *b, float *c, int n)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index<n){
        c[index]=a[index]+b[index];
    }
}

int main(void){

    int n=10;
    size_t bytes=n*sizeof(float);

    float *h_a,*h_b,*h_c;

    float *d_a,*d_b,*d_c;

    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    for (int i=0; i<n; i++){
        h_a[i] = i;
        h_b[i] = i*2;
    }

    cudaMalloc(&d_a,bytes);
    cudaMalloc(&d_b,bytes);
    cudaMalloc(&d_c,bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize=256;
    int gridSize = (n+blockSize-1)/blockSize;


    addVectors<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    printf("Vector addition results:\n");
    for (int i = 0; i < n; i++) {
        printf("%0.1f + %0.1f = %0.1f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;



}