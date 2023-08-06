/* Uncomment a specific define for the method that you want to try.
*/

// #define BASELINE
// #define FP_16
// #define STREAMS // final choice for submission.
#define TILING
// #define CONSTANT_MEM

#define TILE_WIDTH 16
#define mask_width 7

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"



#ifdef FP_16
    #include "cuda_fp16.h"
#endif

#ifdef STREAMS
    #define STREAMS_NUM 10
    static cudaStream_t stream_arr[STREAMS_NUM];
#endif

#ifdef CONSTANT_MEM
__constant__ float Mask[mask_width * mask_width * 64];
#endif
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
   #if defined(FP_16)
        #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
        #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
        #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

        const int Height_out = Height - K + 1;
        const int Width_out = Width - K + 1;
        const int H_size = ceil((float)(Height_out)/ (TILE_WIDTH * 1.0));
        const int W_size = ceil((float)(Width_out) / (float)(TILE_WIDTH));

        const int m = blockIdx.x;
        const int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
        const int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
        const int b = blockIdx.z;

        if (b < Batch && h < Height_out && w < Width_out && m < Map_out) {
             __half total = __float2half(0.0);
             for (int c = 0; c < Channel; c++) {
                for (int p = 0; p < K; p++) {
                    for (int q = 0; q < K; q++) {
                        total = __hfma(__float2half(in_4d(b,c, h + p, w + q)), __float2half(mask_4d(m, c, p, q)), total);
                    }
                }
             }
             out_4d(b, m, h, w) = __half2float(total);
        }

        #undef out_4d
        #undef in_4d
        #undef mask_4d

    #endif

    #if defined(STREAMS) || defined(BASELINE)
        #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
        #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
        #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

        const int Height_out = Height - K + 1;
        const int Width_out = Width - K + 1;

        const int H_size = ceil((float)(Height_out)/ (TILE_WIDTH * 1.0));
        const int W_size = ceil((float)(Width_out) / (float)(TILE_WIDTH));

        const int m = blockIdx.x;
        const int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
        const int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
        const int b = blockIdx.z;


        if (h < Height_out && w < Width_out && b < Batch && m < Map_out) {
            float acc = 0.0f;
            for (int c = 0; c < Channel; c++) {
                for (int p = 0; p < K; p++) {
                    for (int q = 0; q < K; q++) {
                        acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
                    }
                }
            }
            out_4d(blockIdx.z, blockIdx.x, h, w) = acc;
        }

        #undef out_4d
        #undef in_4d
        #undef mask_4d

    #endif

    #if defined(TILING)
        #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
        #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
        #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
        
        extern __shared__ float shared_mem[];
        const int Height_out = Height - K + 1;
        const int Width_out = Width - K + 1;


        const int H_size = ceil((float)(Height_out)/ (TILE_WIDTH * 1.0));
        const int W_size = ceil((float)(Width_out) / (float)(TILE_WIDTH));

        const int m = blockIdx.x;
        const int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
        const int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
        const int b = blockIdx.z;

        float* input_shared = &shared_mem[0];
        float* mask_shared = &shared_mem[(TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1)];

        #define in_2d(i1, i0) input_shared[(i1) * (TILE_WIDTH + K - 1) + i0]
        #define mask_2d(i1, i0) mask_shared[(i1) * (K) + i0]

        float acc = 0.0f;

        for (int c = 0; c < Channel; c++) {
            if (threadIdx.x < K && threadIdx.y < K && m < Map_out) {
                mask_2d(threadIdx.y, threadIdx.x) = mask_4d(m, c, threadIdx.y, threadIdx.x);
            }

            if (b < Batch) {
                for (int i = h; i < (blockIdx.y / W_size) * TILE_WIDTH + (TILE_WIDTH + K - 1); i++) {
                    for (int j = w; j < (blockIdx.y % W_size) * TILE_WIDTH + (TILE_WIDTH + K - 1); j++) {
                        in_2d(i - (blockIdx.y / W_size) * TILE_WIDTH, j - (blockIdx.y % W_size) * TILE_WIDTH) = in_4d(b, c, i, j);
                    }
                }
            }
            __syncthreads();
            
            if (h < Height_out && w < Width_out && b < Batch && m < Map_out) {
                for (int p = 0; p < K; p++) {
                    for (int q = 0; q < K; q++) {
                        acc += in_2d(threadIdx.y + p, threadIdx.x + q) * mask_2d(p, q);
                    }
                }
            }

            __syncthreads();
        }

        if (h < Height_out && w < Width_out && b < Batch && m < Map_out) {
            out_4d(b, m, h, w) = acc;
        }



        #undef out_4d
        #undef in_4d
        #undef mask_4d
        #undef in_2d
        #undef mask_2d


    #endif
        
    #if defined(CONSTANT_MEM)
        const int Height_out = Height - K + 1;
        const int Width_out = Width - K + 1;


        #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
        #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
        #define mask_4d(i3, i2, i1, i0) Mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

        // Insert your GPU convolution kernel code here
        const int W_grid = ceil(1.0*Width_out/TILE_WIDTH);
        int h = (blockIdx.z / W_grid) * blockDim.y + threadIdx.y;
        int w = (blockIdx.z % W_grid) * blockDim.x + threadIdx.x;

        
        if (h < Height_out && w < Width_out) {
            float acc = 0.0f;
            for (int c = 0; c < Channel; c++) {
                for (int p = 0; p < K; p++) {
                    for (int q = 0; q < K; q++) {
                        acc += in_4d(blockIdx.x, c, h + p, w + q) * mask_4d(blockIdx.y, c, p, q);
                    }
                }
            }
            out_4d(blockIdx.x, blockIdx.y, h, w) = acc;
        }
        
        

        #undef out_4d
        #undef in_4d
        #undef mask_4d
    #endif
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    #if defined(FP_16) || defined(TILING) || defined(BASELINE)
        cudaMalloc((void **) device_output_ptr, (Batch * Map_out * (Height - K + 1) * (Width - K + 1))*sizeof(float));
        cudaMalloc((void **) device_input_ptr, (Batch * Channel * Height * Width)*sizeof(float));
        cudaMalloc((void**) device_mask_ptr, (K * K * Map_out * Channel) * sizeof(float));
        cudaMemcpy(*device_input_ptr, host_input, (Batch * Channel * Height * Width)*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)*device_mask_ptr, (void*)host_mask, (Map_out * Channel * K * K) * sizeof(float), cudaMemcpyHostToDevice);
    #endif

    #if defined(STREAMS)
        cudaMalloc((void **) device_output_ptr, (Batch * Map_out * (Height - K + 1) * (Width - K + 1))*sizeof(float));
        cudaMalloc((void **) device_input_ptr, (Batch * Channel * Height * Width)*sizeof(float));
        cudaMalloc((void**) device_mask_ptr, (K * K * Map_out * Channel) * sizeof(float));

        const unsigned int in_section =  ceil(((float) (Batch * Channel * Height * Width)) / ((float) STREAMS_NUM));
        const int batch_section = ceil(((float) Batch) / ((float) STREAMS_NUM));
        cudaMemcpy((void*)*device_mask_ptr, (void*)host_mask, (Map_out * Channel * K * K) * sizeof(float), cudaMemcpyHostToDevice);

        for (int i = 0; i < STREAMS_NUM; i++) {
            cudaStreamCreate(&stream_arr[i]);
            cudaMemcpyAsync((void*) (*device_input_ptr + i * in_section), (void*) (host_input + i * in_section), in_section * sizeof(float), cudaMemcpyHostToDevice, stream_arr[i]);
        }

    #endif

    #if defined(CONSTANT_MEM)
        cudaMalloc((void **) device_output_ptr, (Batch * Map_out * (Height - K + 1) * (Width - K + 1))*sizeof(float));
        cudaMalloc((void **) device_input_ptr, (Batch * Channel * Height * Width)*sizeof(float));
        cudaMalloc((void**) device_mask_ptr, (K * K * Map_out * Channel) * sizeof(float));
        cudaMemcpy(*device_input_ptr, host_input, (Batch * Channel * Height * Width)*sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy((void*)*device_mask_ptr, (void*)host_mask, (Map_out * Channel * K * K) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(Mask, host_mask, (Map_out * Channel * K * K)*sizeof(float), 0, cudaMemcpyHostToDevice);
    #endif
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    #if defined(FP_16) || defined(BASELINE)
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 dimGrid(Map_out, ceil((float)(Height - K + 1)/(float)TILE_WIDTH)*ceil((float)(Width - K + 1)/(float)TILE_WIDTH), Batch);
        conv_forward_kernel<<<dimGrid,dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

    #endif

    #if defined(STREAMS)
        const int batch_sect = ceil((Batch * 1.0)/ ((float)STREAMS_NUM));

        const unsigned int out_sect = ceil((float)(Batch * Map_out * (Height - K + 1) * (Width - K + 1))/ ((float) STREAMS_NUM));
        const unsigned int in_section = ceil(((float) (Batch * Channel * Height * Width)) / ((float) STREAMS_NUM));

        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 dimGrid(Map_out,  ceil((float)(Height - K + 1)/(float)TILE_WIDTH)*ceil((float)(Width - K + 1)/(float)TILE_WIDTH), batch_sect);

        for (int i = 0; i < STREAMS_NUM; i++) {
            conv_forward_kernel<<<dimGrid, dimBlock, 0, stream_arr[i]>>>(device_output + i * out_sect, device_input + i * in_section, device_mask, Batch, Map_out, Channel, Height, Width, K);
        }
    #endif

    #if defined(CONSTANT_MEM)
        dim3 dimGrid(Batch, Map_out, ceil((float)(Height - K + 1)/TILE_WIDTH)*ceil((float)(Width - K + 1)/TILE_WIDTH));
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    #endif

    #if defined(TILING) 
        // Set the kernel dimensions and call the kernel
        const int Height_size = ceil((float) (Height - K + 1) / (float) TILE_WIDTH);
        const int Width_size = ceil((float) (Width - K + 1) / (float) TILE_WIDTH);
        //const int Batch_size = ceil((float) Batch / (float) TILE_SIZE);

        const int shmem_size = sizeof(float) * ( ((TILE_WIDTH + K-1)*(TILE_WIDTH + K-1)) + (K*K) );

        const int Y = Height_size * Width_size;

        dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridSize(Map_out, Y, Batch);

        // printf("This is K: %d\n", K);

        conv_forward_kernel<<<gridSize, blockSize, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    #endif
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    #if defined(FP_16) || defined(TILING) || defined(BASELINE)
        cudaMemcpy(host_output, device_output, (Batch * Map_out * (Height - K + 1) * (Width - K + 1))*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(device_output);
        cudaFree(device_input);
        cudaFree(device_mask);
    #endif

    #if defined(STREAMS)
        const unsigned int out_sect = ceil((float)(Batch * Map_out * (Height - K + 1) * (Width - K + 1))/ ((float) STREAMS_NUM));

        for (int i = 0; i < STREAMS_NUM; i++) {
            cudaMemcpyAsync((void*) (host_output + i * out_sect), (void*)(device_output + i * out_sect), out_sect * sizeof(float), cudaMemcpyDeviceToHost, stream_arr[i]);
            cudaStreamDestroy(stream_arr[i]);
        }
        cudaFree(device_output);
        cudaFree(device_input);
        cudaFree(device_mask);

    #endif

    #if defined(CONSTANT_MEM)
        // Copy the output back to host
        cudaMemcpy(host_output, device_output, (Batch * Map_out * (Height - K + 1) * (Width - K + 1))*sizeof(float), cudaMemcpyDeviceToHost);
        // Free device memory
        cudaFree(device_output);
        cudaFree(device_input);
        cudaFree(device_mask);
    #endif


}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
