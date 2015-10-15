#include <CL/cl.h>
#include <iostream>
#include <assert.h>
#include <fstream>
#include <streambuf>
#include <vector>

namespace ocl {
  class Param;

  class OpenCL {

  private:
    cl_platform_id mPlatform;
    cl_device_id mDevice;
    cl_context mContext;
    cl_command_queue mQueue;
    cl_program mProgram;
    cl_int mCode;
    cl_kernel mKernel;
    cl_uint mArgs;

    void buildKernel(std::string source);
    void createKernel(std::string kernel);

  public:
    OpenCL(cl_platform_id platform);
    cl_device_id *devices(cl_uint &n);
    void init(cl_device_id device, std::string source, std::string kernel);
    void addParam(Param *p);
    void run(size_t global_work, size_t local_work);
    static void check(cl_int code, std::string tr);
    static std::string loadKernel(std::string file);
    static cl_platform_id *platforms(cl_uint &n);
  };


  class Param {
  private:
    size_t mSize;
    cl_mem_flags mFlags;
    cl_command_queue *mQueue;
    cl_mem mMem;
    cl_kernel *mKernel;

  public:
    Param(size_t size, cl_mem_flags flags) {
      mSize = size;
      mFlags = flags;
    }
    void bind(cl_context *context, cl_kernel *kernel, cl_command_queue *queue, int pos) {
      cl_int code;
      mKernel = kernel;
      mQueue = queue;
  	  mMem = clCreateBuffer(*context, mFlags, mSize, NULL, &code);
  	  OpenCL::check(code, "clCreateBuffer");

    	code = clSetKernelArg(*kernel, pos, sizeof(mMem), (void*)&mMem);
    	OpenCL::check(code, "clSetKernelArg");
    }

    void write(void *p) {
    	cl_int code = clEnqueueWriteBuffer(*mQueue, mMem, CL_TRUE, 0, mSize, p, 0,
          NULL, NULL);
    	OpenCL::check(code, "clEnqueueWriteBuffer");
    }

    void read(void *p) {
    	cl_int code = clEnqueueReadBuffer(*mQueue, mMem, CL_TRUE, 0, mSize, p, 0,
          NULL, NULL);
    	OpenCL::check(code, "clEnqueueReadBuffer");
    }

    cl_mem *getMem() { return &mMem; }
    size_t getSize() { return mSize; }
  };

}
