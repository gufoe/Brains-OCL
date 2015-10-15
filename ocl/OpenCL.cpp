#include "OpenCL.hpp"

using namespace ocl;

OpenCL::OpenCL(cl_platform_id platform) {
  mPlatform = platform;
  mArgs = 0;
}

cl_device_id *OpenCL::devices(cl_uint &n) {
  cl_device_id *devices = new cl_device_id[n];
  mCode = clGetDeviceIDs(mPlatform, CL_DEVICE_TYPE_GPU, n, devices, &n);
  check(mCode, "clGetDeviceIDs");
  return devices;
}

void OpenCL::init(cl_device_id device, std::string source, std::string kernel) {
  mDevice = device;

  // Create context
	mContext = clCreateContext(NULL, 1, &mDevice, NULL, NULL, &mCode);
	check(mCode, "clCreateContext");

	// Create command queue
	mQueue = clCreateCommandQueue(mContext, mDevice, 0, &mCode);
	check(mCode, "clCreateCommandQueue");

  // Finally build and create the kernel
  buildKernel(source);
  createKernel(kernel);
}

cl_platform_id *OpenCL::platforms(cl_uint &n) {
  cl_platform_id *platforms = new cl_platform_id[n];
  cl_int mCode = clGetPlatformIDs(10, platforms, &n);
	check(mCode, "clGetPlatformIDs");
  return platforms;
}

void OpenCL::buildKernel(std::string source) {
  const char *src = source.c_str();
  size_t src_len = source.length();

    // Build the kernel from source
	mProgram = clCreateProgramWithSource(mContext, 1, &src,
	     &src_len, &mCode);
	check(mCode, "clCreateProgramWithSource");
  std::cout << mProgram << mDevice << std::endl;
	mCode = clBuildProgram(mProgram, 1, &mDevice, "-I ./", NULL, NULL);
	if (mCode) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(mProgram, mDevice, CL_PROGRAM_BUILD_LOG, 0, NULL,
        &log_size);

		// Allocate memory for the log
		char *log = (char *) malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(mProgram, mDevice, CL_PROGRAM_BUILD_LOG, log_size,
        log, NULL);

    std::cerr << log << std::endl;
    check(mCode, "clBuildProgram");
  }
}



std::string OpenCL::loadKernel(std::string file) {
  std::ifstream t(file.c_str());
  if (!t.is_open()) {
    check(1, "Cannot load kernel file");
  }
  std::string str((std::istreambuf_iterator<char>(t)),
                   std::istreambuf_iterator<char>());
  t.close();
  return str;
}

void OpenCL::check(cl_int mCode, std::string str) {
  if (mCode != CL_SUCCESS) {
    std::cerr << str << ": " << mCode << std::endl;
    assert(mCode == CL_SUCCESS);
  }
}

void OpenCL::createKernel(std::string kernel) {
  	// Create the kernel
  	mKernel = clCreateKernel(mProgram, kernel.c_str(), &mCode);
  	check(mCode, "clCreateKernel");
}

void OpenCL::addParam(Param *p) {
    p->bind(&mContext, &mKernel, &mQueue, mArgs++);
}

void OpenCL::run(size_t global_work, size_t local_work) {
  	// Execute the kernel
  	mCode = clEnqueueNDRangeKernel(mQueue, mKernel, 1, 0, &global_work,
  			&local_work, 0, NULL, NULL);
  	check(mCode, "clEnqueueNDRangeKernel");
}
