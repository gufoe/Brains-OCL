#include <vector>
#include <iostream>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <CL/cl.h>

#include "evo/Population.hpp"
#include "evo/Subject.hpp"
#include "ocl/OpenCL.hpp"
#define HOST
//#define CPU
#include "brain/brain.h"

class Brain : public evo::Subject {
private:
	brain mBrain;
	nnet mParents[2];

public:
  nnet mNet;
	syn *mOutputs;
  double mFitness;

	static syn target;

	Brain(Brain *dad, Brain *mom) {
    mFitness = 0;
		mBrain = dad->mBrain;
		mParents[0] = dad->mNet;
		mParents[1] = mom->mNet;
	}
	Brain(brain b, nnet mem, syn *mem_out, int pos) {
    mFitness = 0;
		mBrain = b;
		mNet = (nnet)((size_t)mem+bSize(mBrain)*pos*sizeof(syn));
		mOutputs = (nnet)((size_t)mem_out+bOutputs(mBrain)*pos*sizeof(syn));
		bCreate(b, mNet);
	}
	void growUp(nnet mem, syn *mem_out, size_t pos) {
    mNet = (nnet)((size_t)mem+bSize(mBrain)*pos*sizeof(syn));
    mOutputs = (nnet)((size_t)mem_out+bOutputs(mBrain)*pos*sizeof(syn));
		bMix(mBrain, mParents[0], mParents[1], mNet, .02F);
	}
	Brain *reproduce(evo::Subject *b) {
		return new Brain(this, (Brain *)b);
	}
	double getFitness() {
		return (int)(1.5-fabs(Brain::target-mOutputs[0]));
	}
  void updateFitness() {
    mFitness+= getFitness();
  }
  ~Brain() {}
};

syn Brain::target;

ocl::OpenCL * initOCL() {
	cl_uint n = 1;
	ocl::OpenCL *ocl = new ocl::OpenCL(ocl::OpenCL::platforms(n)[0]);
	ocl->init(ocl->devices(n)[0], ocl::OpenCL::loadKernel("kernel.cl"), "slave");
	return ocl;
}

int main() {
	size_t pSize = 10240;
	layer layers[] = {2, 1000, 10, 100, 0};
	brain brain = bDefine(0, layers);

	ocl::OpenCL *ocl = initOCL();
	evo::Population *pop;
	std::vector<evo::Subject *> *subjects = new std::vector<evo::Subject *>;

	// Prepare memory to store brains and their children
	nnet bMem = (nnet) malloc(bSize(brain)*pSize*sizeof(syn));
	assert(bMem != 0);
	nnet bMemChildren = (nnet) malloc(bSize(brain)*pSize*sizeof(syn));
	assert(bMemChildren != 0);
	nnet bMemOutputs = (nnet) malloc(bOutputs(brain)*pSize*sizeof(syn));
	assert(bMemOutputs != 0);

	std::cout << bDefSize(brain) << " : " << bSize(brain) << " : " << brain[BH_SYNAPSES] << std::endl;

	// Create brains
	for (size_t i = 0; i < pSize; i++)
		subjects->push_back(new Brain(brain, bMem, bMemOutputs, i));
	//return 0;
	// Initialize population
	pop = new evo::Population(subjects);

  #ifndef CPU
	// Prepare kernel parameters
	ocl::Param pBrain(bDefSize(brain)*sizeof(bint), CL_MEM_READ_ONLY); // Brain def
	ocl::Param pInput(bInputs(brain)*sizeof(syn), CL_MEM_READ_WRITE); // Inputs
	ocl::Param pNetwork(bSize(brain)*sizeof(syn)*pSize, CL_MEM_READ_WRITE); // Brains
	ocl::Param pOutputs(bOutputs(brain)*sizeof(syn)*pSize, CL_MEM_WRITE_ONLY); // Output array

	ocl->addParam(&pBrain);
	ocl->addParam(&pInput);
	ocl->addParam(&pNetwork);
	ocl->addParam(&pOutputs);

	pBrain.write(brain);
  #endif

	// Preparing samples
	size_t samples = 4;
	syn inputs[4][2] = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};
	syn outputs[4][1] = {
		{0},
		{1},
		{1},
		{0}
	};


	// Do stuff
	for (size_t cycle = 0; cycle < 10; cycle++) {
		std::cout << "Running the kernel... " << std::flush;

    #ifndef CPU
    pNetwork.write(bMem);
    #endif

    for (size_t j = 0; j < samples; j++) {
	    Brain::target = outputs[j%samples][0];

  		// Write inputs and brains
      #ifndef CPU
  		pInput.write(inputs[cycle%samples]);
  		//pOutputs.write(bMemOutputs);
      std::cout << "running... " << std::flush;
      ocl->run(pSize, 512);

      pOutputs.read(bMemOutputs);

      #else
	    for (size_t i = 0; i < pSize; i++) {
        bProcess(brain, ((Brain*)subjects->at(i))->mNet, inputs[j%samples],
          ((Brain*)subjects->at(i))->mOutputs);
      }
      #endif

	    for (size_t i = 0; i < pSize; i++) {
        ((Brain*)subjects->at(i))->updateFitness();
      }
    }

    // Read output
		//
		//std::cout << "done.\n";

		//std::cout << "Evaluating the brains... " << std::flush;
		int tot_fitness = pop->getTotalFitness();
		evo::Population *old = pop;
		pop = old->reproduce(10, 1);
		subjects = pop->getSubjects();
		assert(subjects->size() == pSize);
		for (size_t i = 0; i < pSize; i++)
			((Brain *)subjects->at(i))->growUp(bMemChildren, bMemOutputs, i);
		std::cout << "done (" << tot_fitness << ").\n";

		// Swap children memory and brain memory
		nnet m = bMem;
		bMem = bMemChildren;
		bMemChildren = m;

		// Delete old population
		delete(old);
	}

	return 0;
}
