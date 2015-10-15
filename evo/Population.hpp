#ifndef POPULATION_H_
#define POPULATION_H_

#include <vector>
#include <cstdlib>
#include <math.h>

#include "Subject.hpp"

namespace evo {
  class Population {

  private:
    Subject* mBest;
    std::vector<Subject*> *mSubjects;
  public:
    Population(std::vector<Subject*> *);
    ~Population();
    double getTotalFitness();
    double getAverageFitness();
    Population *reproduce(float multiplier, float gift);
    std::vector<Subject*> *getSubjects() { return mSubjects; };
    Subject* getBest() {return mBest;}

  };
}

#endif
