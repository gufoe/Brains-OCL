#include "Population.hpp"

using namespace std;
using namespace evo;

Population::Population(vector<Subject*> *subjects) {
  mSubjects = subjects;
}

double Population::getTotalFitness() {
  double tot = 0;
  for(size_t i = 0; i < mSubjects->size(); i++) {
    tot+= mSubjects->at(i)->getFitness();
  }
  return tot;
}

double Population::getAverageFitness() {
  return getTotalFitness()/mSubjects->size();
}

Population *Population::reproduce(float multiplier, float gift) {

  vector<Subject*> *pool = new vector<Subject*>;
  double fit, best = -1;
  for(size_t i = 0; i < mSubjects->size(); i++) {
    fit = mSubjects->at(i)->getFitness();
    if(fit > best) {
      mBest = mSubjects->at(i);
      best = fit;
    }
    fit*= multiplier;
    fit+= gift;
    //std::cout << fit << "\n";
    for(int j = 0; j < fit; j++)
      pool->push_back(mSubjects->at(i));
  }
  vector<Subject*> *children = new vector<Subject*>;
  Subject* dad;
  Subject* mom;
  for(size_t i = 0; i < mSubjects->size(); i++) {
    dad = pool->at(rand()%pool->size());
    mom = pool->at(rand()%pool->size());
    children->push_back(dad->reproduce(mom));
  }
  delete pool;
  Population *p = new Population(children);
  return p;
}

Population::~Population() {
  for(size_t i = 0; i < mSubjects->size(); i++)
    delete mSubjects->at(i);

  delete mSubjects;
}
