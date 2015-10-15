#ifndef SUBJECT_H_
#define SUBJECT_H_

#include <vector>
#include <assert.h>
#include <iostream>


namespace evo {

	class Subject {
		public:
			virtual Subject* reproduce(Subject*) = 0;
			virtual double getFitness() = 0;
      virtual ~Subject() {}
	};
}

#endif
