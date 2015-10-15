#include "brain/brain.h"


__kernel void slave(__global const bint *b, __global const syn *input,
  __global syn *network, __global syn *outputs) {
 size_t id = get_global_id(0);
 bProcess(b, &network[bSize(b)*id], input, &outputs[bOutputs(b)*id]);
}
