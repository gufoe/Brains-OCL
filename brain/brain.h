#ifdef HOST
#include <math.h>
#include <malloc.h>
#include <iomanip>
#endif

// Brain definition structure:
// 1: flags
// 1: number of syn layers
// N: incremental number of neurons in each layer

typedef float syn; // Single sinapse multiplier
typedef syn* nnet;
typedef unsigned short bint;
typedef bint synapse; // Array of synapses
typedef bint neuron; // Array of synapses
typedef bint layer; // A complete neural network
typedef bint* brain; // Contains the brain definition

// Brain header size and indexes
#define BH_LEN 3
#define BH_FLAGS  0
#define BH_SYNAPSES  1
#define BH_LAYERS 2

// Brain flags
#define BF_RECURRENT 1

// Fast rand functions

#ifdef HOST
#define __global
#endif


unsigned short bRand() {
  static unsigned int g_seed = 0;
  g_seed = (214013*g_seed+2531011);
  return (g_seed>>16)&0x7FFF;
}

inline syn bActivate(syn x) {
  return (syn)(1/(1 + exp(-x)));
}

inline syn bWeight(syn factor) {
  return (syn)(4*((syn)(bRand()%0xFFFF)/(syn)0xFFFF-.25F)*factor);
}

inline bint bFlags(__global const bint *b) {
  return b[BH_FLAGS];
}
inline bint bBytes(__global const bint *b) {
  return b[BH_SYNAPSES]*sizeof(syn);
}
inline bint bSize(__global const bint *b) {
  return b[BH_SYNAPSES];
}
inline  bint bLayers(const __global bint *b) {
  return b[BH_LAYERS];
}
inline bint bNeurons(__global const bint *b, layer l) {
  return b[BH_LEN+l]-(!l ? 0 : b[BH_LEN+l-1]);
}
inline bint bInputs(__global const bint *b) {
  return bNeurons(b, 0);
}
inline bint bSynapses(__global const bint *b, layer l) {
  // Base synapses are for output and bias
  bint size = 2+bNeurons(b, l-1);

  if (bFlags(b) & BF_RECURRENT)
    if (l < bLayers(b)-1)
      size+= bNeurons(b, l+1);

  return size;
}

inline bint bLayerSize(__global const bint *b, layer l) {
  return bSynapses(b, l)*bNeurons(b, l);
}


inline bint bLayer(__global const bint *b, layer l) {
  // First layer
  if (!l) return 0;

  // Other layers
  bint idx = 0;
  for (bint i = 1; i < l; i++)
    idx+= bLayerSize(b, i);

  return idx;
}

inline bint bNeuron(__global const bint *b, layer l, neuron n) {
  return bLayer(b, l)+bSynapses(b, l)*n;
}
inline bint bSynapse(__global const bint *b, layer l, neuron n, synapse s) {
  return bLayer(b, l)+bSynapses(b, l)*n+s;
}

bint bCalcSize(__global const bint *b) {
  bint size = 0;
  for (bint i = 1; i < bLayers(b); i++) {
    size+= bLayerSize(b, i);
  }
  return size;
}
inline bint bDefSize(__global const bint *b) {
  return BH_LEN+bLayers(b);
}

inline bint bOutputs(__global const bint *b) {
  return bNeurons(b, bLayers(b)-1);
}

#ifdef HOST
inline bint *bDefine(bint flags, const layer *layers) {
  bint num_layers = 0;
  while (layers[++num_layers]);

  bint *b = (brain)malloc((BH_LEN+num_layers)*sizeof(bint));
  b[BH_FLAGS] = flags;
  b[BH_LAYERS] = num_layers;

  // Skip one layer (input layer)
  for (bint l = 0; layers[l]; l++) {
    b[BH_LEN+l] = layers[l]+(!l ? 0 : b[BH_LEN+l-1]);
  }

  b[BH_SYNAPSES] = bCalcSize(b);

  return b;
}
#endif
void bCreate(__global bint *b, nnet net) {
  for (bint l = 1; l < bLayers(b); l++)
    for(bint n = 0; n < bNeurons(b, l); n++)
      for(bint s = 0; s < bSynapses(b, l); s++)
        net[bSynapse(b, l, n, s)] = bWeight(1);
}

#ifdef HOST
inline void bPrint(brain b, syn *net) {
  for (int i = 0 ; i < bSize(b); i++)
    std::cout << "   " << std::fixed << std::setw(3) << std::setprecision(1)
      << fabs(net[i]);
  std::cout << "\n";
}
#endif

void bMix(__global bint *b, __global syn *dad, __global syn *mom, __global syn *child, float mrate) {
  __global syn *src;
  for (bint l = 1; l < bLayers(b); l++)
    for(bint n = 0; n < bNeurons(b, l); n++) {
      src = bRand()%2?dad:mom;
      for(bint s = 0; s < bSynapses(b, l); s++) {
        child[bSynapse(b, l, n, s)] = src[bSynapse(b, l, n, s)];
      }
    }
}
inline void bProcess(__global const bint *b, __global syn *net, const __global syn *input, __global syn *output) {

   for (bint l = 1; l < bLayers(b); l++)
    for(bint n = 0; n < bNeurons(b, l); n++) {
      bint o = bNeuron(b, l, n);
      net[o] = net[bSynapse(b, l, n, 1)];
      // Beware: synapses start from 1 (first syn is actually the neuron output)
      for(bint s = 0; s < bNeurons(b, l-1); s++) {
        net[o]+= net[bSynapse(b, l, n, s+2)]*(l==1?input[s]:net[bNeuron(b, l-1, s)]);
      }
      if (bFlags(b)&BF_RECURRENT && l < bLayers(b)-1)
        for(bint s = 0; s < bNeurons(b, l+1); s++)
          net[o]+= net[bSynapse(b, l, n, s+2)]*net[bNeuron(b, l+1, s)];

      net[o] = bActivate(net[o]);
    }

  bint l = bLayers(b)-1;
  for(bint n = 0; n < bNeurons(b, l); n++) {
    output[n] = net[bNeuron(b, l, n)];
  }
}
