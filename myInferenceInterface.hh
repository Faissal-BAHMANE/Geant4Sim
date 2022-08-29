#ifdef USE_INFERENCE
#ifndef MY_INFERENCEINTERFACE_HH
#define MY_INFERENCEINTERFACE_HH

#include "globals.hh"
using namespace std;

class myInferenceInterface
{
 public:
  virtual ~myInferenceInterface(){};

  virtual void RunInference(vector<float> aGenVector, std::vector<G4double>& aEnergies,
                            int aSize) = 0;
};

#endif // MY_INFERENCEINTERFACE_HH
#endif // USE_INFERENCE
