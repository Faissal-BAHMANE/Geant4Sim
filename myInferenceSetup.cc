#ifdef USE_INFERENCE
#include "myInferenceSetup.hh"

#include "myInferenceInterface.hh"
#ifdef USE_INFERENCE_ONNX
#include "myOnnxInference.hh"
#endif
#ifdef USE_INFERENCE_LWTNN
#include "myLwtnnInference.hh"
#endif
#include "G4RotationMatrix.hh"
#include "CLHEP/Random/RandGauss.h"


myInferenceSetup::myInferenceSetup()
  : fInferenceMessenger(new myInferenceMessenger(this))
{}


myInferenceSetup::~myInferenceSetup() {}


G4bool myInferenceSetup::IfTrigger(G4double aEnergy)
{
  /// Energy of electrons used in training dataset
  if(aEnergy > 1 * CLHEP::GeV || aEnergy < 1024 * CLHEP::GeV)
    return true;
}

void myInferenceSetup::SetInferenceLibrary(G4String aName)
{
  fInferenceLibrary = aName;

#ifdef USE_INFERENCE_ONNX
  if(fInferenceLibrary == "ONNX")
    fInferenceInterface = std::unique_ptr<myInferenceInterface>(
      new myOnnxInference(fModelPathName, fProfileFlag, fOptimizationFlag, fIntraOpNumThreads));
#endif
#ifdef USE_INFERENCE_LWTNN
  if(fInferenceLibrary == "LWTNN")
    fInferenceInterface =
      std::unique_ptr<myInferenceInterface>(new myLwtnnInference(fModelPathName));
#endif
  CheckInferenceLibrary();
}

void myInferenceSetup::CheckInferenceLibrary()
{
  G4String msg = "Please choose inference library from available libraries (";
#ifdef USE_INFERENCE_ONNX
  msg += "ONNX,";
#endif
#ifdef USE_INFERENCE_LWTNN
  msg += "LWTNN";
#endif
  if(fInferenceInterface == nullptr)
    G4Exception("myInferenceSetup::CheckInferenceLibrary()", "InvalidSetup", FatalException,
                (msg + "). Current name: " + fInferenceLibrary).c_str());
}

void myInferenceSetup::GetEnergies(std::vector<G4double>& aEnergies, G4double aInitialEnergy,
                                      G4float aInitialAngle)
{
  // First check if inference library was set correctly
  CheckInferenceLibrary();

  //inferece gode here

  // Run the inference
  fInferenceInterface->RunInference(genVector, aEnergies, size);
  // After the inference rescale back to the initial energy (in this example the
  // energies of cells were normalized to the energy of the particle)
  for(int i = 0; i < size; ++i)
  {
    aEnergies[i] = aEnergies[i] * aInitialEnergy;
  }
}


void myInferenceSetup::GetPositions(std::vector<G4ThreeVector>& aPositions, G4ThreeVector pos0,
                                       G4ThreeVector direction)
{
}

#endif
