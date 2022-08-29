#ifdef USE_INFERENCE_ONNX
#ifndef ONNXINFERENCE_HH
#define ONNXINFERENCE_HH

#include "myInferenceInterface.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "G4ThreeVector.hh"
#include "globals.hh"
#include "core/session/onnxruntime_cxx_api.h"


class myOnnxInference : public myInferenceInterface
{
 public:
  myOnnxInference(G4String, G4int, G4int, G4int);
  myOnnxInference();

  void RunInference(vector<float> aGenVector, std::vector<G4double>& aEnergies, int aSize);

 private:
  /// Pointer to the ONNX enviroment
  std::unique_ptr<Ort::Env> fEnv;
  /// Pointer to the ONNX inference session
  std::unique_ptr<Ort::Session> fSession;
  /// ONNX settings
  Ort::SessionOptions fSessionOptions;
  /// ONNX memory info
  const OrtMemoryInfo* fInfo;
  struct MemoryInfo;
  /// the input names represent the names given to the model
  /// when defining  the model's architecture (if applicable)
  /// they can also be retrieved from model.summary()
  std::vector<const char*> fInames;
};

#endif // ONNXINFERENCE_HH
#endif
