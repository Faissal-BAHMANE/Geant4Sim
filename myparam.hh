#ifndef MYPARAM_HH
#define MYPARAM_HH

#include "G4VFastSimulationModel.hh"
#include "G4Step.hh"
#include "G4TouchableHandle.hh"
#include <vector>

class MyParamModel : public G4VFastSimulationModel
{
public:
  MyParamModel (G4String, G4Region*);
  MyParamModel (G4String);
  ~MyParamModel ();

  virtual G4bool IsApplicable(const G4ParticleDefinition&);
  virtual G4bool ModelTrigger(const G4FastTrack &);
  virtual void DoIt(const G4FastTrack&, G4FastStep&);

private:
  /// Inference model that is NN aware
  myInferenceSetup* fInference;
  /// Inference model that is NN aware
  /// Helper class for creation of hits within the sensitive detector
  std::unique_ptr<G4FastSimHitMaker> fHitMaker;
  /// Vector of energy values
  std::vector<G4double> fEnergies;
  /// Vector of positions corresponding to energy values (const for one NN
  /// model)
  std::vector<G4ThreeVector> fPositions;

};
#endif // MYPARAM_HH
