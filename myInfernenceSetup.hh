#ifdef USE_INFERENCE
#ifndef MY_INFEERENCESETUP_HH
#define MY_INFEERENCESETUP_HH

#include "G4ThreeVector.hh"
#include "globals.hh"
#include "CLHEP/Units/SystemOfUnits.h"

#include "construction.hh"
#include "myInferenceMessenger.hh"
#include "myInferenceInterface.hh"

namespace CLHEP
{
  class HepRandomEngine;
}
class myInferenceMessenger;


class myInferenceSetup
{
 public:
  myInferenceSetup();
  ~myInferenceSetup();

  /// Geometry setup
  /// Check if inference should be performed for the particle
  /// @param[in] aEnergy Particle's energy
  G4bool IfTrigger(G4double aEnergy);
  /// Set path and name of the model
  inline void SetModelPathName(G4String aName) { fModelPathName = aName; };
  /// Get path and name of the model
  inline G4String GetModelPathName() const { return fModelPathName; };
  /// Set profiling flag
  inline void SetProfileFlag(G4int aNumber) { fProfileFlag = aNumber; };
  /// Get profiling flag
  inline G4int GetProfileFlag() const { return fProfileFlag; };
  /// Set optimization flag
  inline void SetOptimizationFlag(G4int aNumber) { fOptimizationFlag = aNumber; };
  /// Get optimization flag
  inline G4int GetOptimizationFlag() const { return fOptimizationFlag; };
  /// Get name of the inference library
  inline G4String GetInferenceLibrary() const { return fInferenceLibrary; };
  /// Set name of the inference library and create a pointer to chosen inference interface
  void SetInferenceLibrary(G4String aName);
  /// Check settings of the inference library
  void CheckInferenceLibrary();

  /// Execute inference
  /// @param[out] aDepositsEnergies of inferred energies deposited in the
  /// detector
  /// @param[in] aParticleEnergy Energy of initial particle
  void GetEnergies(std::vector<G4double>& aEnergies, G4double aParticleEnergy,
                   G4float aInitialAngle);

  /// Calculate positions
  /// @param[out] aDepositsPositions Vector of positions corresponding to
  /// energies deposited in the detector
  /// @param[in] aParticlePosition Initial particle position which is centre of
  /// transverse plane of the mesh
  ///            and beginning of the mesh in the longitudinal direction
  /// @param[in] aParticleDirection Initial particle direction for the mesh
  /// rotation
  void GetPositions(std::vector<G4ThreeVector>& aDepositsPositions, G4ThreeVector aParticlePosition,
                    G4ThreeVector aParticleDirection);

 private:
  /// Pointer to detector construction to retrieve (once) the detector
  /// dimensions
  MyDetectorConstruction* fDetector;
  /// Inference interface
  std::unique_ptr<myInferenceInterface> fInferenceInterface;
  /// Inference messenger
  myInferenceMessenger* fInferenceMessenger;
  /// Maximum particle energy value (in MeV) in the training range
  float fMaxEnergy = 1024000.0;
  /// Maximum particle angle (in degrees) in the training range
  float fMaxAngle = 90.0;
  /// Name of the inference library
  G4String fInferenceLibrary = "ONNX";
  G4String fModelPathName = "MLModels/Generator.onnx";
  /// ONNX specific
  /// Profiling flag
  G4bool fProfileFlag = false;
  /// Optimization flag
  G4bool fOptimizationFlag = false;
  /// Intra-operation number of threads
  G4int fIntraOpNumThreads = 1;
};

#endif // PAR04INFEERENCESETUP_HH
#endif // USEINFERENCE
