#ifdef USE_INFERENCE

#include "MyParamModel.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4FastHit.hh"
#include "G4FastSimHitMaker.hh"
#include "Randomize.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include <numeric>


MyParamModel::MyParamModel(G4String aModelName, G4Region* aEnvelope)
  : G4VFastSimulationModel(aModelName, aEnvelope)
  , fInference(new Par04InferenceSetup)
  , fHitMaker(new G4FastSimHitMaker)
{}


MyParamModel::MyParamModel(G4String aModelName)
  : G4VFastSimulationModel(aModelName)
  , fInference(new myInferenceSetup)
  , fHitMaker(new G4FastSimHitMaker)
{}

MyParamModel::~MyParamModel() {}


G4bool MyParamModel::IsApplicable(const G4ParticleDefinition& aParticleType)
{
  return &aParticleType == G4Electron::ElectronDefinition();
}

G4bool MyParamModel::ModelTrigger(const G4FastTrack& aFastTrack)
{
  return fInference->IfTrigger(aFastTrack.GetPrimaryTrack()->GetKineticEnergy());
}

void MyParamModel::DoIt(const G4FastTrack& aFastTrack, G4FastStep& aFastStep)
{
  // remove particle from further processing by G4
  aFastStep.KillPrimaryTrack();
  aFastStep.SetPrimaryTrackPathLength(0.0);
  G4double energy = aFastTrack.GetPrimaryTrack()->GetKineticEnergy();
  aFastStep.SetTotalEnergyDeposited(energy);
  G4ThreeVector position  = aFastTrack.GetPrimaryTrack()->GetPosition();
  G4ThreeVector direction = aFastTrack.GetPrimaryTrack()->GetMomentumDirection();

  // calculate the incident angle
  G4float angle = direction.theta();

  // calculate how to deposit energy within the detector
  // get it from inference model
  fInference->GetEnergies(fEnergies, energy, angle);
  fInference->GetPositions(fPositions, position, direction);

  // deposit energy in the detector using calculated values of energy deposits
  // and positions
  for(size_t iHit = 0; iHit < fPositions.size(); iHit++)
  {
    fHitMaker->make(G4FastHit(fPositions[iHit], fEnergies[iHit]), aFastTrack);
  }
}
#endif // USEINFERENCE
