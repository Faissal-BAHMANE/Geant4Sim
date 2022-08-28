#ifndef DETECTOR_HH
#define DETECTOR_HH

#include "G4VSensitiveDetector.hh"
#include "G4AnalysisManager.hh"
#include "G4RunManager.hh"
#include "myHit.hh"


class MySensitiveDetector : public G4VSensitiveDetector{
public:
  MySensitiveDetector(const G4String& name,
                      const G4String& hitsCollectionName);
  ~MySensitiveDetector() override;

  // methods from base class
  void   Initialize(G4HCofThisEvent* hitCollection) override;
  G4bool ProcessHits(G4Step* step, G4TouchableHistory* history) override;
  void   EndOfEvent(G4HCofThisEvent* hitCollection) override;

private:
  myHitsCollection* fHitsCollection = nullptr;

};
#endif
