#include "detector.hh"

MySensitiveDetector::MySensitiveDetector(const G4String& name,
                                         const G4String& hitsCollectionName) : G4VSensitiveDetector(name)
{
  collectionName.insert(hitsCollectionName);
}

MySensitiveDetector::~MySensitiveDetector(){}

void MySensitiveDetector::Initialize(G4HCofThisEvent* hce)
{
  // Create hits collection

  fHitsCollection = new myHitsCollection(SensitiveDetectorName, collectionName[0]);

  // Add this collection in hce

  // G4int hcID
  //   = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  // hce->AddHitsCollection( hcID, fHitsCollection );
}

G4bool MySensitiveDetector::ProcessHits(G4Step *aStep, G4TouchableHistory*)
{
  G4double edep = aStep->GetTotalEnergyDeposit();

  if (edep==0.) return false;

  myHit* newHit = new myHit();

  newHit->SetTrackID  (aStep->GetTrack()->GetTrackID());
  newHit->SetChamberNb(aStep->GetPreStepPoint()->GetTouchableHandle() ->GetCopyNumber());
  newHit->SetEdep(edep);
  newHit->SetPos (aStep->GetPostStepPoint()->GetPosition());

  // fHitsCollection->insert(newHit);

  newHit->Print();
  return true;
  // G4StepPoint *postStepPoint = aStep->GetPostStepPoint();

  // G4ThreeVector postPosPhoton = postStepPoint->GetPosition();

  // G4double postEnergy = postStepPoint->GetKineticEnergy();
  // // G4double totalEnergy = postStepPoint->GetTotalEnergy();

  // G4cout<< "Photon position: " << postPosPhoton  << G4endl;
  // G4cout<< "Photon energy: " << postEnergy  << G4endl;
  // return 0;
}

void MySensitiveDetector::EndOfEvent(G4HCofThisEvent*)
{
  // G4int nofHits = fHitsCollection->entries();
  // G4cout << G4endl
  //        << "-------->Hits Collection: in this event they are " << nofHits
  //        << " hits in the tracker chambers: " << G4endl;
  // for ( G4int i=0; i<nofHits; i++ ) (*fHitsCollection)[i]->Print();
}
