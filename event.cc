#include "event.hh"
#include "G4UnitsTable.hh"

MyEventAction::MyEventAction(MyRunAction*)
{
}

MyEventAction::~MyEventAction(){}


void MyEventAction::BeginOfEventAction(const G4Event*)
{
}

void MyEventAction::EndOfEventAction(const G4Event* event)
{
  // get number of stored trajectories

  G4TrajectoryContainer* trajectoryContainer = event->GetTrajectoryContainer();
  G4int n_trajectories = 0;
  if (trajectoryContainer) n_trajectories = trajectoryContainer->entries();

  // periodic printing

  G4int eventID = event->GetEventID();
  G4cout << ">>> Event: " << eventID  << G4endl;
  if ( trajectoryContainer ) {
    G4cout << "    " << n_trajectories
           << " trajectories stored in this event." << G4endl;
  }

  // G4VHitsCollection* hc = event->GetHCofThisEvent()->GetHC(0);
  // G4cout << "    "
  //        << hc->GetSize() << " hits stored in this event" << G4endl;

}
