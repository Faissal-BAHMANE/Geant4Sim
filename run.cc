#include "run.hh"

MyRunAction::MyRunAction(){
  // G4AnalysisManager *man = G4AnalysisManager::Instance();
  // man->CreateNtuple("Hits","Hits");
  // man->CreateNtupleIColumn("Event");
  // man->CreateNtupleDColumn("positionX");
  // man->CreateNtupleDColumn("positionY");
  // man->CreateNtupleDColumn("positionZ");
  // man->CreateNtupleDColumn("energy");
  // man->FinishNtuple(0);

  // man->CreateNtuple("Scoring", "Scoring");
  // man->CreateNtupleDColumn("EnergyDeposit");
  // man->FinishNtuple(1);
}
MyRunAction::~MyRunAction(){}

void MyRunAction::BeginOfRunAction(const G4Run*)
{
  // G4AnalysisManager *man = G4AnalysisManager::Instance();
  // G4int runID = run->GetRunID();
  // std::stringstream strRunID;
  // strRunID << runID;

  // man->OpenFile("output"+strRunID.str()+".root");
}

void MyRunAction::EndOfRunAction(const G4Run*)
{
  // G4AnalysisManager *man = G4AnalysisManager::Instance();
  // man->Write();
  // man->CloseFile();
}
