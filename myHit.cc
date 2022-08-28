#include "myHit.hh"
#include "G4UnitsTable.hh"
#include <fstream>

// #include "G4VVisManager.hh"
// #include "G4Circle.hh"
// #include "G4Colour.hh"
// #include "G4VisAttributes.hh"
// #include <iomanip>

G4ThreadLocal G4Allocator<myHit>* myHitAllocator = nullptr;


myHit::myHit()
{}


myHit::~myHit() {}


G4bool myHit::operator==(const myHit& right) const
{
  return ( this == &right ) ? true : false;
}


void myHit::Draw()
{
  // G4VVisManager* pVVisManager = G4VVisManager::GetConcreteInstance();
  // if(pVVisManager)
  // {
  //   G4Circle circle(fPos);
  //   circle.SetScreenSize(4.);
  //   circle.SetFillStyle(G4Circle::filled);
  //   G4Colour colour(1.,0.,0.);
  //   G4VisAttributes attribs(colour);
  //   circle.SetVisAttributes(attribs);
  //   pVVisManager->Draw(circle);
  // }
}


void myHit::Print()
{

  std::ofstream logFileCsv;
  logFileCsv.open("hits_csv.csv", std::ios::app);
  logFileCsv
    << fChamberNb <<"," << fEdep << G4endl;
  logFileCsv.close();

  std::ofstream logFile;
  logFile.open("hits_log", std::ios::app);
  logFile
    << "trackID: "  << std::setw(7) << fTrackID
    << "chamberNb: " << std::setw(7) << fChamberNb
    << "Edep: "       << std::setw(7) << G4BestUnit(fEdep,"Energy")
    << "Position: " << std::setw(7) << G4BestUnit( fPos,"Length")
    << G4endl;
  logFile.close();

  G4cout
     << "trackID: "  << std::setw(7) << fTrackID
     << "  chamberNb: " << std::setw(7) << fChamberNb
     << "  Edep: "       << std::setw(7) << G4BestUnit(fEdep,"Energy")
     << "  Position: "  << std::setw(7) << G4BestUnit( fPos,"Length")
     << G4endl;
}
