#include "construction.hh"

MyDetectorConstruction::MyDetectorConstruction(){}
MyDetectorConstruction::~MyDetectorConstruction(){}

G4VPhysicalVolume* MyDetectorConstruction::Construct()
{
  G4NistManager *nist = G4NistManager::Instance();

  // Define the World Material
  G4Material *worldMat = nist->FindOrBuildMaterial("G4_AIR");

  // Construct the World volume
  G4double worldSize=1*m;
  G4Box *solidWorld = new G4Box("solidWorld", 0.5*worldSize, 0.5*worldSize, 0.5*worldSize);
  G4LogicalVolume *logicWorld = new G4LogicalVolume(solidWorld, worldMat, "logicWorld");
  G4VPhysicalVolume *physWorld = new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), logicWorld, "physWorld", 0, false, 0, true);

  // Define Water material
  G4Element* elH  = new G4Element("Hydrogen","H" ,  1., 1.01*g/mole);
  G4Element* elO  = new G4Element("Oxygen", "O" ,  8., 16.00*g/mole);
  G4Material* H2O = new G4Material("Water",1.000*g/cm3, 2);

  H2O->AddElement(elH, 2);
  H2O->AddElement(elO, 1);
  // G4cout << "H2O is " << H2O << G4endl;

  // Array of sensitive detectors
  G4double waterBoxSize=0.8*worldSize;
  G4int numberOfVoxelsPerAxis=9;
  G4double voxelSize=waterBoxSize/numberOfVoxelsPerAxis;

  G4Box *solidDetector = new G4Box("solidDetector", // name
                                   voxelSize*0.5,       // half width
                                   voxelSize*0.5,       // half length
                                   voxelSize*0.5);      // half height

  logicDetector = new G4LogicalVolume(solidDetector, H2O, "logicDetector");

  for(G4int i = 0; i<numberOfVoxelsPerAxis; i++){
    for(G4int j=0; j<numberOfVoxelsPerAxis; j++){
      for(G4int k=0; k<numberOfVoxelsPerAxis; k++){
      G4VPhysicalVolume *physDetector = new G4PVPlacement(0,  // rotation
                                                          G4ThreeVector(-waterBoxSize*0.5+voxelSize*0.5+i*voxelSize,
                                                                        -waterBoxSize*0.5+voxelSize*0.5+j*voxelSize,
                                                                        -waterBoxSize*0.5+voxelSize*0.5+k*voxelSize),  // position
                                                          logicDetector, // attached logical volume
                                                          "physDetector", // name
                                                          logicWorld, // mother world
                                                          false, // boolean
                                                          i+j*10+k*100, // copy number
                                                          true); // check overlaps
      }
    }
  }
  return physWorld;
}

void MyDetectorConstruction::ConstructSDandField()
{
  MySensitiveDetector *sensDet = new MySensitiveDetector("SensitiveDetector", "myHisCollection");
  logicDetector->SetSensitiveDetector(sensDet);
}
