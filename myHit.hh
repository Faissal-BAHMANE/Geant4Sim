#ifndef MYHIT_HH
#define MYHIT_HH

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
// #include "tls.hh"

class myHit : public G4VHit
{
  public:
    myHit();
    myHit(const myHit&) = default;
    ~myHit() override;

    // operators
    myHit& operator=(const myHit&) = default;
    G4bool operator==(const myHit&) const;

    inline void* operator new(size_t);
    inline void  operator delete(void*);

    // methods from base class
    void Draw() override;
    void Print() override;

    // Set methods
    void SetTrackID  (G4int track)      { fTrackID = track; };
    void SetChamberNb(G4int chamb)      { fChamberNb = chamb; };
    void SetEdep     (G4double de)      { fEdep = de; };
    void SetPos      (G4ThreeVector xyz){ fPos = xyz; };

    // Get methods
    G4int GetTrackID() const     { return fTrackID; };
    G4int GetChamberNb() const   { return fChamberNb; };
    G4double GetEdep() const     { return fEdep; };
    G4ThreeVector GetPos() const { return fPos; };

  private:
    G4int         fTrackID = -1;
    G4int         fChamberNb = -1;
    G4double      fEdep = 0.;
    G4ThreeVector fPos;
};


typedef G4THitsCollection<myHit> myHitsCollection;

extern G4ThreadLocal G4Allocator<myHit>* myHitAllocator;


inline void* myHit::operator new(size_t)
{
  if(!myHitAllocator)
      myHitAllocator = new G4Allocator<myHit>;
  return (void *) myHitAllocator->MallocSingle();
}


inline void myHit::operator delete(void *hit)
{
  myHitAllocator->FreeSingle((myHit*) hit);
}

#endif //MYHIT_HH
