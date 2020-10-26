/////////////////////////////////////////////////////   
//                                                 //
//			S.JAN - sjan@cea.fr - March 2007	   //
//			updated by							   //
//			U. Pietrzyk - u.pietrzyk@fz-juelich.de //
//			March 2010							   //
//                                                 //
//   Example of a ROOT C++ code-macro to:          // 
//   -----------------------------------           //
//   1/ read an output root data file              //
//   2/ create a loop on each event which are      //
//      stored during the simulation               //
//   3/ perform data processing                    //
//   4/ plot the results                           //
//                                                 //
/////////////////////////////////////////////////////
#include <fstream>
#include <iomanip>
#include <stdio.h>

void root2csv4()
{
 
   gROOT->Reset();
	TFile *f = new TFile("LUTDavisModel_80_25mm_crystal.root");
   //output the true coincidence event(position) into a .txt file
	std::ofstream outhits("hits.csv", std::ios_base::out);
	// std::ofstream outoptical("optical_64crystal.csv", std::ios_base::out);
	
   //std::ofstream outhdr("normal_scan_true_only_true.txt.hdr");

	//TTree *Coincidences = (TTree*)gDirectory->Get("Coincidences");
   //TTree *Gate = (TTree*)gDirectory->Get("Gate");
   TTree *Hits = (TTree*)gDirectory->Get("Hits");
   // TTree *OpticalData = (TTree*)gDirectory->Get("OpticalData");   
   //TTree *Singles = (TTree*)gDirectory->Get("Singles");
   
//
//Declaration of leaves types - TTree Hits
//  
   Int_t           PDGEncoding;
   Int_t           trackID;
   Int_t           parentID;
   // Double_t         trackLocalTime;
   // Double_t        time1;
   Float_t         edep;
   // Float_t           stepLength;
   // Float_t           trackLength;
   Float_t         posX;
   Float_t         posY;
   Float_t         posZ;
   Float_t         localPosX;
   Float_t         localPosY;
   Float_t         localPosZ;
   // Float_t         momDirX;   
   // Float_t         momDirY;
   // Float_t         momDirZ;
   // Int_t           headID;
   Int_t           crystalID;
   // Int_t           pixelID;
   Int_t           photonID;
   // Int_t           nPhantomCompton;
   // Int_t           nCrystalCompton;
   // Int_t           nPhantomRayleigh;
   // Int_t           nCrystalRayleigh;
   // Int_t           primaryID; 
   Float_t        sourcePosX;
   Float_t        sourcePosY;
   Float_t        sourcePosZ;  
   // Int_t           sourceID;
   Int_t           eventID;
   // Int_t           runID;
   // Float_t         axialPos;
   // Float_t         rotationAngle; 
   // Float_t           volumeID; 
   Char_t          processName[40];
   // Char_t          comptVolName[40];
   // Char_t          RayleighVolName[40];

//Declaration of leaves types - TTree OpticalData

   // Int_t           NumScintillation;
   // Int_t           NumCrystalWLS;
   // Int_t           NumPhantomWLS;
   // Double_t         CrystalLastHitPos_X;
   // Double_t         CrystalLastHitPos_Y;
   // Double_t         CrystalLastHitPos_Z;
   // Double_t         CrystalLastHitEnergy;
   // Double_t         PhantomLastHitPos_X;
   // Double_t         PhantomLastHitPos_Y;
   // Double_t         PhantomLastHitPos_Z;
   // Double_t         PhantomLastHitEnergy;
   // Double_t         PhantomWLSPos_X;
   // Double_t         PhantomWLSPos_Y;
   // Double_t         PhantomWLSPos_Z;
   // Char_t          PhantomProcessName[40];
   // Char_t          CrystalProcessName[40];
   // Double_t         MomentumDirectionx;
   // Double_t         MomentumDirectiony;
   // Double_t         MomentumDirectionz;
   
//   
//Set branch addresses - TTree Hits
//  
   Hits->SetBranchAddress("PDGEncoding",&PDGEncoding);
   Hits->SetBranchAddress("trackID",&trackID);
   Hits->SetBranchAddress("parentID",&parentID);
   // Hits->SetBranchAddress("trackLocalTime",&trackLocalTime);
   // Hits->SetBranchAddress("time",&time1);
   Hits->SetBranchAddress("edep",&edep);
   // Hits->SetBranchAddress("stepLength",&stepLength);
   // Hits->SetBranchAddress("trackLength",&trackLength);
   Hits->SetBranchAddress("posX",&posX);
   Hits->SetBranchAddress("posY",&posY);
   Hits->SetBranchAddress("posZ",&posZ);
   Hits->SetBranchAddress("localPosX",&localPosX);
   Hits->SetBranchAddress("localPosY",&localPosY);
   Hits->SetBranchAddress("localPosZ",&localPosZ);
   // Hits->SetBranchAddress("momDirX",&momDirX);
   // Hits->SetBranchAddress("momDirY",&momDirY);
   // Hits->SetBranchAddress("momDirZ",&momDirZ);
   // Hits->SetBranchAddress("headID",&headID);
   Hits->SetBranchAddress("crystalID",&crystalID);
   // Hits->SetBranchAddress("pixelID",&pixelID);
   Hits->SetBranchAddress("photonID",&photonID);
   // Hits->SetBranchAddress("nPhantomCompton",&nPhantomCompton);
   // Hits->SetBranchAddress("nCrystalCompton",&nCrystalCompton);
   // Hits->SetBranchAddress("nPhantomRayleigh",&nPhantomRayleigh);
   // Hits->SetBranchAddress("nCrystalRayleigh",&nCrystalRayleigh);
   // Hits->SetBranchAddress("primaryID",&primaryID);
   Hits->SetBranchAddress("sourcePosX",&sourcePosX);
   Hits->SetBranchAddress("sourcePosY",&sourcePosY);
   Hits->SetBranchAddress("sourcePosZ",&sourcePosZ);
   // Hits->SetBranchAddress("sourceID",&sourceID);
   Hits->SetBranchAddress("eventID",&eventID);
   // Hits->SetBranchAddress("runID",&runID);
   // Hits->SetBranchAddress("rotationAngle",&rotationAngle);
   // Hits->SetBranchAddress("volumeID",&volumeID);
   Hits->SetBranchAddress("processName",&processName);
   // Hits->SetBranchAddress("comptVolName",&comptVolName);
   // Hits->SetBranchAddress("RayleighVolName",&RayleighVolName);
   
//   
//Set branch addresses - TTree OpticalData
//
   // OpticalData->SetBranchAddress("NumScintillation",&NumScintillation);
   // OpticalData->SetBranchAddress("NumCrystalWLS",&NumCrystalWLS);
   // OpticalData->SetBranchAddress("NumPhantomWLS",&NumPhantomWLS);
   // OpticalData->SetBranchAddress("CrystalLastHitPos_X",&CrystalLastHitPos_X);
   // OpticalData->SetBranchAddress("CrystalLastHitPos_Y",&CrystalLastHitPos_Y);
   // OpticalData->SetBranchAddress("CrystalLastHitPos_Z",&CrystalLastHitPos_Z);
   // OpticalData->SetBranchAddress("CrystalLastHitEnergy",&CrystalLastHitEnergy);
   // OpticalData->SetBranchAddress("PhantomLastHitPos_X",&PhantomLastHitPos_X);
   // OpticalData->SetBranchAddress("PhantomLastHitPos_Y",&PhantomLastHitPos_Y);
   // OpticalData->SetBranchAddress("PhantomLastHitPos_Z",&PhantomLastHitPos_Z);
   // OpticalData->SetBranchAddress("PhantomLastHitEnergy",&PhantomLastHitEnergy);
   // OpticalData->SetBranchAddress("PhantomWLSPos_X",&PhantomWLSPos_X);
   // OpticalData->SetBranchAddress("PhantomWLSPos_Y",&PhantomWLSPos_Y);
   // OpticalData->SetBranchAddress("PhantomWLSPos_Z",&PhantomWLSPos_Z);
   // OpticalData->SetBranchAddress("PhantomProcessName",&PhantomProcessName);
   // OpticalData->SetBranchAddress("CrystalProcessName",&CrystalProcessName);
   // OpticalData->SetBranchAddress("MomentumDirectionx",&MomentumDirectionx);
   // OpticalData->SetBranchAddress("MomentumDirectiony",&MomentumDirectiony);
   // OpticalData->SetBranchAddress("MomentumDirectionz",&MomentumDirectionz);


   Int_t nentries = Hits->GetEntries();
   // Int_t nOptical = OpticalData->GetEntries();
	cout<<" Hits:   "<<  nentries <<endl;
  	// cout<<" OpticalData     :   "<<  nOptical <<endl;
   
   // Int_t nbytes = 0;
   // Int_t nsipm = 0;
   // Int_t nbytes2 = 0; 
//
//loop on the events in the TTree Hits
//

if(outhits.is_open())
{
   outhits << "PDGEncoding" << ',' << "trackID" << ',' << "parentID" << ',' << "eventID" << ',' << "crystalID" << ',' << "photonID" << ',' << "processName" << ',' << "edep" << ',' << "posX" << ',' << "posY" << ',' << "posZ" << ',' << "localPosX" << ',' << "localPosY" << ',' << "localPosZ" << ',' << "sourcePosX"<< ',' << "sourcePosY"<< ',' << "sourcePosZ" << endl;
	
   for (Int_t i=0; i<nentries;i++) {
    Hits->GetEntry(i);
   //  if((posZ>=12.56) || (posZ<=-12.56)){
   //      nsipm += 1;}
      // cout<<" source     :   "<<  sourcePosY <<endl;


   outhits << PDGEncoding << ',' << trackID << ',' << parentID << ',' << eventID << ',' << crystalID << ',' << photonID << ',' << processName << ',' << edep << ',' << posX << ',' << posY << ',' << posZ << ',' << localPosX << ',' << localPosY << ',' << localPosZ   << ',' << sourcePosX<< ',' << sourcePosY<< ',' << sourcePosZ << endl;	
   }
	
}

outhits.close();


// if(outoptical.is_open())
// {
     
//    outoptical << "NumScintillation" << ',' << "NumCrystalWLS" << ',' << "NumPhantomWLS" << ',' << "CrystalLastHitPos_X" << ',' << "CrystalLastHitPos_Y" << ',' << "CrystalLastHitPos_Z" \
//    << ',' << "CrystalLastHitEnergy" << ',' << "PhantomLastHitPos_X" << ',' << "PhantomLastHitPos_Y" << ',' << "PhantomLastHitPos_Z" << ',' << "PhantomLastHitEnergy" << ',' << "PhantomWLSPos_X" \
//    << ',' << "PhantomWLSPos_Y"<< ',' << "PhantomWLSPos_Z"<< ',' << "PhantomProcessName"<< ',' << "CrystalProcessName"<< ',' << "MomentumDirectionx" << ',' << "MomentumDirectiony" \
//    << ',' << "MomentumDirectionz"<< endl;	


//    for (Int_t i=0; i<nOptical;i++) {
//    	nbytes2 += OpticalData->GetEntry(i);
//       // cout<<" source     :   "<<  sourcePosY <<endl;
//       outoptical << NumScintillation << ',' << NumCrystalWLS << ',' << NumPhantomWLS << ',' << CrystalLastHitPos_X << ',' << CrystalLastHitPos_Y << ',' << CrystalLastHitPos_Z \
//    << ',' << CrystalLastHitEnergy << ',' << PhantomLastHitPos_X << ',' << PhantomLastHitPos_Y << ',' << PhantomLastHitPos_Z << ',' << PhantomLastHitEnergy << ',' << PhantomWLSPos_X \
//    << ',' << PhantomWLSPos_Y<< ',' << PhantomWLSPos_Z<< ',' << PhantomProcessName<< ',' << CrystalProcessName<< ',' << MomentumDirectionx << ',' << MomentumDirectiony \
//    << ',' << MomentumDirectionz<< endl;	
//    }

// }
// outoptical.close();

// cout<<" Hits_in_sipm:"<< nsipm <<endl;
}
