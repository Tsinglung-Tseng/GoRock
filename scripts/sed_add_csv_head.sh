#!/bin/bash

sed -i '/PDGEncoding/d' hits_80crystal.csv 
sed -i '1 i\PDGEncoding,trackID,parentID,eventID,crystalID,photonID,processName,edep,posX,posY,posZ,localPosX,localPosY,localPosZ,sourcePosX,sourcePosY,sourcePosZ' hits_80crystal.csv 
