#!/bin/bash

#curl localhost:5600 --data-binary "@/datadrive/odota-parser/7355186741_1742953546.dem"
#curl localhost:5600 --data-binary "@/datadrive/odota-parser/6298221747_1156784242.dem"

REPLAY_PATH="/Users/Xavi/Documents/GitHub/dota2coach/Replays/8360992551.dem"
curl -X POST --data-binary @"$REPLAY_PATH" localhost:5600


#curl localhost:5600/blob?replay_url=http://replay118.valve.net/570/8218938996_1424352668.dem.bz2