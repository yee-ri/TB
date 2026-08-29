#!/usr/bin/env python3
import cv2
import yaml
import numpy as np
import math

PGM_PATH='/home/sj/catkin_ws/src/turtlebot3_simulations/maze/tunnel.pgm'
YAML_PATH='/home/sj/catkin_ws/src/turtlebot3_simulations/maze/tunnel.yaml'

with open(YAML_PATH,'r') as f:
    info=yaml.safe_load(f)

res=info['resolution']
origin_x=info['origin'][0]
origin_y=info['origin'][1]

img=cv2.imread(PGM_PATH,cv2.IMREAD_GRAYSCALE)

occupied=np.zeros_like(img)
occupied[img<50]=255

contours,_=cv2.findContours(occupied,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

points=[]

for contour in contours:
    if cv2.arcLength(contour,False)<20:
        continue

    eps=3
    approx=cv2.approxPolyDP(contour,eps,False)

    for p in approx:
        x=int(p[0][0])
        y=int(p[0][1])
        points.append((x,y))

unique=[]

for p in points:
    duplicated=False

    for q in unique:
        if math.hypot(p[0]-q[0],p[1]-q[1])<5:
            duplicated=True
            break

    if not duplicated:
        unique.append(p)

print('WALL END POINTS')

for x,y in unique:
    wx=origin_x+x*res
    wy=origin_y+(img.shape[0]-y)*res

    print(
        'pixel=(%d,%d) map=(%.3f,%.3f)'
        %(x,y,wx,wy)
    )

candidates=[]

for i in range(len(unique)):
    for j in range(i+1,len(unique)):
        x1,y1=unique[i]
        x2,y2=unique[j]

        dist_px=math.hypot(x2-x1,y2-y1)
        dist_m=dist_px*res

        if 0.3<dist_m<0.6:
            cx=(x1+x2)/2.0
            cy=(y1+y2)/2.0

            wx=origin_x+cx*res
            wy=origin_y+(img.shape[0]-cy)*res

            candidates.append((dist_m,wx,wy,x1,y1,x2,y2))

print()
print('OPENING CANDIDATES')

for width,x,y,x1,y1,x2,y2 in candidates:
    print(
        'width=%.2f map=(%.3f,%.3f) endpoints=(%d,%d)-(%d,%d)'
        %(width,x,y,x1,y1,x2,y2)
    )