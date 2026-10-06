#!/usr/bin/env python
import math
import heapq
import numpy as np

class MazeNavigation:
    def __init__(self,resolution=0.05,size=160):
        self.resolution=resolution
        self.size=size
        self.map=np.zeros((size,size),dtype=np.int8)
        self.value=np.full((size,size),np.inf)
        self.points=[]
        self.path=[]
        self.start_x=None
        self.start_y=None
        self.start_yaw=None
        self.goal_x=0.465
        self.goal_y=0.53
        self.goal_yaw=math.radians(85)
        self.avoid_direction=None
        
        self.final_phase=0
        self.approach_distance=0.15

    def reset(self):
        self.map[:]=0
        self.value[:]=np.inf
        self.points=[]
        self.path=[]
        self.start_x=None
        self.start_y=None
        self.avoid_direction=None
        self.start_yaw=None
        self.final_phase=0

    def update_lidar(self,points):
        self.points=points

    def relative_pose(self,odom_x,odom_y,yaw):
        dx=odom_x-self.start_x
        dy=odom_y-self.start_y
        c=math.cos(self.start_yaw)
        s=math.sin(self.start_yaw)
        x=c*dx+s*dy
        y=-s*dx+c*dy
        a=math.atan2(math.sin(yaw-self.start_yaw),math.cos(yaw-self.start_yaw))
        return x,y,a

    def update_map(self,odom_x,odom_y,yaw):
        self.map[:]=0
        center=self.size//2
        rx,ry,ra=self.relative_pose(odom_x,odom_y,yaw)
        c=math.cos(ra)
        s=math.sin(ra)

        for x,y in self.points:
            gx=rx+c*x-s*y
            gy=ry+s*x+c*y
            mx=center+int(gx/self.resolution)
            my=center-int(gy/self.resolution)

            if 3<=mx<self.size-3 and 3<=my<self.size-3:
                self.map[my-4:my+5,mx-4:mx+5]=np.maximum(self.map[my-4:my+5,mx-4:mx+5],30)
                self.map[my-2:my+3,mx-2:mx+3]=np.maximum(self.map[my-2:my+3,mx-2:mx+3],75)
                self.map[my,mx]=100

        sx=center+int(rx/self.resolution)
        sy=center-int(ry/self.resolution)
        gx=center+int(self.goal_x/self.resolution)
        gy=center-int(self.goal_y/self.resolution)
        self.map[max(0,sy-1):min(self.size,sy+2),max(0,sx-1):min(self.size,sx+2)]=0
        self.map[gy,gx]=0

        # self.map[max(0,sy-3):min(self.size,sy+4),max(0,sx-3):min(self.size,sx+4)]=0
        # self.map[max(0,gy-3):min(self.size,gy+4),max(0,gx-3):min(self.size,gx+4)]=0

    def cells(self,odom_x,odom_y,yaw):
        center=self.size//2
        rx,ry,_=self.relative_pose(odom_x,odom_y,yaw)
        sx=center+int(rx/self.resolution)
        sy=center-int(ry/self.resolution)
        gx=center+int(self.goal_x/self.resolution)
        gy=center-int(self.goal_y/self.resolution)
        print("CELL RXRY",rx,ry)
        return (sy,sx),(gy,gx)

    def update_value(self,start):
        self.value[:]=np.inf
        sy,sx=start
        if not (0<=sx<self.size and 0<=sy<self.size):return

        self.value[sy,sx]=0.0
        queue=[(0.0,sy,sx)]
        moves=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        while queue:
            value,y,x=heapq.heappop(queue)
            if value>self.value[y,x]:continue

            for dy,dx in moves:
                ny=y+dy
                nx=x+dx
                if not (0<=nx<self.size and 0<=ny<self.size):continue

                if dx!=0 and dy!=0:
                    if self.map[y,nx]>=100 or self.map[ny,x]>=100:continue

                cost=self.map[ny,nx]
                if cost>=100:continue
                if cost==0:cost=1

                new_value=value+math.hypot(dx,dy)*cost

                if new_value<self.value[ny,nx]:
                    self.value[ny,nx]=new_value
                    heapq.heappush(queue,(new_value,ny,nx))

    def make_path(self,start,goal):
        sy,sx=start
        y,x=goal

        if not (0<=x<self.size and 0<=y<self.size):return None
        if not np.isfinite(self.value[y,x]):return None

        path=[(y,x)]
        moves=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        for _ in range(500):
            if abs(x-sx)<=1 and abs(y-sy)<=1:
                path.append((sy,sx))
                path.reverse()
                return path

            best=None
            best_value=self.value[y,x]

            for dy,dx in moves:
                ny=y+dy
                nx=x+dx
                if not (0<=nx<self.size and 0<=ny<self.size):continue

                if self.value[ny,nx]<best_value:
                    best_value=self.value[ny,nx]
                    best=(ny,nx)

            if best is None:
                print("PATH STUCK y=",y,"x=",x,"value=",self.value[y,x],"start=",start,"goal=",goal)
                return None
            y,x=best
            path.append((y,x))

        return None

    def line_of_sight(self,p1,p2):
        y0,x0=p1
        y1,x1=p2
        distance=max(abs(x1-x0),abs(y1-y0))
        if distance==0:return True

        for i in range(distance+1):
            t=float(i)/distance
            x=int(round(x0+(x1-x0)*t))
            y=int(round(y0+(y1-y0)*t))
            if not (0<=x<self.size and 0<=y<self.size):return False
            if self.map[y,x]>=75:return False

        return True

    def enhance_path(self,path):
        if path is None or len(path)<3:return path

        result=[path[0]]
        i=0

        while i<len(path)-1:
            j=len(path)-1

            while j>i+1:
                if self.line_of_sight(path[i],path[j]):break
                j-=1

            result.append(path[j])
            i=j

        return result

    # def velocity(self,odom_x,odom_y,yaw,path):
    #     if path is None or len(path)<2:return 0.0,0.0

    #     center=self.size//2
    #     rx,ry,ra=self.relative_pose(odom_x,odom_y,yaw)
    #     target=path[1]

    #     for p in path[1:]:
    #         tx=(p[1]-center)*self.resolution
    #         ty=(center-p[0])*self.resolution

    #         if math.hypot(tx-rx,ty-ry)>=0.20:
    #             target=p
    #             break

    #     tx=(target[1]-center)*self.resolution
    #     ty=(center-target[0])*self.resolution
    #     target_yaw=math.atan2(ty-ry,tx-rx)
    #     error=math.atan2(math.sin(target_yaw-ra),math.cos(target_yaw-ra))
    #     angular=float(np.clip(error*1.5,-0.35,0.35))

    #     if abs(error)>0.65:return 0.01,angular
    #     if abs(error)>0.35:return 0.03,angular
    #     return 0.06,angular


    def velocity(self,odom_x,odom_y,yaw,left_distance,front_distance,right_distance):
        rx,ry,ra=self.relative_pose(odom_x,odom_y,yaw)

        dx=self.goal_x-rx
        dy=self.goal_y-ry
        target_yaw=math.atan2(dy,dx)
        error=math.atan2(math.sin(target_yaw-ra),math.cos(target_yaw-ra))

        if self.avoid_direction is None and front_distance<=0.25:
            self.avoid_direction='left' if left_distance>right_distance else 'right'

        if self.avoid_direction is not None:
            if front_distance>0.45:
                self.avoid_direction=None
            elif self.avoid_direction=='left':
                return 0.025,0.22
            elif self.avoid_direction=='right':
                return 0.025,-0.22

        if left_distance<0.22:return 0.04,-0.2
        if right_distance<0.22:return 0.04,0.2

        angular=float(np.clip(error*1.3,-0.30,0.30))

        if abs(error)>0.7:return 0.0,angular
        if abs(error)>0.35:return 0.03,angular
        return 0.06,angular


    def run(self,odom_x,odom_y,yaw,left_distance,front_distance,right_distance):
        if self.start_x is None:
            self.start_x=odom_x
            self.start_y=odom_y
            self.start_yaw=yaw
            print("MAZE INIT",self.start_x,self.start_y,self.start_yaw)
            return 0.0,0.0,False

        rx,ry,ra=self.relative_pose(odom_x,odom_y,yaw)
        print("REL",rx,ry,ra)

        goal_distance=math.hypot(self.goal_x-rx,self.goal_y-ry)

        approach_x=self.goal_x-math.cos(self.goal_yaw)*self.approach_distance
        approach_y=self.goal_y-math.sin(self.goal_yaw)*self.approach_distance

        if self.final_phase==0 and goal_distance<0.25:
            self.final_phase=1

        if self.final_phase==1:
            dx=approach_x-rx
            dy=approach_y-ry
            approach_distance=math.hypot(dx,dy)

            if approach_distance<0.01:
                self.final_phase=2
                return 0.0,0.0,False

            target_yaw=math.atan2(dy,dx)
            yaw_error=target_yaw-ra
            yaw_error=math.atan2(math.sin(yaw_error),math.cos(yaw_error))
            angular=float(np.clip(yaw_error*1.5,-0.15,0.15))

            if abs(yaw_error)>math.radians(25):
                linear=0.0
            else:
                linear=0.05

            return linear,angular,False

        if self.final_phase==2:
            yaw_error=self.goal_yaw-ra
            yaw_error=math.atan2(math.sin(yaw_error),math.cos(yaw_error))

            if abs(yaw_error)<math.radians(4):
                self.final_phase=3
                return 0.0,0.0,False

            angular=float(np.clip(yaw_error*1.0,-0.15,0.15))
            return 0.0,angular,False

        if self.final_phase==3:
            dx=self.goal_x-rx
            dy=self.goal_y-ry

            forward_error=math.cos(self.goal_yaw)*dx+math.sin(self.goal_yaw)*dy
            lateral_error=-math.sin(self.goal_yaw)*dx+math.cos(self.goal_yaw)*dy

            print("FINAL ERROR",forward_error,lateral_error)

            if abs(forward_error)<0.01 and abs(lateral_error)<0.015:
                return 0.0,0.0,True

            yaw_error=self.goal_yaw-ra
            yaw_error=math.atan2(math.sin(yaw_error),math.cos(yaw_error))
            angular=float(np.clip(yaw_error*1.5,-0.15,0.15))

            if forward_error>0.01:
                linear=0.04
            elif forward_error<-0.01:
                linear=-0.035
            else:
                linear=0.0

            return linear,angular,False

        linear,angular=self.velocity(odom_x,odom_y,yaw,left_distance,front_distance,right_distance)
        print("GOAL DIST",goal_distance,"CMD",linear,angular)
        return linear,angular,False