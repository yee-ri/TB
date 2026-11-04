#!/usr/bin/env python
import math
import heapq
import numpy as np

class MazeNavigation:
    def __init__(self):
        self.maze_resolution=0.05
        self.maze_size=160

        self.maze_map=np.zeros((self.maze_size,self.maze_size),dtype=np.int8)
        self.maze_value=np.full((self.maze_size,self.maze_size),np.inf)

        self.maze_points=[]
        self.maze_path=[]

        self.maze_start_x=None
        self.maze_start_y=None
        self.maze_start_yaw=None

        # 미로 입구 기준 목표 위치
        self.maze_goal_x=2.0
        self.maze_goal_y=1.5

    def update_lidar(self,points):
        self.maze_points=points

    def get_relative_pose(self,odom_x,odom_y,yaw):
        dx=odom_x-self.maze_start_x
        dy=odom_y-self.maze_start_y

        ca=math.cos(self.maze_start_yaw)
        sa=math.sin(self.maze_start_yaw)

        rx=ca*dx+sa*dy
        ry=-sa*dx+ca*dy
        ryaw=math.atan2(math.sin(yaw-self.maze_start_yaw),math.cos(yaw-self.maze_start_yaw))

        return rx,ry,ryaw

    def update_maze_map(self,odom_x,odom_y,yaw):
        c=self.maze_size//2
        rx,ry,ryaw=self.get_relative_pose(odom_x,odom_y,yaw)

        ca=math.cos(ryaw)
        sa=math.sin(ryaw)

        for x,y in self.maze_points:
            gx=rx+ca*x-sa*y
            gy=ry+sa*x+ca*y

            mx=c+int(gx/self.maze_resolution)
            my=c-int(gy/self.maze_resolution)

            if 3<=mx<self.maze_size-3 and 3<=my<self.maze_size-3:
                self.maze_map[my,mx]=100
                self.maze_map[my-1:my+2,mx-1:mx+2]=100
                self.maze_map[my-2:my+3,mx-2:mx+3]=np.maximum(self.maze_map[my-2:my+3,mx-2:mx+3],75)

    def update_maze_value(self,start):
        h,w=self.maze_map.shape
        self.maze_value[:]=np.inf

        sy,sx=start

        if not (0<=sx<w and 0<=sy<h):
            return

        self.maze_value[sy,sx]=0.0

        queue=[(0.0,sy,sx)]
        moves=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        while queue:
            value,y,x=heapq.heappop(queue)

            if value>self.maze_value[y,x]:
                continue

            for dy,dx in moves:
                ny=y+dy
                nx=x+dx

                if not (0<=nx<w and 0<=ny<h):
                    continue

                cost=self.maze_map[ny,nx]

                if cost>=100:
                    continue

                if cost==0:
                    cost=1

                distance=math.hypot(dx,dy)
                new_value=value+distance*cost

                if new_value<self.maze_value[ny,nx]:
                    self.maze_value[ny,nx]=new_value
                    heapq.heappush(queue,(new_value,ny,nx))

    def make_maze_path(self,start,goal):
        sy,sx=start
        y,x=goal

        if not (0<=x<self.maze_size and 0<=y<self.maze_size):
            return None

        if not np.isfinite(self.maze_value[y,x]):
            return None

        path=[(y,x)]
        moves=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        for _ in range(500):
            if abs(x-sx)<=1 and abs(y-sy)<=1:
                path.append((sy,sx))
                path.reverse()
                return path

            best=None
            best_value=self.maze_value[y,x]

            for dy,dx in moves:
                ny=y+dy
                nx=x+dx

                if not (0<=nx<self.maze_size and 0<=ny<self.maze_size):
                    continue

                if self.maze_value[ny,nx]<best_value:
                    best_value=self.maze_value[ny,nx]
                    best=(ny,nx)

            if best is None:
                return None

            y,x=best
            path.append((y,x))

        return None

    def maze_line_of_sight(self,p1,p2):
        y0,x0=p1
        y1,x1=p2

        distance=max(abs(x1-x0),abs(y1-y0))

        if distance==0:
            return True

        for i in range(distance+1):
            t=float(i)/distance
            x=int(round(x0+(x1-x0)*t))
            y=int(round(y0+(y1-y0)*t))

            if not (0<=x<self.maze_size and 0<=y<self.maze_size):
                return False

            if self.maze_map[y,x]>1:
                return False

        return True

    def enhance_maze_path(self,path):
        if path is None or len(path)<3:
            return path

        result=[path[0]]
        i=0

        while i<len(path)-1:
            j=len(path)-1

            while j>i+1:
                if self.maze_line_of_sight(path[i],path[j]):
                    break
                j-=1

            result.append(path[j])
            i=j

        return result

    def get_maze_cells(self,odom_x,odom_y,yaw):
        c=self.maze_size//2
        rx,ry,_=self.get_relative_pose(odom_x,odom_y,yaw)

        sx=c+int(rx/self.maze_resolution)
        sy=c-int(ry/self.maze_resolution)

        gx=c+int(self.maze_goal_x/self.maze_resolution)
        gy=c-int(self.maze_goal_y/self.maze_resolution)

        return (sy,sx),(gy,gx)

    def follow_maze_path(self,odom_x,odom_y,yaw,path):
        if path is None or len(path)<2:
            return 0.0,0.0

        target=path[1]
        c=self.maze_size//2

        tx=(target[1]-c)*self.maze_resolution
        ty=(c-target[0])*self.maze_resolution

        rx,ry,ryaw=self.get_relative_pose(odom_x,odom_y,yaw)

        target_yaw=math.atan2(ty-ry,tx-rx)
        error=math.atan2(math.sin(target_yaw-ryaw),math.cos(target_yaw-ryaw))

        angular=np.clip(error*1.5,-0.3,0.3)

        if abs(error)>0.45:
            return 0.0,angular

        return 0.08,angular

    def run(self,odom_x,odom_y,yaw):
        if self.maze_start_x is None:
            self.maze_start_x=odom_x
            self.maze_start_y=odom_y
            self.maze_start_yaw=yaw
            self.maze_map[:]=0
            return 0.0,0.0,False

        rx,ry,_=self.get_relative_pose(odom_x,odom_y,yaw)

        if math.hypot(self.maze_goal_x-rx,self.maze_goal_y-ry)<0.2:
            return 0.0,0.0,True

        self.update_maze_map(odom_x,odom_y,yaw)

        start,goal=self.get_maze_cells(odom_x,odom_y,yaw)

        if not (0<=start[0]<self.maze_size and 0<=start[1]<self.maze_size):
            return 0.0,0.0,False

        self.update_maze_value(start)

        path=self.make_maze_path(start,goal)

        if path is None:
            self.maze_path=[]
            return 0.0,0.0,False

        path=self.enhance_maze_path(path)
        self.maze_path=path

        linear,angular=self.follow_maze_path(odom_x,odom_y,yaw,path)

        return linear,angular,False

    def reset(self):
        self.maze_start_x=None
        self.maze_start_y=None
        self.maze_start_yaw=None
        self.maze_points=[]
        self.maze_path=[]
        self.maze_map[:]=0
        self.maze_value[:]=np.inf