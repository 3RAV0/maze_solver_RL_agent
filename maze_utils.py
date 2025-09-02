from collections import deque

def bfs_shortest_length(walls, start=(0,0), goal=None):
    H,W = walls.shape[:2]
    goal = goal or (H-1,W-1)
    q = deque([(start[0], start[1], 0)])
    seen = {start}
    while q:
        y,x,d = q.popleft()
        if (y,x) == goal:
            return d
        for a,(dy,dx) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
            if walls[y,x,a] == 0:
                ny,nx = y+dy, x+dx
                if 0<=ny<H and 0<=nx<W and (ny,nx) not in seen:
                    seen.add((ny,nx))
                    q.append((ny,nx,d+1))
    return None  # ulaşılmazsa
