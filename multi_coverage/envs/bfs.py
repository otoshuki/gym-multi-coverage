from collections import deque
class point:
	def __init__(self,loc):
		self.c = loc[1]
		self.r = loc[0]

class queueNode:
	def __init__(self,point,dis,first):
		self.point = point
		self.dis = dis
		self.first = first

class vs:
	def __init__(self,dis,first):
		self.dis = dis
		self.first = first

def valid(r,c,R,C):
	return (r>=0) and (c>=0) and (c<C) and (r<R)

def bfs(mat,src):
	R = len(mat)
	C = len(mat[0])
	vis = []

	for r in range(R):
		x = []
		for c in range(C):
			x.append(vs(-1,'X'))
		vis.append(x)

	q = deque()
	q.append(queueNode(src,0,'O'))
	vis[src.r][src.c] = vs(0,'O')
	dr = [-1,1,0,0]
	dc = [0,0,-1,1]
	mv = ['D','U','R','L']

	while q:
		node = q.popleft()
		pt = node.point
		ds = node.dis
		ft = node.first

		for i in range(4):
			tr = pt.r + dr[i]
			tc = pt.c + dc[i]
			if(valid(tr,tc,R,C)):
				if(mat[tr][tc]==0 and vis[tr][tc].dis==-1):
					vis[tr][tc].dis=ds+1
					vis[tr][tc].first=mv[i]
					q.append(queueNode(point((tr,tc)),vis[tr][tc].dis,vis[tr][tc].first))
	return vis

def main():
	maze = [[ 1, 0, 1, 1, 1, 1, 0, 1, 1, 1 ],
		[ 1, 0, 1, 0, 1, 1, 1, 0, 1, 1 ],
		[ 1, 1, 1, 0, 1, 1, 0, 1, 0, 1 ],
		[ 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ],
		[ 1, 1, 1, 0, 1, 1, 1, 1, 1, 0 ],
		[ 1, 0, 1, 1, 1, 1, 0, 1, 0, 0 ],
		[ 1, 0, 0, 0, 0, 0, 0, 1, 0, 1 ],
		[ 1, 0, 1, 1, 1, 1, 0, 1, 1, 1 ],
		[ 1, 1, 0, 0, 0, 0, 1, 0, 0, 1 ]]
	src = point((0,0))
	dst = point((8,9))
	vis = bfs(maze,src,dst)
	# Uncomment to print
	for i in range(len(maze)):
		for j in range(len(maze[0])):
			print(vis[i][j].dis,end=" ")
		print()
	for i in range(len(maze)):
		for j in range(len(maze[0])):
			print(vis[i][j].first,end=" ")
		print()
	if(vis[dst.r][dst.c].dis==-1):
		print("Not Reachable")
	else:
		print("Destination is ",vis[dst.r][dst.c].dis,"and",end=" ")
		print("First Move is ",vis[dst.r][dst.c].first)

	return

if __name__ == "__main__":
	main()
