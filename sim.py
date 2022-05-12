
import pygame
import math
from queue import PriorityQueue
import numpy as np
WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("ENES Final Project Simulation")
img = pygame.image.load('projectIcon.ico')
pygame.display.set_icon(img)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
init_barrier = [[0,7],[1,6],[2,5],[3,4],[4,3],[5,2],[6,1],[7,0],[49,7],[48,6],[47,5],[46,4],[45,3],[44,2],[43,1],[42,0]]
for i in range(50):
    init_barrier.append([0,i])
    if (i != 24):
        init_barrier.append([49,i])
init_barrier.append([24,49])
road_cord = []
init_barrier_flag = 0
class Spot:
    def __init__(self,row,col,width,total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows
    def is_road(self):
        if self in road_cord:
            return True
        else:
            return False
    def get_pos(self):
        return self.row, self.col
    def is_closed(self):
        return self.color == RED
    def is_open(self):
        return self.color == GREEN
    def is_barrier(self):
        return self.color == BLACK
    def is_start(self):
        return self.color == ORANGE
    def is_end(self):
        return self.color == TURQUOISE
    def reset(self):
        self.color = WHITE
    def make_start(self):
        self.color = ORANGE
    def make_closed(self):
        self.color = RED
    def make_open(self):
        self.color = GREEN
    def make_barrier(self):
        self.color = BLACK
    def make_end(self):
        self.color = TURQUOISE
    def make_path(self):
        self.color = PURPLE
    def make_road(self):
        self.color = GREY
    def draw(self,win):
        pygame.draw.rect(win,self.color,(self.x,self.y,self.width,self.width))
    def update_neighbors(self,grid):
        self.neighbors = {}
        if self.row < self.total_rows -1 and not grid[self.row + 1][self.col].is_barrier(): # down actually right
            self.neighbors.update({"right":grid[self.row + 1][self.col]})
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # up actually left 
            self.neighbors.update({"left":(grid[self.row - 1][self.col])})
        if self.col < self.total_rows -1 and not grid[self.row][self.col + 1].is_barrier():    # right actually down
            self.neighbors.update({"down":grid[self.row][self.col + 1]})
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # left actually up
            self.neighbors.update({"up":grid[self.row][self.col - 1]})
        if (self.col > 0 and self.row >0)  and not grid[self.row-1][self.col - 1].is_barrier(): # up left
            self.neighbors.update({"up_left":grid[self.row-1][self.col - 1]})
        if (self.row < self.total_rows-1 and self.col >0)  and not grid[self.row+1][self.col - 1].is_barrier(): # down left
            self.neighbors.update({"down_left":grid[self.row+1][self.col - 1]})
        if (self.col < self.total_rows-1 and self.row >0)  and not grid[self.row-1][self.col + 1].is_barrier(): # up right
            self.neighbors.update({"up_right":grid[self.row-1][self.col + 1]})
        if (self.col < self.total_rows-1 and self.row <self.total_rows-1)  and not grid[self.row+1][self.col + 1].is_barrier(): # down right
            self.neighbors.update({"down_right":grid[self.row+1][self.col + 1]})
    def update_og_neighbors(self,grid):
        self.og_neighbors = []
        if self.row < self.total_rows -1 and not grid[self.row + 1][self.col].is_barrier(): # down actually right
            self.og_neighbors.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # up actually left
            self.og_neighbors.append(grid[self.row - 1][self.col])
        if self.col < self.total_rows -1 and not grid[self.row][self.col + 1].is_barrier():    # right actually down
            self.og_neighbors.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # left actually up
            self.og_neighbors.append(grid[self.row][self.col - 1])
        if (self.col > 0 and self.row >0)  and not grid[self.row-1][self.col - 1].is_barrier(): # up left
            self.og_neighbors.append(grid[self.row-1][self.col - 1])
        if (self.row < self.total_rows-1 and self.col >0)  and not grid[self.row+1][self.col - 1].is_barrier(): # down left
            self.og_neighbors.append(grid[self.row+1][self.col - 1])
        if (self.col < self.total_rows-1 and self.row >0)  and not grid[self.row-1][self.col + 1].is_barrier(): # up right
            self.og_neighbors.append(grid[self.row-1][self.col + 1])
        if (self.col < self.total_rows-1 and self.row <self.total_rows-1)  and not grid[self.row+1][self.col + 1].is_barrier(): # down right
            self.og_neighbors.append(grid[self.row+1][self.col + 1])
    def update_road_neighbors(self,grid):
        self.road_neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier() and grid[self.row + 1][self.col].is_road():
            self.road_neighbors.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier() and grid[self.row - 1][self.col].is_road():
            self.road_neighbors.append(grid[self.row - 1][self.col])
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier() and grid[self.row][self.col + 1].is_road():
            self.road_neighbors.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier() and grid[self.row][self.col - 1].is_road():
            self.road_neighbors.append(grid[self.row][self.col - 1])
        if (self.col > 0 and self.row >0)  and not grid[self.row-1][self.col - 1].is_barrier() and grid[self.row-1][self.col - 1].is_road():
            self.road_neighbors.append(grid[self.row-1][self.col - 1])
        if (self.row < self.total_rows-1 and self.col >0)  and not grid[self.row+1][self.col - 1].is_barrier() and grid[self.row+1][self.col - 1].is_road():
            self.road_neighbors.append(grid[self.row+1][self.col - 1])
        if (self.col < self.total_rows-1 and self.row >0)  and not grid[self.row-1][self.col + 1].is_barrier() and grid[self.row-1][self.col + 1].is_road():
            self.road_neighbors.append(grid[self.row-1][self.col + 1])
        if (self.col < self.total_rows-1 and self.row <self.total_rows-1)  and not grid[self.row+1][self.col + 1].is_barrier() and grid[self.row+1][self.col + 1].is_road():
            self.road_neighbors.append(grid[self.row+1][self.col + 1])
    def __lt__(self, other):
        return False

def h(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return abs(x1-x2) + abs(y1-y2)

def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()
def find_closest_road_spot(self,grid):
    closest_g_score = float("inf")
    closest_road = None
    for row in grid:
        for spot in row:
            if spot.is_road():
                spot_g_score = h(self.get_pos(),spot.get_pos())
                if spot_g_score < closest_g_score:
                    closest_g_score = spot_g_score
                    closest_road = spot
    return {"spot":closest_road,"g_score":closest_g_score}


def ownalgo(draw,grid,start,end):
    #road_start = None
    #if len(start.road_neighbors) == 0:
        #road_start = find_closest_road_spot(start,grid)
    count = 0
    #start = road_start["spot"]
    open_set = PriorityQueue()
    open_set.put((0,count,start))
    index = 0
    came_from = {}
    open_set_hash = {start}
    #move forward until we find a road or until we reach a barrier
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current = open_set.get()[2]
        open_set_hash.remove(current)
        #current.make_open()
        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True
        breakVal = False
        if len(current.neighbors) < 7:
            breakVal = True
            if current.is_road():
                breakVal = False
                break
            if "up" not in current.neighbors:
                if current.neighbors["down"] not in open_set_hash:
                    index += 1
                    count +=1
                    open_set.put((index,count,current.neighbors["down"]))
                    open_set_hash.add(current.neighbors["down"])
                    current.neighbors["down"].make_open()
            elif "down" not in current.neighbors:
                if current.neighbors["up"] not in open_set_hash:
                    index += 1
                    count +=1
                    open_set.put((index,count,current.neighbors["up"]))
                    open_set_hash.add(current.neighbors["up"])
                    current.neighbors["up"].make_open()
            elif "left" not in current.neighbors:
                if current.neighbors["right"] not in open_set_hash:
                    index += 1
                    count +=1
                    open_set.put((index,count,current.neighbors["right"]))
                    open_set_hash.add(current.neighbors["right"])
                    current.neighbors["right"].make_open()
            elif "right" not in current.neighbors:
                if current.neighbors["left"] not in open_set_hash:
                    index += 1
                    count +=1
                    open_set.put((index,count,current.neighbors["left"]))
                    open_set_hash.add(current.neighbors["left"])
                    current.neighbors["left"].make_open()
        if (breakVal != True):
            intersection_check = False
            left_check = False
            right_check = False
            up_check = False
            down_check = False
            if (current.neighbors["up"].is_road() and current.neighbors['down'].is_road() and current.neighbors['left'].is_road() and current.neighbors['right'].is_road()):
                intersection_check = True

            

            if not current.neighbors['left'].is_barrier() and current.neighbors['left'].is_road() and (not current.neighbors['up_left'].is_barrier() and current.neighbors['up_left'].is_road()) and (not current.neighbors['down_left'].is_barrier() and current.neighbors['down_left'].is_road()):
                left_check = True
            if not current.neighbors['right'].is_barrier() and current.neighbors['right'].is_road() and (not current.neighbors['up_right'].is_barrier() and current.neighbors['up_right'].is_road()) and (not current.neighbors['down_right'].is_barrier() and current.neighbors['down_right'].is_road()):
                right_check = True
            if (not current.neighbors['up'].is_barrier()) and (current.neighbors['up'].is_road()):
                up_check = True
            if (not current.neighbors['down'].is_barrier())and (current.neighbors['down'].is_road()):
                down_check = True
            if intersection_check:
                x,y = current.get_pos()
                if (x == 24 and y == 24):
                    index+=1 
                    count+= 1
                    if current.neighbors['right'] not in open_set_hash:
                        open_set.put((index,count,current.neighbors['right']))
                        open_set_hash.add(current.neighbors['right'])
                elif (x == 42 and y == 24):
                    index+=1
                    count+= 1
                    if current.neighbors['right'] not in open_set_hash:
                        open_set.put((index,count,current.neighbors['right']))
                        open_set_hash.add(current.neighbors['right'])
                elif (x == 24 and y == 6):
                    if current.neighbors['right'] not in open_set_hash and current.neighbors['right'].is_road():
                        index+=1
                        count+= 1
                        open_set.put((index,count,current.neighbors['right']))
                        open_set_hash.add(current.neighbors['right'])
                    if current.neighbors['down_right'] not in open_set_hash and current.neighbors['down_right'].is_road():
                        index+=1
                        count+= 1
                        open_set.put((index,count,current.neighbors['right_down']))
                        open_set_hash.add(current.neighbors['right_down'])
                    if current.neighbors['up_right'] not in open_set_hash and current.neighbors['up_right'].is_road():
                        index+=1
                        count+= 1
                        open_set.put((index,count,current.neighbors['right_up']))
                        open_set_hash.add(current.neighbors['right_up'])
                elif (x == 24 and y == 42):
                    if current.neighbors['right'] not in open_set_hash and current.neighbors['right'].is_road():
                        index+=1
                        count+= 1
                        open_set.put((index,count,current.neighbors['right']))
                        open_set_hash.add(current.neighbors['right'])
                    if current.neighbors['down_right'] not in open_set_hash and current.neighbors['down_right'].is_road():
                        index+=1
                        count+= 1
                        open_set.put((index,count,current.neighbors['right_down']))
                        open_set_hash.add(current.neighbors['right_down'])
                    if current.neighbors['up_right'] not in open_set_hash and current.neighbors['up_right'].is_road():
                        index+=1
                        count+= 1
                        open_set.put((index,count,current.neighbors['right_up']))
                        open_set_hash.add(current.neighbors['right_up'])
            else:
                if left_check:
                    index += 1
                    count += 1
                    if current.neighbors['left'].is_road():
                        if current.neighbors['left'] not in open_set_hash:
                            open_set.put((index,count,current.neighbors['left']))
                            open_set_hash.add(current.neighbors['left'])
                            current.neighbors['left'].make_open()
                    if current.neighbors['up_left'].is_road():
                        if current.neighbors['up_left'] not in open_set_hash:
                            index +=1
                            count +=1
                            open_set.put((index,count,current.neighbors['up_left']))
                            open_set_hash.add(current.neighbors['up_left'])
                            current.neighbors['up_left'].make_open()
                    if current.neighbors['down_left'].is_road():
                        if current.neighbors['down_left'] not in open_set_hash:
                            index +=1
                            count +=1
                            open_set.put((index,count,current.neighbors['down_left']))
                            open_set_hash.add(current.neighbors['down_left'])
                            current.neighbors['down_left'].make_open()
                if right_check:
                    if current.neighbors['right'].is_road():
                        if current.neighbors['right'] not in open_set_hash:
                            index += 1
                            count += 1
                            open_set.put((index,count,current.neighbors['right']))
                            open_set_hash.add(current.neighbors['right'])
                            current.neighbors['right'].make_open()
                    if current.neighbors['up_right'].is_road():
                        if current.neighbors['up_right'] not in open_set_hash:
                            index +=1
                            count +=1
                            open_set.put((index,count,current.neighbors['up_right']))
                            open_set_hash.add(current.neighbors['up_right'])
                            current.neighbors['up_right'].make_open()
                    if current.neighbors['down_right'].is_road():
                        if current.neighbors['down_right'] not in open_set_hash:
                            index +=1
                            count +=1
                            open_set.put((index,count,current.neighbors['down_right']))
                            open_set_hash.add(current.neighbors['down_right'])
                            current.neighbors['down_right'].make_open()
                if down_check:
                    if current.neighbors['down'].is_road():
                        if current.neighbors['down'] not in open_set_hash:
                            index += 1
                            count += 1
                            open_set.put((index,count,current.neighbors['down']))
                            open_set_hash.add(current.neighbors['down'])
                            current.neighbors['down'].make_open()
                if up_check:
                    if current.neighbors['up'].is_road():
                        if current.neighbors['up'] not in open_set_hash:
                            index += 1
                            count += 1
                            open_set.put((index,count,current.neighbors['up']))
                            open_set_hash.add(current.neighbors['up'])
                            current.neighbors['up'].make_open()
                if ((not left_check) and (not right_check) and (not up_check)):
                    index+=1
                    count+=1
                    if current.neighbors['right'] not in open_set_hash:
                        open_set.put((index,count,current.neighbors['right']))
                        open_set_hash.add(current.neighbors['right'])
                        current.neighbors['right'].make_open()
        #current.make_closed()
        draw()
        if current != start:
            current.make_closed()




        

def algo_closest_road(draw,grid,start,end):
    closest_road_count = 0
    open_set = PriorityQueue()
    open_set.put((0,closest_road_count,start))
    came_from = {}
    g_score = {spot:float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot:float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(),end.get_pos())
    open_set_hash = {start}
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current = open_set.get()[2]
        open_set_hash.remove(current)
        if current == end:
            #reconstruct_path(came_from,end,draw)
            return True
        for neighbor in current.og_neighbors:
            temp_g_score = g_score[current] + 1
            if (temp_g_score < g_score[neighbor]):
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(),end.get_pos())
                if neighbor not in open_set_hash:
                    closest_road_count += 1
                    open_set.put((f_score[neighbor],closest_road_count,neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()
    
            
def ogalgorithm(draw,grid,start,end):
    road_start = None
    if len(start.road_neighbors) == 0 :
        road_start = find_closest_road_spot(start,grid)
    count = 0
    if (road_start != None):
        algo_closest_road(draw,grid,start,road_start["spot"])
        start = road_start["spot"]
    open_set = PriorityQueue()
    open_set.put((0,count,start))
    came_from = {}
    g_score = {spot:float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot:float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(),end.get_pos())
    open_set_hash = {start}
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current = open_set.get()[2]
        open_set_hash.remove(current)
        if current == end:
            #reconstruct_path(came_from,end,draw)
            end.make_end()
            return True
        for neighbor in current.og_neighbors:
            temp_g_score = g_score[current] + 1
            if (temp_g_score < g_score[neighbor] and neighbor.is_road()) or neighbor == end:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(),end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor],count,neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()
def make_grid(rows,width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i,j,gap,rows)
            grid[i].append(spot)
    return grid
def draw_grid(win,rows,width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win,GREY,(0,i*gap),(width,i*gap))
        for j in range(rows):
            pygame.draw.line(win,GREY,(j*gap,0),(j*gap,width))
def draw(win,grid,rows,width,init_barrier,reset):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win,rows,width)
    lower_Rightbarrier = [[42,49],[43,48],[44,47],[45,46],[46,45],[47,44],[48,43],[49,42]]
    if (init_barrier_flag < 1) or reset:
        for row in init_barrier:
            spot = grid[row[0]][row[1]]
            spot.make_barrier()
            reverse_spot = grid[row[1]][row[0]]
            reverse_spot.make_barrier()
        for row in lower_Rightbarrier:
            spot = grid[row[0]][row[1]]
            spot.make_barrier()
        for i in range(1,49):
            spot = grid[24][i]
            road_cord.append(spot)
            spot.make_road()
        for i in range(1,49):
            spot = grid[i][24]
            road_cord.append(spot)
            spot.make_road()
        t = np.linspace(0, 2*np.pi, 100)
        R = 18
        x = R*np.cos(t)
        y = R*np.sin(t)
        for i in range(len(x)):
            x[i] += 24
            y[i] += 24
        for i in range(len(x)):
            spot = grid[round(x[i])][round(y[i])]
            road_cord.append(spot)
            spot.make_road()
        spot = grid[22][6]
        road_cord.append(spot)
        spot.make_road()
        spot = grid[22][42]
        road_cord.append(spot)
        spot.make_road()
        spot = grid[32][40]
        road_cord.append(spot)
        spot.make_road()
        spot = grid[32][8]
        road_cord.append(spot)
        spot.make_road()
        spot = grid[42][28]
        road_cord.append(spot)
        spot.make_road()
        spot = grid[42][20]
        road_cord.append(spot)
        spot.make_road()
    pygame.display.update()

def get_clicked_pos(pos,rows,width):
    gap = width // rows
    y,x = pos
    row = y // gap
    col = x // gap
    return row,col
def main(win,width):
    global init_barrier_flag
    ROWS = 50
    grid = make_grid(ROWS,width)
    start = None
    end = grid[49][24]
    end.make_end()
    run = True
    while run:
        draw(win,grid,ROWS,width,init_barrier,False)
        init_barrier_flag += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if pygame.mouse.get_pressed()[0]: #left click
                pos = pygame.mouse.get_pos()
                row,col = get_clicked_pos(pos,ROWS,width)
                spot = grid[row][col]
                if not start and spot != end:
                    start = spot
                    start.make_start()
                elif not end and spot != start:
                    end = spot
                    end.make_end()
                elif spot != end and spot != start:
                    spot.make_road()
            elif pygame.mouse.get_pressed()[2]: #right click
                pos = pygame.mouse.get_pos()
                row,col = get_clicked_pos(pos,ROWS,width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                            spot.update_road_neighbors(grid)
                            spot.update_og_neighbors(grid)
                    #ogalgorithm(lambda: draw(win,grid,ROWS,width,init_barrier,False),grid,start,end)
                    ownalgo(lambda: draw(win,grid,ROWS,width,init_barrier,False),grid,start,end)
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS,width)
                    draw(win,grid,ROWS,width,init_barrier,True)
    pygame.quit()

main(WIN,WIDTH)