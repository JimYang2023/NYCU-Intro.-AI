import csv
from queue import PriorityQueue
edgeFile = 'edges.csv'

def create_graph(filename):
    graph = {}
    csv_read = csv.reader(open(filename,newline='',encoding='utf-8'))
    next(csv_read)
    for row in csv_read:
        node1 , node2 , weight = int(row[0]) , int(row[1]) , float(row[2])
        if node1 not in graph:
            graph[node1] = []
        graph[node1].append((node2 , weight)) 
    return graph

def find_path(parent , start , end):
    path = []
    temp_node = end
    while temp_node != start:
        path.append(temp_node)
        temp_node = parent[temp_node][0]
    path.append(start)    
    path.reverse()   
    return path , parent[end][1]


def ucs(start, end):
    # Begin your code (Part 3)
    graph = create_graph(edgeFile)
    path = []
    dist = 0.0
    visited = set()
    parent = {}

    q = PriorityQueue()
    q.put((0,start))
    visited.add(start)
    parent[start] = (None , 0)

    while(q.qsize() != 0):
        (cost , current_node) = q.get()
        
        if current_node == end:
            path , dist = find_path(parent , start , end)
            return path , round(dist,4) , len(visited)
        
        if current_node not in graph:
            continue
        
        for node , weight in graph[current_node]:
            total_cost = cost + weight
            if node in visited and total_cost >= parent[node][1]:
                continue
            visited.add(node)
            q.put((total_cost,node))
            parent[node] = (current_node , total_cost)

    return [] , 0 , len(visited)  
    raise NotImplementedError("To be implemented")
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
    print("")
