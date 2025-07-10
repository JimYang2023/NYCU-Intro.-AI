import csv , queue
edgeFile = 'edges.csv'
heuristicFile = 'heuristic_values.csv'

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

def create_heuristic(filename):
    heuristic = {}
    csv_read = csv.reader(open(filename,newline='',encoding='utf-8'))
    m = {}
    
    first_line = next(csv_read)
    idx = 0
    for str in first_line:
        if idx != 0:
            m[int(str)] = idx
        idx += 1
        
    for row in csv_read:
        idx = 0
        node = 0
        for str in row:
            if idx == 0:
                node = int(str) 
                heuristic[node] = []
                idx += 1
            else:
                heuristic[node].append(float(str))

    return (heuristic , m)

def find_path(parent , start , end):
    path = []
    temp_node = end 
    while temp_node != start:
        path.append(temp_node)
        temp_node = parent[temp_node][0]
    path.append(start)
    path.reverse()
    return path , parent[end][1]

def astar(start, end):
    # Begin your code (Part 4)
    graph = create_graph(edgeFile)
    heuristic , m = create_heuristic(heuristicFile)
    openList = queue.PriorityQueue()
    visited = set()
    closed = set()
    target = 0
    parent = {}
    predict = {}
    
    # change target
    if start not in m:
        target = 0
    else:
        target = m.get(start)

    openList.put((0,0,start))
    visited.add(start)
    predict[start] = 0.0

    while not openList.empty():
        predict_cost , real_cost , current_node = openList.get()
        
        if current_node in closed or current_node not in graph:
            continue
        
        for node , weight in graph[current_node]:
            cost = real_cost + weight
            pred_cost = cost + heuristic[node][target]

            if node in closed or (node in visited and pred_cost >= predict[node]):
                continue

            visited.add(node)
            parent[node] = (current_node , cost)
            predict[node] = pred_cost
            openList.put((pred_cost,cost,node))

            if node == end:
                path , dist = find_path(parent , start , end)
                return path , round(dist , 4) , len(visited) 
            
        closed.add(current_node)

    return [] , 0.0 , len(visited)
    raise NotImplementedError("To be implemented")
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(426882161, 1737223506)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
    print("")
