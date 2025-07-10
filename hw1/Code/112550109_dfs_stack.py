import csv
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
    dist = 0.0
    path = []
    temp_node = end
    while temp_node != start:
        path.append(temp_node)
        dist += parent[temp_node][1]
        temp_node = parent[temp_node][0]
    path.append(start)    
    path.reverse()   
    return path , dist

def dfs(start, end):
    # Begin your code (Part 2)
    graph = create_graph(edgeFile)
    stack = []
    visited = set()
    parent = {}

    stack.append(start)
    visited.add(start)
    
    while(len(stack)!=0):
        current_node = stack.pop()
        if current_node == end:
            break
        if current_node not in graph:
            continue
        for node , weight in graph[current_node]:
            if node in visited:
                continue
            stack.append(node)
            visited.add(node)
            parent[node] = (current_node , weight)

    path , dist = find_path(parent , start , end)
    count = len(visited)
    
    return path , round(dist,4) , count

    raise NotImplementedError("To be implemented")
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
    print("")
