# A* search without reopen the node i.e. use admissible and consistent heuristic

tree = {'S': [['A', 1], ['B', 5], ['C', 8]],
        'A': [['S', 1], ['D', 3], ['E', 7], ['G', 9]],
        'B': [['S', 5], ['G', 4]],
        'C': [['S', 8], ['G', 5]],
        'D': [['A', 3]],
        'E': [['A', 7]],
        'G': [['A', 9], ['B', 4], ['C', 5]]}

tree2 = {'S': [['A', 1], ['B', 2]],
         'A': [['S', 1]],
         'B': [['S', 2], ['C', 3]],
         #'B': [['S', 2], ['C', 3], ['D', 4]],
         'C': [['B', 2], ['E', 5], ['F', 6]],
         'D': [['G', 7]],
         #'D': [['B', 4], ['G', 7]],
         'E': [['C', 5]],
         'F': [['C', 6]],
         'G': [['D', 7]]
         }

tree3 = {'a': [['b', 4], ['c', 3]],
         'b': [['f', 5], ['e', 12]],
         'c': [['e', 10], ['d', 7]],
         'd': [['c', 7], ['e', 2]],
         'e': [['z', 5]],
         'f': [['z', 16]],
         'z': [['e', 6], ['f', 16]]
         }

heuristic = {'S': 8, 'A': 8, 'B': 4, 'C': 3, 'D': 5000, 'E': 5000, 'G': 0}
# heuristic = {'S': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'G': 0}
heuristic2 = {'S': 0, 'A': 5000, 'B': 2, 'C': 3, 'D': 4, 'E': 5000, 'F': 5000, 'G': 0}
heuristic3 = {'a': 14, 'b': 12, 'c': 11, 'd': 6, 'f': 11, 'e': 4, 'z': 0}

cost = {'a': 0}             # total cost for nodes visited


def AStarSearch(tree, heuristic, start_index, goal_index):
    # global tree, heuristic
    closed = []                # closed nodes
    # opened = [['a', 14]]     # opened nodes
    opened = [[start_index, heuristic[start_index]]]
    cost = {start_index: 0}

    '''find the visited nodes'''
    counter = 0
    while True:
        fn = [i[1] for i in opened]     # fn = f(n) = g(n) + h(n)
        chosen_index = fn.index(min(fn))  # can add a line to check, if open is empty, then no solution is found
        #print(chosen_index)
        node = opened[chosen_index][0]  # current node
        closed.append(opened[chosen_index])
        del opened[chosen_index]
        # if closed[-1][0] == 'z':        # break the while loop if node G has been found
        if closed[-1][0] == goal_index:
            break
        for item in tree[node]:
            if item[0] in [closed_item[0] for closed_item in closed]:  # to make sure to not to revist the node
                continue  # continue the next iteration of if loop
            cost.update({item[0]: cost[node] + item[1]})            # add nodes to cost dictionary (will update the cost to be the cost associated with the optimal node. (same key, value will be overwritten in dictionary)
            fn_node = cost[node] + heuristic[item[0]] + item[1]     # calculate f(n) of current node
            temp = [item[0], fn_node]
            opened.append(temp)                                     # store f(n) of current node in array opened
        #print(f'counter:{counter}')
        #print(f'fn:{fn}')
        #print(f'cost:{cost}')
        #print(f'temp:{temp}')
        #print(f'opened:{opened}')
        #print(f'closed:{closed}')
        counter = counter + 1

    '''find optimal sequence'''
    # trace_node = 'z'                        # correct optimal tracing node, initialize as node G
    # optimal_sequence = ['z']                # optimal node sequence
    trace_node = goal_index
    optimal_sequence = [goal_index]
    #print(f'length of closed:{len(closed)}')
    for i in range(len(closed)-2, -1, -1):  # start, stop, step (hence, start from len-2, back to 0)
        check_node = closed[i][0]           # current node
        #print(f'check_node:{check_node}')
        if trace_node in [children[0] for children in tree[check_node]]:
            children_costs = [temp[1] for temp in tree[check_node]]
            children_nodes = [temp[0] for temp in tree[check_node]]
            #print(f'children_nodes:{children_nodes}')

            '''check whether cost(before final node)+cost(to final node)=cost(total). If so, append current node to optimal sequence
            change the correct optimal tracing node to current node'''
            if cost[check_node] + children_costs[children_nodes.index(trace_node)] == cost[trace_node]:
                #print(cost[check_node], children_costs[children_nodes.index(trace_node)], cost[trace_node])
                optimal_sequence.append(check_node)
                #print(f'optimal_sequence_before_reverse:{optimal_sequence}')
                trace_node = check_node
    optimal_sequence.reverse()              # reverse the optimal sequence

    return closed, optimal_sequence


# start_index = 'S'
# goal_index = 'G'
# visited_nodes, optimal_nodes = AStarSearch(tree, heuristic, start_index, goal_index)
# print('visited nodes: ' + str(visited_nodes))
# print('optimal nodes sequence: ' + str(optimal_nodes))