
def AStarSearch(tree, heuristic, start_index, goal_index):
    # global tree, heuristic
    closed = []                # closed nodes
    opened = [[start_index, heuristic[start_index]]]  # opened nodes
    cost = {start_index: 0}


    counter = 0
    while True:
        fn = [i[1] for i in opened]     # fn = f(n) = g(n) + h(n)
        chosen_index = fn.index(min(fn))  # can add a line to check, if open is empty, then no solution is found
        # print(chosen_index)
        node = opened[chosen_index][0]  # current node
        closed.append(opened[chosen_index])
        del opened[chosen_index]
        if closed[-1][0] == goal_index:
            break
        for item in tree[node]:
            if item[0] in [closed_item[0] for closed_item in closed]:  # to make sure to not to revist the node
                continue  # continue the next iteration of if loop
            cost.update({item[0]: cost[node] + item[1]})            # add nodes to cost dictionary (will update the cost to be the cost associated with the optimal node. (same key, value will be overwritten in dictionary))
            fn_node = cost[node] + heuristic[item[0]] + item[1]     # calculate f(n) of current node
            temp = [item[0], fn_node]
            opened.append(temp)                                     # store f(n) of current node in array opened
        # print(f'counter:{counter}')
        # print(f'fn:{fn}')
        # print(f'cost:{cost}')
        # print(f'temp:{temp}')
        # print(f'opened:{opened}')
        # print(f'closed:{closed}')
        counter = counter + 1



    trace_node = goal_index
    optimal_sequence = [goal_index]
    # print(f'length of closed:{len(closed)}')
    for i in range(len(closed)-2, -1, -1):  # start, stop, step (hence, start from len-2, back to 0)
        check_node = closed[i][0]           # current node
        # print(f'check_node:{check_node}')
        if trace_node in [children[0] for children in tree[check_node]]:
            children_costs = [temp[1] for temp in tree[check_node]]
            children_nodes = [temp[0] for temp in tree[check_node]]
            # print(f'children_nodes:{children_nodes}')


            if cost[check_node] + children_costs[children_nodes.index(trace_node)] == cost[trace_node]:
                # print(cost[check_node], children_costs[children_nodes.index(trace_node)], cost[trace_node])
                optimal_sequence.append(check_node)
                # print(f'optimal_sequence_before_reverse:{optimal_sequence}')
                trace_node = check_node
    optimal_sequence.reverse()              # reverse the optimal sequence

    return closed, optimal_sequence
