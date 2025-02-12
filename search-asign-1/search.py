# search.py
# Assignment 1:
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE *** (Q1)" #https://learn.ontariotechu.ca/courses/31258/files/4912386?module_item_id=712851

    # need to use a stack (LIFO) for DFS
    # expand the top node and add its children to the top of the stack 
    # e.g.,     1
    #         /   \
    #        2     3
    #       / \   / \
    #      4   5 6   7
    #               [7]
    #        [3]    [6]
    # [1] -> [2] -> [2]
    # (this example does right first)

    '''
    graph search pseudcode
    function graph_search(problem, fringe) return a solutioin, or failure
        closed <- an empty set
        fringe <- INSERT(MAKE-NODE(INITIAL-STATE[problem]), fringe)
        loop do
            if fringe is empty then return failure
            node <- REMOVE-FRONT(fringe)
            if GOAL_TEST(problem, STATE[node]) then return node
            if STATE[node] is not in closed then
                add STATE[node] to closed
                for child-node in EXPAND(STATE[node], problem) do
                    fringe <- INSERT(child-node, fringe)
                end
        end
    '''

    # initialize the stack
    stack = util.Stack()
    # initialize the closed set
    closed = set()  # set is a data structure that stores unique elements
                    # same as list but no duplicates (?)
    # push the start state to the stack
    stack.push((problem.getStartState(), []))

    # loop while the stack is not empty
    while not stack.isEmpty():
        # pop the top node from the stack and store it
        # node is a tuple of (state, actions) - recall above, we push (start state, []) to the stack
        # state is the current state and actions is the list of actions to reach this state (i.e., path)
        node, actions = stack.pop() 
        # print("Node:", node)
        # print("Actions:", actions)

        # check if the current state is the goal state
        if problem.isGoalState(node):
            return actions  # if goal, return the list of actions to reach this state
        
        # otherwise...
        # check if the current state is in the closed set
        if node not in closed:
            # add the current state to the closed set
            closed.add(node)
            # get the successors of the current state
            # note: .getSuccessors() checks W, E, S, N in that order
            successors = problem.getSuccessors(node)

            # push the successors to the stack by looping through them
            for successor in successors: 
                # successor is a tuple of (state, actions, cost)
                nextState = successor[0]
                action = successor[1]
                # cost = successor[2]
                # state is the successor state, actions is the list of actions to reach this state, and cost is the cost to reach this state
                # we don't worry about cost; cost is not used in DFS
                # only push state and actions to the stack
                stack.push((nextState, actions + [action]))
    
    return actions # if no solution, return the list of actions to reach this state?

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    
    "*** YOUR CODE HERE *** (Q2)"

    # need to use a queue (FIFO) for BFS
    # basically, expand current first node and add its children to the end of the queue
    # e.g.,     1
    #         /   \
    #        2     3
    #       / \   / \
    #      4   5 6   7
    # [1] -> [2, 3] -> [3, 4, 5] ...
    
    # just DFS but with queue instead of stack

    # initialize the queue
    queue = util.Queue()
    #initialize the closed set
    closed = set()

    # push the start state (root node) to the queue
    queue.push((problem.getStartState(), []))

    while not queue.isEmpty():
        # pop the front node from the queue and store it
        node, actions = queue.pop()

        # check if the current node's state is the goal state
        if problem.isGoalState(node):
            return actions
        
        # check if current node has been visited
        if not node in closed:
            # add the current node
            closed.add(node)
            # get the successors of the current node
            successors = problem.getSuccessors(node)

            # push the successors into the queue
            for successor in successors:
                nextState = successor[0]
                action = successor[1]
                # cost = successor[2]
                queue.push((nextState, actions + [action]))

    return actions

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    
    "*** YOUR CODE HERE *** (Q3)"
    
    # need to use a priority queue (lowest cost node goes first) for UCS
    # basically, expand the node with the lowest cost and add its children to the queue (queue sorts newly added nodes)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    "*** YOUR CODE HERE *** (Q4)"

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
