# searchAgents.py
# Assignment 1: 
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import pacman

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions(): #checks if moving west is valid
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # fn: the search function to use, prob: the type of search problem, heuristic: the heuristic function to use (for unformed search)
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames: # if the function does not have a heuristic parameter
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func # set the search function to the function
        else: # if the function does have a heuristic parameter
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur) # set the search function to a lambda function that takes a search problem and returns the result of the search function with the search problem and heuristic as parameters

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)



    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Creates an instance of the search problem
        self.actions  = self.searchFunction(problem) # Runs the search function (e.g., DFS) on the search problem to get a path
        if self.actions == None:
            self.actions = []
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        # if the problem tracks expanded nodes, print the number of nodes expanded
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0 
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):   # If there are still actions left
            return self.actions[i]  # Return the next action
        else:
            return Directions.STOP # Return STOP if no more actions

class PositionSearchProblem(search.SearchProblem):
    # this is the default search problem ('prob') for the SearchAgent above
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  
    This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py) | pacman map
        costFn: A function from a search state (tuple) to a non-negative number | compute movement cost
        goal: A position in the gameState | goal position
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState # returns starting position

    def isGoalState(self, state):
        isGoal = state == self.goal # simply checks if 'state' is equal to 'goal' and stores bool in 'isGoal'

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        # basically, checks all four directions (all possible moves in pacman)
        # and returns the ones that are valid (does not go into walls)
        successors = [] # initialize the list of successors
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]: # iterate over the possible actions
            x,y = state # get the current position
            dx, dy = Actions.directionToVector(action) # get the direction vector for the action
            nextx, nexty = int(x + dx), int(y + dy) # get the next position
            if not self.walls[nextx][nexty]:    # if the next position is not a wall
                nextState = (nextx, nexty)      # set the next state
                cost = self.costFn(nextState)   # get the cost of the next state
                successors.append( ( nextState, action, cost) ) # add the next state to the list of successors

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited: # if the state has not been visited
            self._visited[state] = True # mark the state as visited
            self._visitedlist.append(state) # add the state to the list of visited states

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        # basically, computes total cost of a path
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy) # same as in getSuccessors, except we are updating x, y instead of using nextx, nexty
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is (1/2)^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0] # (1/2)^x makes the cost of moving to the left increase exponentially as x decreases (x more negative = more cost)
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0] # 2^x makes the cost of moving to the right increase exponentially as x increases (x more positive = more cost)
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1]) # Sum of absolute differences in x and y coordinates

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5 # Direct straight-line distance

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState: pacman.GameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls() # this is same as PositionSearchProblem 
            # but name is 'startingGameState' instead of 'gameState' 
            # and 'startingPosition' instead of 'startState'
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top)) # this assumes the maze is rect and corners are legal positions (not wall or cut off by walls)
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        self.startingGameState = startingGameState

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state space)
        """ 
        
        "*** YOUR CODE HERE *** (Q5)"
        """ A state space can be the start coordinates and a list to hold visited corners"""
        state = (self.startingPosition, ()) # since this is start state, no corners have been visited, so the list is empty
        return state

    def isGoalState(self, state: Any):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE *** (Q5)"
        
        """ Check to see if a state is a corner, and if so are the other corners visited """
        
        # visited_corner_count = 0
        # for corner in self.corners:
        #     if state[0] == corner:
        #         for visited_corner in state[1]:
        #             if visited_corner == corner:
        #                 visited_corner_count = visited_corner_count + 1
        # return visited_corner_count == 4

        # simply check if there are 4 corners in state's visited_corners tuple
        # naive, but should work fine if we update the state's visited_corners properly (in getSuccessors)
        return len(state[1]) == 4

    def getSuccessors(self, state: Any):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE *** (Q5)"
            # determine the next state (position) and update the list of visited corners
            x, y = state[0] # currentPosition
            dx, dy = Actions.directionToVector(action) 
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]
            if not hitsWall:
                # if the next position is not a wall, create a new state
                nextState = (nextx, nexty)
                cost = 1
                visitedCorners = tuple(state[1][:]) # copy the list of visited corners
                # if this next position is a corner and it has not been visited
                # add it to the list of visited corners
                if nextState in self.corners and nextState not in visitedCorners:
                    visitedCorners = visitedCorners + (nextState,)  # 'add' to the tuple, this is so set() works
                
                # add the new state to the list of successors
                successors.append(( (nextState, visitedCorners), action, cost))

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999 # if the next position is a wall, return a large cost
        return len(actions) # return the length of the path (number of actions taken)


def cornersHeuristic(state: Any, problem: CornersProblem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE *** (Q6)"

    # return 0 # testing with no heuristic - should behave same as UCS (and same as Q5)

    # general idea
    # for each new path (newState), want to calculate for the heuristic cost
    
    # want to calculate manhattan distance from this state to closest, unvisited corner
    # then calculate distance from that corner to next closest, unvisited, etc.
    # return sum of this distance

    # get the state and visited corners
    position, visited_corners = state

    # get all unvisited corners
    # unvisited_corners = []
    # # loop through all corners
    # for corner in corners:
    #     if corner not in visited_corners:
    #         unvisited_corners.append(corner)
    unvisited_corners = [corner for corner in corners if corner not in visited_corners]


    # if there are no unvisited corners, we've reached goal state, return 0
    if len(unvisited_corners) == 0:
        return 0
    
    # get the cost from current position to all unvisited corners
    # done by summing the (manhattan) distance from current position to each unvisited corner
    # and changing the position to be the new closest corner

    current_position = position
    total_distance = 0
    
    # while there are still unvisited corners
    while len(unvisited_corners) > 0:
        unvisited_corners_distances = []

        # go throguh the remaining unvisited corners
        for corner in unvisited_corners:
            x, y = corner
            # get the remaining unvisited corner distances 
            unvisited_corners_distances.append([abs(current_position[0] - x) + abs(current_position[1] - y), corner])
        
        # get the lowest distance
        closest_corner = unvisited_corners[0]
        closest_distance = unvisited_corners_distances[0][0]
        current_position = closest_corner
        # loop through the distances
        for distance in unvisited_corners_distances:
            # print(distance)
            # if the distance is closer than current
            if distance[0] < closest_distance:
                closest_distance = distance[0]  # update the closest distance
                closest_corner = distance[1]    # update the cloeset corner | recall: distance = [number, (x,y)], is kind of a 'bandage' solution, but it's simple...
                current_position = closest_corner # update the current position to now use the closest corner (since this will be where we take pacman next)

        total_distance = total_distance + closest_distance
        unvisited_corners.remove(closest_corner)

    return total_distance


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState: pacman.GameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))


def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
