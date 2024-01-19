# multiAgents.py
#
# Author: Thomas Sixuan Lou
# from CSE 473 at UW

# --------------
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


from sys  import maxint
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        # newGhostPos = successorGameState.getGhostPositions()
        newGhostStates = successorGameState.getGhostStates()

        # list of ghost states who are not scared
        newGhostPos = [ g.getPosition() for g in newGhostStates if g.scaredTimer == 0 ]

        # weighted list of ghost states who are scared
        newScaredGhostPos = [ (g.scaredTimer, g.getPosition()) for g in newGhostStates if g.scaredTimer != 0 ]

        # list of capsules
        newCapsules = successorGameState.getCapsules()

        "*** YOUR CODE HERE ***"
        newFoodPos = [ (x,y) for x in range(newFood.width) for y in range(newFood.height) if newFood[x][y]]

        capsuleDist = manhattanDistanceToAll(newPos, newCapsules)

        foodDist = manhattanDistanceToAll(newPos, newFoodPos)

        pGhostDist = sum([ util.manhattanDistance(newPos, xy) for xy in newGhostPos ])

        nGhostDist = sum([ w * util.manhattanDistance(newPos, xy) for (w,xy) in newScaredGhostPos ])

        delta = 0.000001

        ## parameters, parameters, parameters!!
        return 13 * (1.0/(capsuleDist + delta)) \
             + 9 * (1.0/(foodDist + delta))    \
             + 9 * (1.0/(nGhostDist + delta))  \
             - 0.5 * (1.0/(pGhostDist + delta)) \
             + successorGameState.getScore()


def manhattanDistanceToAllList(pos, targets):
    return [ util.manhattanDistance(pos, xy) for xy in targets ]

def manhattanDistanceToAll(pos, targets):
    return sum([ util.manhattanDistance(pos, xy) for xy in targets ])

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    # compute min/max values
    def maxValue(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = - maxint - 1
        for action in gameState.getLegalActions(0):
            v = max(v, self.minValue(gameState.generateSuccessor(0, action), 1, depth))
        return v

    ## recurse through all ghosts
    def minValue(self, gameState, agentIdx, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        v = maxint
        if agentIdx == gameState.getNumAgents() - 1:
            for action in gameState.getLegalActions(agentIdx):
                v = min(v, self.maxValue(gameState.generateSuccessor(agentIdx, action), depth - 1))
            return v
        else:
            for action in gameState.getLegalActions(agentIdx):
                v = min(v, self.minValue(gameState.generateSuccessor(agentIdx, action), agentIdx + 1, depth))
            return v

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        act = False
        v   = -maxint - 1
        for action in gameState.getLegalActions(0):
            tv = self.minValue(gameState.generateSuccessor(0, action), 1, self.depth)
            if tv > v:
                v = tv
                act = action

        # print act
        return act



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    # \alpha: MAX's best option on path to root
    # \beta : MIN's best option on path to root
    # compute min/max values
    def maxValue(self, gameState, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = - maxint - 1
        for action in gameState.getLegalActions(0):
            v = max(v, self.minValue(gameState.generateSuccessor(0, action), 1, depth, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha,v)
        return v

    ## recurse through all ghosts
    def minValue(self, gameState, agentIdx, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        v = maxint
        if agentIdx == gameState.getNumAgents() - 1:
            for action in gameState.getLegalActions(agentIdx):
                v = min(v, self.maxValue(gameState.generateSuccessor(agentIdx, action), depth - 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta,v)
            return v
        else:
            for action in gameState.getLegalActions(agentIdx):
                v = min(v, self.minValue(gameState.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta,v)
            return v

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        act = False
        alpha = -maxint - 1
        beta = maxint
        for action in gameState.getLegalActions(0):
            tv = self.minValue(gameState.generateSuccessor(0, action), 1, self.depth, alpha, beta)
            if tv > alpha:
                alpha = tv
                act = action

        return act

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maxValue(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = - maxint - 1
        for action in gameState.getLegalActions(0):
            v = max(v, self.expectiValue(gameState.generateSuccessor(0, action), 1, depth))
        return v

    ## recurse through all ghosts to compute expected value
    def expectiValue(self, gameState, agentIdx, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        v = 0.0
        actions = gameState.getLegalActions(agentIdx)
        pr = 1.0 / len(actions)

        if agentIdx == gameState.getNumAgents() - 1:
            for action in actions:
                v += pr * self.maxValue(gameState.generateSuccessor(agentIdx, action), depth - 1)
            return v
        else:
            for action in actions:
                v += pr * self.expectiValue(gameState.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
            return v

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        act = False
        v   = -maxint - 1
        for action in gameState.getLegalActions(0):
            tv = self.expectiValue(gameState.generateSuccessor(0, action), 1, self.depth)
            if tv > v:
                v = tv
                act = action

        return act

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
        features considered
      f1: total distance to all capsules left
      f2: total distance to all food pellets left
      f3: total distance to 10 nearest pellets
      f4: total distance to all scared ghosts
      f5: total distance to all non-scared ghosts
      f6: score of the state

      Notice, we want to maximize 1/f1, 1/f2, 1/f3, 1/f4 and f6,
      but to minimize 1/f5

      I came up with the following weights:
            (w1,...,w6) = (5, 1, 1, 1, -0.5, 1)
      and the evaluated value is simply a linear combinations of those
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    # list of ghost states who are not scared
    newGhostPos = [ g.getPosition() for g in newGhostStates if g.scaredTimer == 0 ]

    # weighted list of ghost states who are scared
    # each scared ghost's position is weighted by the scaredTimer of that ghost
    newScaredGhostPos = [ (g.scaredTimer, g.getPosition()) for g in newGhostStates if g.scaredTimer != 0 ]

    # list of capsules
    newCapsules = currentGameState.getCapsules()

    # positions of all food
    newFoodPos = [ (x,y) for x in range(newFood.width) for y in range(newFood.height) if newFood[x][y]]

    # total distance to all capsules left (f1)
    capsuleDist = manhattanDistanceToAll(newPos, newCapsules)

    # a list of distance to each of the food pellet
    foodDistList = manhattanDistanceToAllList(newPos, newFoodPos)
    foodDistList.sort()

    # total distance to all food left (f2)
    foodDist = sum(foodDistList)

    # total distance to 10 nearest food pellets (f3)
    nearFoodDist = sum(foodDistList[:10])

    # total distance to all scared ghosts (weighted by their scared time) (f4)
    nGhostDist = sum([ w * util.manhattanDistance(newPos, xy) for (w,xy) in newScaredGhostPos ])

    # total distance to all non-scared ghosts (f5)
    pGhostDist = sum([ util.manhattanDistance(newPos, xy) for xy in newGhostPos ])

    # to avoid division by zero
    delta = 0.01

    val = 5 * (1.0/(capsuleDist + delta)) \
          + (1.0/(foodDist + delta))    \
          + 2 * (1.0/(nearFoodDist + delta))   \
          + (1.0/(nGhostDist + delta))  \
          - 0.5 * (1.0/(pGhostDist + delta)) \
          + currentGameState.getScore()

    # give states with less capsules left a better score
    # this allows the pacman to get rewarded by eating capsules
    if len(newCapsules) < 2:
        return 2 * val
    else:
        return val

# Abbreviation
better = betterEvaluationFunction

