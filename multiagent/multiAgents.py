# multiAgents.py
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newPos = successorGameState.getPacmanPosition() #Tuple
        newFood = successorGameState.getFood() # 2D Array of bool vals
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()
        food_distance = float("inf")
        ghost_distance = float("inf")
        closest_ghost_idx = 0

        for f in newFood.asList():
            if manhattanDistance(newPos, f) < food_distance:
                food_distance = manhattanDistance(newPos, f)
        
        if food_distance != float("inf") and food_distance > 0:
            score +=  10 * (1/food_distance) # Reward for closest proximity to food

        for g in newGhostStates:
            gPos = g.getPosition()
            if manhattanDistance(newPos, gPos) < ghost_distance:
                ghost_distance = manhattanDistance(newPos, gPos)
                closest_ghost_idx = newGhostStates.index(g)
        
        if ghost_distance != float("inf") and ghost_distance > 0:
            score -= 30 * (1/ghost_distance) # Punish for closer proximity to ghost

        if newScaredTimes[closest_ghost_idx] > 0:
            score += 10 * (1 / ghost_distance)

        return score

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Pacman Max Node
        # PGhost Min Nodes

        def minimax(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0: # Pacman's Turn
                pVals = []
                for action in state.getLegalActions(0):
                    successor = state.generateSuccessor(0, action)
                    pVals.append(minimax(successor, depth, 1))   
                return max(pVals)

            else: # Ghost's Turn
                gVals = []
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)

                    if agentIndex == state.getNumAgents() - 1: # Checks for last ghost
                        gVals.append(minimax(successor, depth+1, 0))
                    else:
                        gVals.append(minimax(successor, depth, agentIndex+1))
                return min(gVals)
        
        
        bestVal = float("-inf")
        bestAction = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            val = minimax(successor, 0, 1)
            if val > bestVal:
                bestVal = val
                bestAction = action
        return bestAction



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0: # Pacman's Turn
                pVal = float("-inf")
                for action in state.getLegalActions(0):
                    successor = state.generateSuccessor(0, action)
                    pVal = max(pVal, alphaBeta(successor, depth, 1, alpha, beta))

                    if pVal > beta:
                        return pVal
                    
                    alpha = max(pVal, alpha)
                return pVal

            else: # Ghost's Turn
                gVal = float("inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)

                    if agentIndex == state.getNumAgents() - 1: # Checks for last ghost
                        gVal = min(gVal, alphaBeta(successor, depth+1, 0, alpha, beta))
                    else:
                        gVal = min(gVal, alphaBeta(successor, depth, agentIndex+1, alpha, beta))

                    if gVal < alpha:
                        return gVal
                    
                    beta = min(beta, gVal)
                return gVal

        bestVal = float("-inf")
        bestAction = None
        alpha = float("-inf")
        beta = float("inf")

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            val = alphaBeta(successor, 0, 1, alpha, beta)
            if val > bestVal:
                bestVal = val
                bestAction = action
            alpha = max(alpha, bestVal)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0: # Pacman's Turn
                pVal = float("-inf")
                for action in state.getLegalActions(0):
                    successor = state.generateSuccessor(0, action)
                    pVal = max(pVal, expectimax(successor, depth, 1))
                return pVal

            else: # Ghost's Turn
                gVal = 0

                if len(state.getLegalActions(agentIndex)) == 0: # Prevents Zero Division Error
                    return 0

                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    
                    if agentIndex == state.getNumAgents() - 1: # Checks for last ghost
                        gVal += expectimax(successor, depth+1, 0)
                    else:
                        gVal += expectimax(successor, depth, agentIndex+1)
                    
                return gVal / len(state.getLegalActions(agentIndex))
        
        bestVal = float("-inf")
        bestAction = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            val = expectimax(successor, 0, 1)
            if val > bestVal:
                bestVal = val
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: My implementation of the betterEvaluationFunction() function
    was directly based on my previous implementation of evaluationFunction(). My
    thought process for my implmenation was to use manhattan distance to calculate 
    the score value I would return. My calculation is as follows:
    score = foodScore + ghostScore + capsuleScore. 
    
    Each score value was assigned a weight to determine how "good" a move is with 
    proximity to food/capsuled being recommended and proximity to ghost as not 
    recommendable. 
    """
    pos = currentGameState.getPacmanPosition() #Tuple
    food = currentGameState.getFood() # 2D Array of bool vals
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    if food.asList():
        food_distance = min(manhattanDistance(pos, f) for f in food.asList())
        score += 10 / (1 + food_distance)

    for ghost, scared in zip(ghostStates, scaredTimes):
        ghostDist = manhattanDistance(pos, ghost.getPosition())
        if scared > 0:
            score += 20 / (1 + ghostDist)
        else:
            score -= 50 / (1 + ghostDist)
    
    if capsules:
        capDist = min(manhattanDistance(pos, c) for c in capsules)
        score += 100 / (1 + capDist)

    return score



# Abbreviation
better = betterEvaluationFunction
