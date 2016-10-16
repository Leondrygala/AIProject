# myTeam.py
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
import copy

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.map = gameState.getWalls()
    if self.red:
      self.enemyIndex = gameState.getBlueTeamIndices()
    else:
      self.enemyIndex = gameState.getRedTeamIndices()

    initialMapState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    actions.remove("Stop")

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights



def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

def threeWall(self,state, position):

    i, j = position
    counter = 0
    driction=""
    if not state[i][j]:

      if i < state.width - 1:
          if state[i + 1][j]:
            counter += 1
          else:
            driction='E'
      if i >  0:
          if state[i - 1][j]:
            counter += 1
          else:
            driction = 'W'
      if j< state.height - 1:
          if state[i][j + 1]:
            counter += 1
          else:
            driction = 'N'
      if j>0:
          if state[i][j - 1]:
            counter += 1
          else:
            driction = 'S'

    if counter == 3:
      self.end[(i, j)]=driction

def isInTunnel(state, position,OldDirection):
  i, j = position
  direction=[]

  if not state[i][j]:
    if i < state.width - 1:
      if not state[i + 1][j]:
        direction.append('E')

    if i > 0:
      if not state[i - 1][j]:
        direction.append('W')

    if j < state.height - 1:
      if not state[i][j + 1]:
        direction.append('N')

    if j > 0:
      if not state[i][j - 1]:
        direction.append('S')

  if len(direction)==2:
    if direction[0]==oppositeDirection(direction[1]):
      newDirection=OldDirection

    else:
      for item in direction:
        if oppositeDirection(OldDirection)!=item:
            newDirection=item


    return (True,newDirection)
  else:
    return (False,False)

def oppositeDirection(direction):
  if direction=='N':
    return 'S';
  if direction == 'S':
    return 'N'
  if direction == 'E':
    return 'W'
  if direction == 'W':
    return 'E'

def checkNeighbor(self,walls,position,direction):
    i, j = position

    if direction=='N':
      j+=1

    if direction == 'S':
      j-=1

    if direction == 'W':
      i-=1

    if direction == 'E':
      i += 1

    isTunnel,newDirection=isInTunnel(walls, (i, j), direction)


    if isTunnel:
        self.endNew[(i, j)] = newDirection

def initialMapState(self, gameState):
  walls = gameState.getWalls()
  width, height = gameState.data.layout.width, gameState.data.layout.height

  self.end = {}
  self.deadEnd = {}
  self.endNew = {}

  if self.red:

    for i in range(width/2, width):
      for j in range(1, height):
         threeWall(self,walls, (i, j))

    self.deadEnd=merge_two_dicts(self.deadEnd, copy.copy(self.end))

    while len(self.end)!=0:
      for item in self.end:
        checkNeighbor(self,walls,item,self.end[item])

      self.deadEnd = merge_two_dicts(self.deadEnd, copy.copy(self.endNew))
      self.end=copy.copy(self.endNew)
      self.endNew={}

  else:

    for i in range(0, width/2):
      for j in range(1, height):
        threeWall(self, walls, (i, j))
    self.deadEnd = merge_two_dicts(self.deadEnd, copy.copy(self.end))

    while len(self.end) != 0:
      for item in self.end:
        checkNeighbor(self, walls, item, self.end[item])

      self.deadEnd = merge_two_dicts(self.deadEnd, copy.copy(self.endNew))
      self.end = copy.copy(self.endNew)
      self.endNew = {}

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z


class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()

    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # get distance to home-side
    midX = int(self.map.width / 2 - self.red) 
    yaxis = range(0, self.map.height)
    ywall = []
    #check for walls and record them
    for y in yaxis:
      if self.map[midX][y]:
        ywall.append(y)
    # remove walls from yaxis
    for y in ywall:
      yaxis.remove(y)
    # search for the closest sqaure on the home-side
    minDistance = 9999
    for y in yaxis:
      newDistance = self.getMazeDistance(myPos, (midX, y))
      if newDistance < minDistance:
        minDistance = newDistance
    features['distanceToMid'] = minDistance

    #Scared of enemy ghosts
    scaredValue = None
    for enemyInd in self.enemyIndex:
      scaredValue = 0
      enemyPos = gameState.getAgentPosition(enemyInd)
      if enemyPos is None:
        continue
      # if enemy is a white ghost
        # continue
      # if we're on our side of the board, 
      # or the pacman is on our side of the board, continue
      if self.red:
        if myPos[0] <= midX or enemyPos[0] <= midX:
          continue
      else:
        if myPos[0] >= midX or enemyPos[0] >= midX:
          continue
      scaredValue =+ self.getMazeDistance(myPos, enemyPos)
    if scaredValue != 0:
      scaredValue = 10.0/scaredValue
      # print scaredValue
    features['distanceToEnemy'] = scaredValue

    return features

  def getWeights(self, gameState, action):
    numFoodCarrying = gameState.getAgentState(self.index).numCarrying
    return {'successorScore': 100, 
            'distanceToFood': -1.0,
            'distanceToMid': -numFoodCarrying/1.7,
            'distanceToEnemy': -(1.0 + numFoodCarrying) # we are more scared if we've got a big payload
            }

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):


    successor = self.getSuccessor(gameState, action)


    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    features = util.Counter()
    getOffensiveFeatures(self, gameState, action,features)
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    gostList = [a for a in enemies if a.isPacman==False and a.getPosition() != None]

    features['dotGo'] =0
    if gameState.getAgentState(self.index).isPacman:
        if len(gostList)!=0:
            minDistance=999999
            for gost in gostList:

                distance = self.getMazeDistance(myPos, gost.configuration.pos)
                if minDistance>distance:
                  minDistance=distance
            print 'dis'
            print  minDistance
            if minDistance <=3:
                print successor.getAgentPosition
                if successor.getAgentPosition in self.endOld.keys():
                     features['dotGo']=1
                     print successor.getAgentPosition



    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    enemyList=[]
    foodList = self.getFood(successor).asList()
    # print foodList
    i=0;
    for enemy in gameState.blueTeam:
      if gameState.data.agentStates[enemy].isPacman:
         if gameState.data.agentStates[enemy].configuration !=None:
           gameState.data.agentStates[enemy].configuration.pos
           enemyList.append(gameState.data.agentStates[enemy].configuration.pos)
           i+=1

    print i
    if i > 0:
      dists = [self.getMazeDistance(myPos, pos) for pos in enemyList]
      features['invaderDistance'] = min(dists)
      features['onDefense'] = 1
      features['numInvader']=len(invaders)
    else:
      features['onDefense'] = 0

      if len(foodList) > 0:  # This should always be True,  but better safe than sorry
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        features['distanceToFood'] = minDistance

    if action == Directions.STOP:
      features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev:
      features['reverse'] = 1


    return features

  def getWeights(self, gameState, action):

    if gameState.getAgentState(self.index).isPacman:
      numFoodCarrying = gameState.getAgentState(self.index).numCarrying
      #max
      return {'successorScore': 1000,
              'distanceToFood': -10.0,
              'distanceToMid': -numFoodCarrying*10,
              'invaderDistance': 30,
              'numInvader':-300,
              'onDefense': 1000,
              'stop': -100, 
              'reverse': -2,
              'dotGo':-100000
              }
    else:
     return {'successorScore': 1000,'distanceToFood': -10.0,'numInvader':-300, 'onDefense': 100, 'invaderDistance': -30, 'stop': -100, 'reverse': -2}


def getTeammate(self):
	if self.index == 0:
	  return 2
	elif self.index == 1:
	  return 3
	elif self.index == 2:
	  return 0
	elif self.index == 3:
	  return 1
	else:
	  raise ValueError('Teammate error!')

def getEnemy(self):
	if self.index == 0:
	  return [1,3]
	elif self.index == 1:
	  return [0,2]
	elif self.index == 2:
	  return [1,3]
	elif self.index == 3:
	  return [0,2]
	else:
	  raise ValueError('Teammate error!')

def getOffensiveFeatures(self, gameState, action,features):
  initialMapState(self, gameState)
  successor = self.getSuccessor(gameState, action)
  foodList = self.getFood(successor).asList()
  # print foodList
  features['successorScore'] = -len(foodList)  # self.getScore(successor)

  # Compute distance to the nearest food
  if len(foodList) > 0:  # This should always be True,  but better safe than sorry
    myPos = successor.getAgentState(self.index).getPosition()
    minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
    features['distanceToFood'] = minDistance

  # get distance to home-side
  midX = int(self.map.width / 2 - self.red)
  yaxis = range(0, self.map.height)
  ywall = []
  # check for walls and record them
  for y in yaxis:
    if self.map[midX][y]:
      ywall.append(y)
  # remove walls from yaxis
  for y in ywall:
    yaxis.remove(y)
  # search for the closest sqaure on the home-side
  minDistance = 9999
  for y in yaxis:
    newDistance = self.getMazeDistance(myPos, (midX, y))
    if newDistance < minDistance:
      minDistance = newDistance
  features['distanceToMid'] = minDistance

  # Scared of enemy ghosts
  scaredValue = None
  for enemyInd in self.enemyIndex:
    scaredValue = 0
    enemyPos = gameState.getAgentPosition(enemyInd)
    if enemyPos is None:
      continue
      # if enemy is a white ghost
      # continue
    # if we're on our side of the board,
    # or the pacman is on our side of the board, continue
    if self.red:
      if myPos[0] <= midX or enemyPos[0] <= midX:
        continue
    else:
      if myPos[0] >= midX or enemyPos[0] >= midX:
        continue
    scaredValue = + self.getMazeDistance(myPos, enemyPos)
  if scaredValue != 0:
    scaredValue = 10.0 / scaredValue
    # print scaredValue
  features['distanceToEnemy'] = scaredValue
  return features





# def mark_until_junction(state, p):
# 	walker = p
# 	directions = my_getLegalAction(state,p,None)
# 	flag = True
# 	while len(directions) == 1 or flag:
# 	  flag = False
# 	  next = my_walk(walker, directions[0])
# 	  if state[next] != 0:  #two deadend converge on a same junction
# 		if state[next] < state[walker]-1:
# 		  state[next] = state[walker]-1
# 		i,j = next
# 		directions = []
# 		if state[(i,j+1)] == 0:
# 		  directions.append('E')
# 		if state[(i,j-1)] == 0:
# 		  directions.append('W')
# 		if state[(i+1,j)] == 0:
# 		  directions.append('N')
# 		if state[(i-1,j)] == 0:
# 		  directions.append('S')
# 		if len(directions) == 1:
# 		  flag = True
# 		  walker = next
# 		  continue
# 	  elif state[next] == 0:
# 		state[next] = state[walker]-1
# 	  walker = next
# 	  if directions[0] == 'N':
# 		dir_from = 'S'
# 	  elif directions[0] == 'S':
# 		dir_from = 'N'
# 	  elif directions[0] == 'W':
# 		dir_from = 'E'
# 	  else:
# 		dir_from = 'W'
# 	  directions = my_getLegalAction(state,walker,dir_from)
# 	return walker
#
# def my_walk(p, direction):
# 	i,j = p
# 	if direction == 'N':
# 	  return (i+1,j)
# 	elif direction == 'S':
# 	  return (i-1,j)
# 	elif direction == 'E':
# 	  return (i,j+1)
# 	else:
# 	  return (i,j-1)
#
#
# def my_getLegalAction(state, p, exclude):  #get legal action for a point, state is walls
# 	ret = []
# 	if state[p] == -1:
# 	 return []
# 	i,j = p
# 	if not state[(i+1,j)] == -1 and not exclude == 'N':
# 	  ret.append('N')
# 	if not state[(i-1,j)] == -1 and not exclude == 'S':
# 	 ret.append('S')
# 	if not state[(i,j+1)] == -1 and not exclude == 'E':
# 	  ret.append('E')
# 	if not state[(i,j-1)] == -1 and not exclude == 'W':
# 	  ret.append('W')
# 	return ret
