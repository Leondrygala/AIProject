# qlearningagent.py


from baselineTeam import ReflexCaptureAgent
import random,util,math
from game import *
from game import Directions, Actions
from capture import AgentRules, GameState
import pickle
import time
import copy
from util import PriorityQueue

def createTeam(firstIndex, secondIndex, isRed,
               first = 'QLearningAgent', second = 'QLearningAgent'):
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

class QLearningAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    FOODTARGET = []
    def registerInitialState(self, gameState):
        self.numTraining = 1
        self.epsilon = 0.1
        self.alpha = 0.2
        self.discount = 0.95

        self.start = gameState.getAgentPosition(self.index)
        ReflexCaptureAgent.registerInitialState(self, gameState)
        self.weights = util.Counter()
        # self.weights['closest-food'] = -0.1
        # self.weights['distanceToMid'] = -0.05
        # self.weights['killpacman'] = 1
        # self.weights['#-of-ghosts-1-step-away'] = -1
        self.loadQWeight("qfile_0")
        self.checkWeights()

        initialMapState(self, gameState)

        self.lastAction = None
        self.lastState = None

        if self.red:
          self.enemyIndex = gameState.getBlueTeamIndices()
          self.agentsOnTeam = gameState.getRedTeamIndices()
        else:
          self.enemyIndex = gameState.getRedTeamIndices()
          self.agentsOnTeam = gameState.getBlueTeamIndices()

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        self.checkWeights()
        # Pick Action
        legalActions = gameState.getLegalActions(self.index)
        legalActions.remove('Stop')
        if len(legalActions) == 1:
                action = legalActions[0]

        elif util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            # (value, action) = max((uniformCostSearchEvaluate(self, gameState, a, 20), a) for a in legalActions)
            # print value, action
            # value, action = self.computeValueActionFromQValues(gameState)
            result = self.simulate(gameState)
            if result==None:
                print "WARNING USING RANDOM ACTION"
                action = random.choice(legalActions)
            else:
                action = result


        self.lastState  = gameState

        if action in gameState.getLegalActions(self.index):

            self.lastAction = action
            return action
        else:
            action = random.choice(gameState.getLegalActions(self.index))
            self.lastAction = action
            return action

    def computeValueActionFromQValues(self, state):
        """
          Compute the best value, action pair to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return 0, None.
        """
        # if random:
        #     #we should pick a mostly random action
        legalActions = state.getLegalActions(self.index)
        if len(legalActions) == 0:
            return (0, None)
        
        max_value = None # None is the 'smaller' than any number
        max_action = None
        for a in legalActions:
            value = self.getQValue(state, a)
            if (value > max_value):
                max_value = value
                max_action = a

        # if random:
        #     legalActions = gameState.getLegalActions(self.index)
        #     if len(legalActions) == 0:
        #     action = random.choice(legalActions)
        #     while action != max_action:

        return (max_value, max_action)

    def getFeatures(self, state, action):
        # Finds the minimum distance to our home state
        def featDistanceToMid(walls, next_x, next_y):
            mapArea = float(walls.width * walls.height)
            # get distance to home-side
            midX = int(walls.width / 2 - self.red ) 
            yaxis = range(0, walls.height)
            ywall = []
            #check for walls and record them
            for y in yaxis:
              if walls[midX][y]:
                ywall.append(y)
            # remove walls from yaxis
            for y in ywall:
              yaxis.remove(y)


            # search for the closest sqaure on the home-side
            minDistance = min(self.getMazeDistance((next_x, next_y), (midX, y)) for y in yaxis)
            distanceToMid = (myState.numCarrying) * float(minDistance) / mapArea
            # if features["ghostDistance"] != 0:
            #     distanceToMid /= features["ghostDistance"] / 3
            return distanceToMid

        nextState = self.getSuccessor(state, action)
        # extract the grid of food and wall locations and get the ghost locations
        food = self.getFood(nextState)
        foodList = food.asList()
        walls = nextState.getWalls()
        #We get the enemy states based on the old state. They shouldn't move unless we ate them
        enemies = [state.getAgentState(i) for i in self.enemyIndex]
        enemies_new = [nextState.getAgentState(i) for i in self.enemyIndex]
        mapArea = (walls.width * walls.height)
        for agentIndex in self.agentsOnTeam:
            if agentIndex == self.index:
                myState = nextState.getAgentState(agentIndex)
                oldState = state.getAgentState(agentIndex)
            else:
                allyState = nextState.getAgentState(agentIndex)

        features = util.Counter()
        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        next_x, next_y = nextState.getAgentPosition(self.index)


        # calculate the ghosts and pacman around us
        ghost_count = 0
        pacman_count = 0
        enemyCarryingCount = 0;
        for enemy in enemies:
            if enemy is None:
                continue
            enemy_pos = enemy.getPosition()
            if enemy_pos is None:
                continue

            distToEnemy = float(self.getMazeDistance((next_x, next_y), enemy.getPosition())) 
            if distToEnemy == 0:
                distToEnemy = 0.1
            enemy.isPacman 
            #Note we want to check if the enemy is a pacman where we are moving to, not where they currently are
            enemyIsPacman = self.checkPacman(walls.width, next_x, not self.red)
            if (    (enemyIsPacman and not myState.scaredTimer > 1) 
                    or (not enemyIsPacman and enemy.scaredTimer > 1)  ):
                # Either they are a pacman and we are a ghost
                # or we are a pacman and they are a scared ghost

                #If we are standing where our enemy used to be, we killed it
                if enemy_pos == (next_x, next_y):
                    features["killpacman"] = 1.0
                    continue
                # features["#-of-pacmen-1-step-away"] += (next_x, next_y) in Actions.getLegalNeighbors(enemy_pos, walls)
                #similar to closest food but for delicous enemy pacmen
                if distToEnemy < features["invaderDistance"] or features["invaderDistance"] == 0:
                    features["invaderDistance"] = distToEnemy
            else:
                features["#-of-ghosts-1-step-away"] += (next_x, next_y) in Actions.getLegalNeighbors(enemy_pos, walls)
                # Note distToEnemy is set at 0.1 if it is actually 0
                if distToEnemy < features["ghostDistance"] or features["ghostDistance"] == 0:
                    features["ghostDistance"] = distToEnemy
        friendDistance = float(self.getMazeDistance((next_x, next_y), allyState.getPosition())) + 0.1
        features["friendDistance"] = 0.01 / friendDistance / mapArea
        # features["#-of-friends-3-steps-away"] = self.getMazeDistance((next_x, next_y), allyState.getPosition()) <= 3
        features['distanceToMid'] = featDistanceToMid(walls, next_x, next_y)
        
        # if there is no danger of ghosts then add the food feature
        if features["ghostDistance"] > 4 or features["ghostDistance"] == 0: #remember this can't be zero even if the real value is 0, see above
            if myState.numCarrying > oldState.numCarrying:
                features["eats-food"] = 1.0
            if len(foodList) > 2:
                myOldFoodTarget = None
                allyFoodTarget = None
                for (index, target_coords) in QLearningAgent.FOODTARGET:
                    if index == self.index:
                        myOldFoodTarget = (index, target_coords)
                    else:
                        #TODO also remove neighbough foods
                        allyFoodTarget = target_coords
                if not myOldFoodTarget is None:
                    QLearningAgent.FOODTARGET.remove(myOldFoodTarget)
                if not allyFoodTarget is None and allyFoodTarget in foodList:
                    foodList.remove(allyFoodTarget)

                foodDist, closestFood = min( [ (self.getMazeDistance((next_x, next_y), f), f) for f in foodList])
                if foodDist == 0:
                    foodDist = 0.1
                if foodDist is not None:
                    QLearningAgent.FOODTARGET.append((self.index, closestFood))
                    features["closest-food"] = 1 / float(foodDist) / mapArea
        else:    
            if (next_x,next_y) in self.deadEnd.keys():
                features['deadEnd'] = 1.0

            capsules = self.getCapsules(nextState)
            if len(capsules) > 0:
                capDist = min([self.getMazeDistance((next_x, next_y), c) for c in capsules])
                if capDist == 0:
                    capDist = 0.1
                if capDist is not None:
                    features["closest-cap"] = float(capDist) / mapArea
        
        # if next_y < walls.height/2.0 and allyState.getPosition()[1] < walls.height/2.0:
        #     features["teamwork"] = 0.01
        if features["ghostDistance"]:
            features["ghostDistance"] = 1 / features["ghostDistance"]
        if features["invaderDistance"]:
            features["invaderDistance"] = 1 / features["invaderDistance"]
        
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]

        features.divideAll(10.0)
        return features

    def getWeights(self, gameState, action):
        return self.weights

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


    def getSuccessorEnemy(self, gameState, action, enemy_index):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(enemy_index, action)
        pos = successor.getAgentState(enemy_index).getPosition()
        if pos is None:
            return None
        if pos != nearestPoint(pos):
          # Only half a grid position was covered
          return successor.generateSuccessor(self.index, action)
        else:
          return successor


    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        QValue = self.weights * self.getFeatures(state,action)
        # print 'Total Q:', QValue
        return QValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """     
        difference = (reward - self.getQValue(state,action) + 
                    self.discount * self.computeValueActionFromQValues(nextState)[0] )

        feature_vector = self.getFeatures(state,action)
        feature_vector.update((key,f_value * self.alpha * difference) 
            for key, f_value in feature_vector.items())
        

        for f in feature_vector:
            self.weights[f] += feature_vector[f]

    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            reward = self.getCustomReward(self.lastState, state)
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state.makeObservation(self.index)

    def observeTransition(self, state, action,nextState,deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments
        """
        # self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)

    def getCustomReward(self, state, newState):
        myPos = state.getAgentPosition(self.index)
        myPosNew = newState.getAgentPosition(self.index)
        scoreReward = (state.getScore() - newState.getScore())
        distance = self.getMazeDistance(myPos, myPosNew)
        if distance > 3:
            #we probably just got eaten
            scoreReward += -5
            print "Ouch, agent ", self.index, " just got eaten!"
        else:
            for eInd in self.enemyIndex:
                enemy_pos = state.getAgentPosition(eInd)
                enemy_pos2 = newState.getAgentPosition(eInd)
                if (enemy_pos == myPosNew and
                    not self.checkPacman(state.getWalls().width, myPosNew[0], self.red) ):
                    print "Yum, agent ", self.index, " just ate ", eInd, "!"
                    scoreReward += 2

        if self.getFood(state)[myPosNew[0]][myPosNew[1]]:
            scoreReward += 1
        
        return scoreReward

    def checkPacman(self, width, next_x, red):
        #if next_x is on the red side, and the agent is red, false.
        #if next_x is on the red side, and the agent is !red, true.
        #if next_x is on the blue side, and the agent is red, false.
        #if next_x is on the blue side, and the agent is !red, true.
        midX = int(width / 2 - 1)
        return (next_x <= midX) != red

    def loadQWeight(self, file_name = None):
        if file_name is None:
            if self.index == 0:
                file_name = "qfile_0"
            elif self.index == 1:
                file_name = "qfile_1"
            elif self.index == 2:
                file_name = "qfile_2"
            else:
                file_name = "qfile_3"

        with open(file_name, 'rb') as inputfile:
            self.weights = pickle.load(inputfile)

    def saveQWeights(self, file_name = None):
        if file_name is None:
            if self.index == 0:
                file_name = "qfile_0"
            elif self.index == 1:
                file_name = "qfile_1"
            elif self.index == 2:
                file_name = "qfile_2"
            else:
                file_name = "qfile_3"
        with open(file_name, 'wb') as outputfile:
            pickle.dump(self.weights, outputfile)

    def final(self, state):
        self.saveQWeights("qfile_0")
        QLearningAgent.FOODTARGET = []
        # ReflexCaptureAgent.final(state)

    def checkWeights(self):
        # return
        # should be negative
        # print "distance to mid: ", self.weights['distanceToMid']
        if self.weights['closest-cap'] > 0:
            self.weights['closest-cap'] = -self.weights['closest-cap']
        if self.weights['distanceToMid'] > 0:
            self.weights['distanceToMid'] = -self.weights['distanceToMid']
        if self.weights['ghostDistance'] > 0:
            self.weights['ghostDistance'] = -self.weights['ghostDistance']
        if self.weights['#-of-ghosts-1-step-away'] > 0:
            self.weights['#-of-ghosts-1-step-away'] = -self.weights['#-of-ghosts-1-step-away']
        if self.weights['deadEnd'] > 0:
            self.weights['deadEnd'] = -self.weights['deadEnd']
        if self.weights["teamwork"] > 0:
            self.weights['teamwork'] = -self.weights['teamwork']
        if self.weights["stop"] > 0:
            self.weights['stop'] = -self.weights['stop']
        if self.weights["reverse"] > 0:
            self.weights['reverse'] = -self.weights['reverse']
        if self.weights['friendDistance'] > 0:
            self.weights['friendDistance'] = -self.weights['friendDistance']


        #should be postitive
        if self.weights['closest-food'] < 0:
            self.weights['closest-food'] = -self.weights['closest-food']
        if self.weights['invaderDistance'] < 0:
            self.weights['invaderDistance'] = -self.weights['invaderDistance']
        if self.weights['killpacman'] < 0:
            self.weights['killpacman'] = -self.weights['killpacman']
        if self.weights['#-of-pacmen-1-step-away'] < 0:
            self.weights['#-of-pacmen-1-step-away'] = -self.weights['#-of-pacmen-1-step-away']

    def simulate(self, state):
        # takes state
        #
        Qvalue = []
        from util import Queue
        queue = Queue()

        positon0 = state.getAgentState(self.index).getPosition()
        x, y = positon0

        positon0 = (int(x), int(y))
        queue.push(([positon0],0,state))

        while (not queue.isEmpty()):
            # positonList = queue.pop()
            positonListOld,QvalueOld,successorOld= queue.pop()
            positonList = copy.copy(positonListOld)
            actions =successorOld.getLegalActions(self.index)
            for action in actions:
                positonListnew = copy.copy(positonList)
                if action != 'Stop':

                    successor= self.getSuccessor(successorOld, action)
                    positon = successor.getAgentState(self.index).getPosition()
                    x, y = positon
                    positon = (int(x), int(y))
                    if positonListnew.count(positon)<2:

                        positonListnew.append(positon)

                        qvalue = self.getQValue(successorOld, action)

                        if len(positonListnew) < 5:
                            q=QvalueOld + math.pow(0.5, len(positonListnew)) * qvalue
                            queue.push((positonListnew, q,
                                       successor))
                        if len(positonListnew)==5:

                            Qvalue.append((copy.copy(positonListnew), QvalueOld + math.pow(0.7, len(positonListnew)) * qvalue,
                                       successor))

        history = self.observationHistory
        historyLength=len(history)
        if historyLength>=4:
            direction0 = history[historyLength-1].getAgentState(self.index).getDirection()
            # print direction0
            direction1 = history[historyLength - 2].getAgentState(self.index).getDirection()
            # print direction1
            direction2 = history[historyLength - 3].getAgentState(self.index).getDirection()
            # print direction2

            if direction0==oppositeDirection2(direction1) and  direction0==direction2:
                if state!= None and  state.getLegalActions(self.index)!=None :
                    if direction2 in state.getLegalActions(self.index):

                        return direction2

                    return None

        return self.maximumValue(Qvalue,state)

    def maximumValue(self,QvalueDict,state):
        max=-9999
        maxList=[];
        for item in QvalueDict:
            x,y,z=item
            x=copy.copy(x)
            if y>max and len(x)>4:
                max=y
                maxList=x
        # print (max,maxList)

        # self.debugDraw(maxList,[1,0,0],True)
        if len(maxList)==0:
            return None

        actionPositon = maxList[1]
        postion = state.getAgentPosition(self.index)
        x,y=postion
        dx,dy=actionPositon
        action= Actions.vectorToDirection((dx-x,dy-y))
        return action

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

def oppositeDirection2(direction):
  if direction=='North':
    return 'South';
  if direction == 'South':
    return 'North'
  if direction == 'East':
    return 'West'
  if direction == 'West':
    return 'East'

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