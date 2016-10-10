# qlearningagent.py


from baselineTeam import ReflexCaptureAgent
import random,util,math
from game import *
from game import Directions, Actions
import pickle

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
    def registerInitialState(self, gameState):
        actionFn = lambda state: state.getLegalActions()
        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = 1
        self.epsilon = 0.05
        self.alpha = 0.2
        self.discount = 0.8

        self.start = gameState.getAgentPosition(self.index)
        ReflexCaptureAgent.registerInitialState(self, gameState)
        self.weights = util.Counter()
        # self.weights['closest-food'] = -0.1
        # self.weights['distanceToMid'] = -0.1
        self.loadQWeight()

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

        # Pick Action
        if util.flipCoin(self.epsilon):
            legalActions = gameState.getLegalActions(self.index)
            if len(legalActions) == 0:
                print "*******************yep"
                return None
            action = random.choice(legalActions)
        else:
            value, action = self.computeValueActionFromQValues(gameState)
        self.lastState  = gameState
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
            distanceToMid = (1+myState.numCarrying) * float(minDistance) / mapArea
            # if features["ghostDistance"] != 0:
            #     distanceToMid *= features["ghostDistance"]
            return distanceToMid

        def checkPacman(walls, next_x, red):
            #if next_x is on the red side, and the agent is red, false.
            #if next_x is on the red side, and the agent is !red, true.
            #if next_x is on the blue side, and the agent is red, false.
            #if next_x is on the blue side, and the agent is !red, true.
            midX = int(walls.width / 2 - 1)
            return (next_x <= midX) != red


        # extract the grid of food and wall locations and get the ghost locations
        food = self.getFood(state)
        foodList = food.asList()
        walls = state.getWalls()
        # enemies = [state.getAgentPosition(i) for i in self.enemyIndex]
        enemies = [state.getAgentState(i) for i in self.enemyIndex]
        mapArea = (walls.width * walls.height)
        for agentIndex in self.agentsOnTeam:
            if agentIndex == self.index:
                myState = state.getAgentState(agentIndex)
            else:
                allyState = state.getAgentState(agentIndex)

        features = util.Counter()
        features["bias"] = 1.0


        # compute the location of pacman after he takes the action
        x, y = myState.getPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        ghost_count = 0;
        pacman_count = 0
        for enemy in enemies:
            if enemy is None:
                continue
            enemy_pos = enemy.getPosition()
            if enemy_pos is None:
                continue

            distToEnemy = float(self.getMazeDistance((next_x, next_y), enemy.getPosition())) 
            if distToEnemy == 0:
                distToEnemy = 0.1
            #Note we want to check if the enemy is a pacman where we are moving to, not where they currently are
            enemyIsPacman = checkPacman(walls, next_x, not self.red)
            if (    (enemyIsPacman and not myState.scaredTimer > 1) 
                    or (not enemyIsPacman and enemy.scaredTimer > 1)  ):

                if enemy.getPosition() == (next_x, next_y):
                    features["killpacman"] = 1.0
                    continue
                features["#-of-pacmen-1-step-away"] += (next_x, next_y) in Actions.getLegalNeighbors(enemy_pos, walls)
                #similar to closest food but for delicous enemy pacmen
                if distToEnemy < features["invaderDistance"] or features["invaderDistance"] == 0:
                    features["invaderDistance"] = 1 / distToEnemy / mapArea
            else:
                features["#-of-ghosts-1-step-away"] += (next_x, next_y) in Actions.getLegalNeighbors(enemy_pos, walls)
                if distToEnemy < features["ghostDistance"] or features["ghostDistance"] == 0:
                    features["ghostDistance"] = 1 / distToEnemy / mapArea
        
        # features["friendDistance"] = float(self.getMazeDistance((next_x, next_y), allyState.getPosition())) / mapArea
        features['distanceToMid'] = featDistanceToMid(walls, next_x, next_y)
        
        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"]:
            features["eats-food"] = 1.0
            if len(foodList) > 2:
                foodDist = min([self.getMazeDistance((next_x, next_y), f) for f in foodList])
                if foodDist == 0:
                    foodDist = 0.1
                if foodDist is not None:
                    # make the distance a number less than one otherwise the update
                    # will diverge wildly
                    features["closest-food"] = float(foodDist) / mapArea
                    # features["closest-food-exp"] = 1 / float(foodDist) / mapArea
        

        features.divideAll(10.0)
        return features

    def getWeights(self, gameState, action):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        return self.weights * self.getFeatures(state,action)

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
            reward = self.getCustomReward(state, self.lastState)
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state.makeObservation(self.index)

    def observeTransition(self, state,action,nextState,deltaReward):
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

        scoreReward = (state.getScore() - newState.getScore()) * 10
        distance = self.getMazeDistance(myPos, myPosNew)
        if distance > 2:
            #we probably just got eaten
            scoreReward += -10

        for eInd in self.enemyIndex:
            enemy_pos = state.getAgentPosition(eInd)
            if enemy_pos == myPosNew:
                scoreReward += 10
                

        if self.getFood(state)[myPosNew[0]][myPosNew[1]]:
            scoreReward += 1
        
        return scoreReward

    def loadQWeight(self):
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

    def saveQWeights(self):
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
        self.saveQWeights()
        # ReflexCaptureAgent.final(state)

    # def final(self, state):
    #     """
    #       Called by Pacman game at the terminal state
    #     """
    #     deltaReward = state.getScore() - self.lastState.getScore()
    #     self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
    #     self.stopEpisode()

    #     # Make sure we have this var
    #     if not 'episodeStartTime' in self.__dict__:
    #         self.episodeStartTime = time.time()
    #     if not 'lastWindowAccumRewards' in self.__dict__:
    #         self.lastWindowAccumRewards = 0.0
    #     self.lastWindowAccumRewards += state.getScore()

    #     NUM_EPS_UPDATE = 100
    #     if self.episodesSoFar % NUM_EPS_UPDATE == 0:
    #         print 'Reinforcement Learning Status:'
    #         windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
    #         if self.episodesSoFar <= self.numTraining:
    #             trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
    #             print '\tCompleted %d out of %d training episodes' % (
    #                    self.episodesSoFar,self.numTraining)
    #             print '\tAverage Rewards over all training: %.2f' % (
    #                     trainAvg)
    #         else:
    #             testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
    #             print '\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining)
    #             print '\tAverage Rewards over testing: %.2f' % testAvg
    #         print '\tAverage Rewards for last %d episodes: %.2f'  % (
    #                 NUM_EPS_UPDATE,windowAvg)
    #         print '\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)
    #         self.lastWindowAccumRewards = 0.0
    #         self.episodeStartTime = time.time()

    #     if self.episodesSoFar == self.numTraining:
    #         msg = 'Training Done (turning off epsilon and alpha)'
    #         print '%s\n%s' % (msg,'-' * len(msg))