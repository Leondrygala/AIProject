# qlearningagent.py


from baselineTeam import ReflexCaptureAgent
import random,util,math
from game import *
from game import Directions, Actions

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
        self.weights["closest-food"] = -0.1

        self.lastAction = None
        self.lastState = None

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # Pick Action
        if util.flipCoin(self.epsilon):
            legalActions = gameState.getLegalActions(self.index)
            if len(legalActions) == 0:
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
        legalActions = state.getLegalActions(self.index)
        if len(legalActions) == 0:
            return (0, None)
        
        max_value = None # None is the 'smaller' than any number
        max_action = None
        for a in legalActions:
            value = self.getQValue(state, a)
            if value > max_value:
                max_value = value
                max_action = a 

        return (max_value, max_action)


    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        
        food = self.getFood(state).asList()
        walls = state.getWalls()
        # ghosts = state.getAgentPosition(enemyInd)

        features = util.Counter()
        features["bias"] = 1.0


        # compute the location of pacman after he takes the action
        x, y = state.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # # count the number of ghosts 1-step away
        # features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # # if there is no danger of ghosts then add the food feature
        # if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
        #     features["eats-food"] = 1.0

        dist = min([self.getMazeDistance((next_x, next_y), f) for f in food])
        # dist = self.getMazeDistance((next_x, next_y), (1,1)) 
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
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
        scoreReward = state.getScore() - newState.getScore()
        return scoreReward


    def final(self, state):
        print 'made final'
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