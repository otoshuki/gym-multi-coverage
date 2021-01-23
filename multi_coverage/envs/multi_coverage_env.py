#Guining Pertin
#Simulator Environment for BTP Mk. I
#21-09-20
#Updating reward for avoiding uneccesary goal changes
# - Devashish Taneja
#Import libraries
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from multi_coverage.envs.bfs import *

class agent_class:
    """
    Description:
        The agent_class class stores agent specific data and helper function
    """
    def __init__(self,start_loc,K):
        self.state = None
        self.steps = None
        self.curr  = start_loc
        self.K = K

    def perform_action(self, action):
        '''
        Arguments:
            action  :   Integer [0, K]
        '''
        #Putting randomness into the environment
        noise = np.random.choice([1,0], p=[0.1, 0.9])
        if noise:
            action = np.random.randint(self.K+1)
        move = self.steps[action]
        if move == 'O': return self.curr
        elif move == 'U': self.curr[0] -= 1
        elif move == 'D': self.curr[0] += 1
        elif move == 'R': self.curr[1] += 1
        elif move == 'L': self.curr[1] -= 1

class MultiCoverageEnv(gym.Env):
    """
    Description:
        We have M agents spawned at fixed locations for each session in an NxN
        map with obstacles and K randomly spawned goals. All agents share the
        same environment but have different states and actions that can be
        addressed independently. The input action for step is an M dim vector with
        each entry being the action type for each agent. The output state is an
        M dim space with K dimensional elements for each agent.

    Actions:
        (a1,a2,a3,...,aM) each element being:
        Type: Discrete(K+1)
        Num     Action
        0       Stop
        1       Move towards goal 1
        2       Move towards goal 2
        ...
        K       Move towards goal K

    State:
        (s1,s2,s3,...,sM) each element being:
        Type: Box(K), np.int16
        Num     Observation                 Min         Max
        0       Distance from goal 1        0           N**2
        1       Distance from goal 2        0           N**2
        ...
        K-1     Distance from goal K        0           N**2

    Reward:
        -1   : for every timestep
        +1*b : for each goal being worked on, until weight ends

    Starting State:
        Random map generated with obstacles area: total area ~ 1/20
        Fixed locations for each agent with max_val-random energy
        Random locations for each goal with max_val-random weight
        Goals sorted in decreasing order with goal init as 0

    Episode Termination:
        Episode length greater than t_lim
        Completing all goals

    Map:
        0   :   empty cells, completed goals
        1   :   agents
        2   :   goals - low weights
        3   :   goals - mid weights
        4   :   goals - high weights
        5   :   walls
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, N=30, M=5, K=5, t_lim=500):
        '''
        Arguments:
            N : length of sides of the map
            M : number of agents
            K : number of goals
        '''
        super(MultiCoverageEnv, self).__init__()
        #Set game parameters
        self.N = N
        self.M = M
        self.K = K
        self.t_lim = t_lim
        self.curr_t = 0
        self.prev_action =  [-1 for i in range(M)]
        #Action space
        actions_list = [spaces.Discrete(K+1) for i in range(M)]
        self.action_space = spaces.Tuple(tuple(actions_list))
        #Observation space
        states_list = [spaces.Box(np.zeros(K),
                        np.ones(K)*(N**2),
                        dtype=np.uint16) for i in range(M)]
        self.observation_space = spaces.Tuple(tuple(states_list))
        self.goal_r_multiplier = 2
        
    def step(self, action):
        '''
        Arguments:
            action  :   M tuple, each element from (0 to K)
        Returns:
            state   :   M tuple, each element K+1 dimensional vector
            reward  :   Integer
            done    :   Boolean
            info    :   Dict {'map': map, 'currs': currs}
        Description:
            Perform the action first and then determine
            state, reward, done and info
        '''
        currs = []
        state = []
        reward = 0
        done = self.check_completion()
        #Perform action for each agent
        add_reward = 0
        for i in range(self.M):
            #Perform actions only if not completed
            if not done:
                self.agents[i].perform_action(action[i])
                if(action[i]==0):
                    add_reward -= 1
                elif(self.goal_weights[action[i]-1]==0):
                     add_reward -= 4
                elif(self.prev_action[i]!=action[i] and self.prev_action[i]!=-1):
                    add_reward -= 2
                self.prev_action[i] = action[i]
            curr = self.agents[i].curr
            currs.append(curr)
            #Update state for each agent
            self.compute_state(i)
            state.append(self.agents[i].state)
        info = {'map': self.map, 'currs': np.array(currs)}
        self.curr_t += 1
        reward = self.compute_reward() + add_reward
        return np.array(state), reward, done, info

    def check_completion(self):
        '''
        Description:
            Checks if the session has ended
        '''
        #Episode completion
        if self.curr_t > self.t_lim: return True
        #Goals done
        if np.sum(self.goal_weights) == 0: return True
        return False

    def compute_reward(self):
        '''
        Description:
            Calculates the reward for the current time step
        '''
        #Timestep reward
        reward_t = -1
        #Check the goal and reduce weight
        reward_g = 0
        for i in range(self.M):
            curr = self.agents[i].curr
            #Find which goal the agent is at
            indexes = np.sum(self.goal_locs == self.agents[i].curr, axis=1)
            if 2 in indexes:
                index = np.argwhere(indexes == 2)
                #Decrease weight and increase reward
                if self.goal_weights[index] != 0:
                    self.goal_weights[index] -= 1
                    reward_g += 1*self.goal_r_multiplier
        reward = reward_g + reward_t
        return reward

    def reset(self, num_tunnels=500):
        '''
        Arguments:
            num_tunnels :   Integer, defines the density of the map
        Description:
            Generates new map and goals, recomputes bfs
        '''
        self.curr_t = 0
        #New map
        self.map = self.generate_map(num_tunnels=num_tunnels)
        #Set agents start location
        self.start_locs = np.linspace([0,0], [self.N-1,0], self.M, dtype=np.uint16)
        #Set goals
        self.goal_locs, self.goal_weights = self.generate_goals()
        self.min_goal = np.min(self.goal_weights)
        self.max_goal = np.max(self.goal_weights)
        #Create M agents
        self.agents = [agent_class(self.start_locs.copy()[i], self.K) for i in range(self.M)]
        #Perform initial BFS for each goal
        self.bfs_goals = []
        for goal in self.goal_locs:
            self.bfs_goals.append(bfs(self.map, point(goal)))
        #Perform bfs for the initial positions
        self.bfs_inits = []
        for init in self.start_locs:
            self.bfs_inits.append(bfs(self.map, point(init)))
        #Update the initial state for every agent
        states = []
        for i in range(self.M):
            self.compute_state(i)
            states.append(self.agents[i].state)
        return np.array(states)

    def render(self, mode='human', close=False):
        '''
        Returns:
            rendered map: NxN int array
        '''
        render_map = self.map.copy()
        #Set agents and start locs
        for i in range(self.M):
            render_map[tuple(self.agents[i].curr)] = 1
            # render_map[tuple(self.start_locs[i])] = 4
        ranges = np.linspace(self.min_goal, self.max_goal, 4, dtype=np.int8)
        for i in range(self.K):
            if self.goal_weights[i] in range(1,ranges[1]+1):
                render_map[tuple(self.goal_locs[i])] = 2
            elif self.goal_weights[i] in range(ranges[1],ranges[2]+1):
                render_map[tuple(self.goal_locs[i])] = 3
            elif self.goal_weights[i] in range(ranges[2],ranges[3]+1):
                render_map[tuple(self.goal_locs[i])] = 4
        return render_map

    def compute_state(self, agent_num):
        '''
        Arguments:
            agent_num   :   Integer
        Description:
            Update agent's state and steps using bfs data
        '''
        agent = self.agents[agent_num]
        curr_loc = point(agent.curr)
        agent.state = []
        agent.steps = ['O'] #0th is stop
        #Get the goals data
        for goal_data in self.bfs_goals:
            agent.state.append(goal_data[curr_loc.r][curr_loc.c].dis)
            agent.steps.append(goal_data[curr_loc.r][curr_loc.c].first)

    def generate_map(self, num_tunnels=500, max_length=8):
        '''
        Arguments:
            num_tunnels :   Integer
            max_length  :   Integer
        Returns:
            new_map     :   NxN numpy array
        Description:
            Generate a random map using improved random walk algorithm
            num_tunnels controls how dense the floors are
            max_length controls how 'tunnely' it is
        '''
        new_map = np.ones((self.N, self.N))*5
        #Agents starting positions
        new_map[:,0] = 0
        #Random start among the starting positions
        curr = [np.random.randint(self.N, size=1)[0],0]
        directions = [[-1,0],[0,-1],[1,0],[0,1]] #U,L,D,R
        last_index = 0
        while num_tunnels > 0:
            #Select direction
            while 1:
                dir_index = np.random.randint(4)
                dir = np.array(directions[dir_index])
                #If same or reversed-new direction again
                if abs(dir_index-last_index)%2: break
            #Select random tunnel length
            random_length = np.random.randint(max_length)
            length = 0
            while length < random_length:
                #Update
                curr += dir
                #Fuck go back!
                if ((curr>self.N-1).any()) or ((curr<0).any()):
                    curr -= dir
                    break
                new_map[tuple(curr)] = 0
                length += 1
            last_index = dir_index
            #Won't be a tunnel if length is 0
            if length: num_tunnels -= 1
        return new_map

    def generate_goals(self, max_weight=20, div=10):
        '''
        Arguments:
            max_weight  :   Integer, the maximum possible weight
            div         :   Integer, the maximum deviation from max
        Returns:
            goal_locs   :   Kx2 numpy array
        '''
        goal_locs = np.zeros((self.K,2), dtype=np.uint16)
        goal_weights = np.zeros((self.K), dtype=np.uint16)
        #For each of K goals
        for k in range(self.K):
            while 1:
                goal = np.random.randint(self.N, size=2)
                #If goal location is free
                if (self.map[tuple(goal)] == 0) and (goal not in goal_locs): break
            goal_locs[k] = goal
            #Create goal weights
            goal_weights[k] = np.random.randint(div)
        #Sort the goal weights
        goal_weights = max_weight - np.sort(goal_weights)
        return goal_locs, goal_weights