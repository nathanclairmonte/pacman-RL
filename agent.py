import sys
import time
import math
import itertools
from random import randrange
from ale_py import ALEInterface
from IPython.display import HTML
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import pytz
import copy
import pandas as pd

# get current time
def getTime(timezone="Canada/Eastern"):
    """
    Creates a 'now' object containing information about the current time.
    Used for logging/saving files.

    Returns:
        now (utc timezone object): Object representing the current date & time
    """
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    now = utc_now.astimezone(pytz.timezone(timezone))
    return now

# function to help with time logging
def stringTime(t_seconds, show_ms=False):
    """
    Takes a given number of seconds and formats it into a string

    Parameters:
        t_seconds (float): Amount of time in seconds
        show_ms (bool): Whether or not to append milliseconds to the string

    Returns:
        t (str): String representing the amount of time, in hours/minutes/seconds/milliseconds
    """
    h = "{0:.0f}".format(t_seconds//3600)
    m = "{0:.0f}".format((t_seconds%3600)//60)
    s = "{0:.0f}".format(math.floor((t_seconds%3600)%60))
    ms = "{0:.2f}".format(((t_seconds%3600)%60 - math.floor((t_seconds%3600)%60))*1000) # remember s = math.floor((t_seconds%3600)%60
    h_str = f"{h} hour{'' if float(h)==1 else 's'}"
    m_str = f"{'' if float(h)==0 else ', '}{m} minute{'' if float(m)==1 else 's'}"
    s_str = f"{'' if (float(h)==0 and float(m)==0) else ', '}{s} second{'' if float(s)==1 else 's'}"
    ms_str = f"{'' if (float(h)==0 and float(m)==0 and float(s)==0) else ', '}{ms} ms"
    
    t = f"{h_str if float(h) != 0 else ''}{m_str if float(m) != 0 else ''}{s_str if float(s) != 0 else ''}{ms_str if show_ms else ''}"
    return t

# Random agent
# Performs a game with random actions, saves screens in process
# Saves both rgb and grayscale
def randomGame(ale, actions):
    """
    Plays a single game (episode), taking random actions at each step.
    The pixels of the screen are saved for each iteration of the while loop.
    Pixels are saved in greyscale and RGB format.

    Parameters:
        ale (ALE Object): The ALE object to be used for the game
        actions (list of int): The list of avilable actions that will be randomly chosen from

    Returns:
        total_reward (int): The final score of the game
        screens (list of np.array): The screens from the game (greyscale)
        screens_rgb (list of np.array): The screens from the game (RGB)
    """
    screens_rgb = []
    screens = []
    total_reward = 0
    while not ale.game_over():
        # get screen data
        (screen_width, screen_height) = ale.getScreenDims()
        screen_data_rgb = np.zeros((screen_width,screen_height,3),dtype=np.uint8)
        screen_data = np.zeros((screen_width,screen_height),dtype=np.uint8)
        ale.getScreenRGB(screen_data_rgb)
        ale.getScreen(screen_data)
        
        # append screen data to screens array
        screens.append(screen_data)
        screens_rgb.append(screen_data_rgb)
        
        # apply a random action
        a = actions[randrange(len(actions))]
        reward = ale.act(a)
        total_reward += reward
    print('Ended with score: %d' % (total_reward))
    ale.reset_game()
    return total_reward, screens, screens_rgb

# Display game screen
def displaySingleScreen(screen, rgb=False):
    """
    Display the pixels from a single frame (in either greyscale or RGB format)

    Parameters:
        screen (np.array): The screen to be displayed (in greyscale or RGB)
        rgb (bool): Whether the supplied screen is RGB format or not
    """
    plt.figure(figsize=(10,10))
    if(rgb):
        plt.imshow(screen)
    else:
        plt.imshow(screen, cmap='gray')
    plt.show()
    
# Animate series of game screens
def animateScreens(screens_full, max_frames=100, rgb=False, display=False):
    """
    Uses matplotlib's FuncAnimation function to create a gif from a list of screens.
    To save space, the max # of frames for the gif is 100 by default, and the speed
    of the gif is increased to accommodate this (depending on the number of frames
    to be animated).

    Parameters:
        screens_full (list of np.array): The screens from a game (gresyscale or RGB)
        max_frames (int): Max number of frames for the gif (default is 100)
        rgb (bool): Whether the supplied screens are RGB format or not
        display (bool): When true, the gif will be displayed in an HTML element (only works in notebooks)
    """
    screen_size = ale.getScreenDims()
    speed_factor = int(np.ceil(len(screens_full)/max_frames))
    print(f'Speed factor: {speed_factor}x')
    screens = []
    for n in range(len(screens_full)):
        if(n%speed_factor==0):
            screens.append(screens_full[n])
    print(f'Length of sped up array: {len(screens)}')
    
    fig = plt.figure(figsize=(10,10))
    if (rgb):
        im = plt.imshow(screens[0])
    else:
        im = plt.imshow(screens[0], cmap='gray')
    
    
    def init():
        im.set_data(screens[0])
        return [im]
    
    def animate(i):
        im.set_array(screens[min(i, len(screens)-1)])
        return [im]
    
    animation = FuncAnimation(fig, animate, init_func=init, frames=min(max_frames, len(screens)))
    now = getTime()
    #gifName = './gifs/' + now.strftime("%Y%m%d-%H%M") + '.gif'
    gifName = f'./gifs/{now.strftime("%Y%m%d-%H%M%S")}.gif'
    animation.save(gifName, writer='imagemagick')
    if (display):
        display(HTML(f"<img src={gifName}>"))
    
def printEp(currEp):
    """
    Prints the info about an episode, given the corresponding currEp dict.

    Parameters:
        currEp (dict): Dict containing information about a given episode of training
    """
    num = str(currEp['ep_num'])
    frames = str(currEp['frames'])
    time_taken = currEp['time_taken']
    end_score = str(currEp['end_score'])
    print(f'------------ Episode #{num} ------------')
    print(f'# Frames:   {frames}')
    print(f'Time taken: {stringTime(time_taken, show_ms=True)}')
    print(f'End score:  {end_score}')
    print('')
    
def preprocess(screen):
    """
    Preprocess a single screen. The screen is cropped and then downsampled by 2.

    Parameters:
        screen (np.array): The screen to be processed

    Returns:
        (np.array): The processed screen
    """
    return copy.deepcopy(screen[1:176:2, ::2])

def preprocessAll(screens):
    """
    Preprocess all screens in a given list of screens.

    Parameters:
        screens (list of np.array): List of screens

    Returns:
        processedScreens (list of np.array): List of preprocessed screens
    """
    processedScreens = []
    for screen in screens:
        processedScreens.append(preprocess(screen))
    return processedScreens

def getPosition(screen, colour, resolution=5):
    """
    Given a screen and a colour value, finds the avg. coordinates of the 
    character that corresponds to that colour. If no colour match is found,
    the function will return a position in the bottom left of the map.

    Parameters:
        screen (np.array): The screen to search on
        colour (int): The colour value to search for
        resolution (int): The resolution of the coordinates returned (default is 5)

    Returns:
        (np.array): Array representing the [y, x] coordinates of the colour value specified.
    """
    pos = np.array(np.where(screen==colour))
    if(pos.shape[1]==0):
        # no matches found, set position to bottom left of map
        (y_len, x_len) = screen.shape
        pos = np.array([y_len, x_len])
    else:
        pos = pos.mean(axis=1).astype(int)
    return pos//resolution

def getState(screen, resolution=5):
    """
    Takes a screen (processed or not) and returns the state of the game (as defined in the report).

    Parameters:
        screen (np.array): The screen to extract state from
        resolution (int): The resolution of the coordinates returned (default is 5)

    Returns:
        state (tuple): Tuple containing the pacman y & x and nearest ghost y & x coordinates.
    """
    # define colours for game objects
    colours = {
        'pacman':[42],
        'ghosts':[38,70,88,184,4,102,82,182,90],
        'env':[144,74,68,24] # path,wall,cherries,lives
    }

    # get pacman's position
    pacmanPos = getPosition(screen, colours['pacman'], resolution)

    # get all ghost positions
    ghostPos = []
    for colour in colours['ghosts']:
        ghostPos.append(getPosition(screen, colour, resolution))
    ghostPos = np.asarray(ghostPos)

    # get all distances of ghosts from pacman
    ghostDists = pacmanPos - ghostPos

    # find nearest ghost to pacman
    ghostDistAbs = np.abs(ghostDists).sum(axis=1)
    nearest = ghostDists[ghostDistAbs.argmin()]

    # build state
    ng_y, ng_x = nearest
    pac_y, pac_x = pacmanPos
    state = (pac_y, pac_x, ng_y, ng_x)
    
    return state

def setupALE(romFile='mspacpan.BIN'):
    """
    Initialize the ALE, define settings and load given ROM file.

    Parameters:
        romFile (str): The ROM file to be loaded (default is pacman)

    Returns:
        ale (ALE Object): The initialized ALE object
    """

    # initialize ALE
    ale = ALEInterface()

    # set desired settings
    ale.setInt(b'random_seed', 123)
    ale.setInt(b'frame_skip', 10)
    ale.setBool(b'color_averaging', True)

    # load ROM file
    ale.loadROM(romFile)

    return ale

class QLearner:
    """
    Class to implement the Q-Learning algorithm.
    Keeps track of the Q-table and its values, chooses an action, and updates Q-values
    """
    def __init__(self, lr=0.1, discount=0.95, epsilon=0.03, q_table=None, actions=[0,2,3,4,5], screen_shape=(88,80), screen_res=5):
        """
        Initializes the QLearner class with the relevant parameters.

        Parameters:
            lr (float): The learning rate
            discount (float): The discount factor
            epsilon (float): Epsilon, for the degree of random exploration
            q_table (pd.DataFrame): A predefined Q-table, if necessary (default is None)
            actions (list of int): The list of available actions for the agent (default includes the noop action)
            screen_shape (tuple): The shape of the screens (default is the shape after preprocessing)
            screen_res (int): The resolution of coordinates to be used
        """

        # get screen size info
        pac_y_len, pac_x_len = np.asarray(screen_shape)//screen_res
        ng_y_len, ng_x_len = np.asarray(screen_shape)//screen_res
        pac_y_len += 1
        pac_x_len += 1
        ng_y_len += 1
        ng_x_len += 1

        # create q_table if it doesn't exist
        if(q_table == None):
            combos = np.asarray(list(itertools.product(range(pac_y_len), range(pac_x_len), range(-ng_y_len+1,ng_y_len), range(-ng_x_len+1,ng_x_len))))
            pac_y_range = combos[:, 0]
            pac_x_range = combos[:, 1]
            ng_y_range = combos[:, 2]
            ng_x_range = combos[:, 3]
            raw_data = {
                'pac_y': pac_y_range,
                'pac_x': pac_x_range,
                'ng_y': ng_y_range,
                'ng_x': ng_x_range
            }
            for action in actions:
                raw_data[str(action)] = np.ones(pac_y_len*pac_x_len*(2*ng_y_len-1)*(2*ng_x_len-1))
            q_table = pd.DataFrame(raw_data, columns=list(raw_data.keys()))
            q_table.set_index(['pac_y', 'pac_x', 'ng_y', 'ng_x'], drop=True, inplace=True)
        
        # initialize attributes
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.q_table = q_table
        self.actions = actions
    
    def get(self, state):
        """
        Returns the array of q_values across all actions in a given state.
        """
        return self.q_table.loc[state].values
    
    def getAction(self, state, decay=False):
        """
        Chooses an action to be taken. Incorporates random exploration based on the size of epsilon.

        Parameters:
            state (tuple): The current state
            decay (bool): Whether or not to implement epsilon decay

        Returns:
            actionIndex (int): The index of the chosen action (in the actions array)
        """
        if(np.random.random() < self.epsilon): # np.random.random() gives a random float on the range [0,1]
            # Randomly explore
            actionIndex = np.random.randint(0, len(self.actions))
        else:
            # Be greedy
            actionIndex = np.argmax(self.get(state))
        if(decay):
            self.epsilon = max(self.epsilon*0.98, 0.0005)
            # self.epsilon = max(self.epsilon*0.995, 0.0005)
        return actionIndex

    def update(self, state, stateNext, actionIndex, reward):
        """
        Updates the Q-table according to the Q-Learning update rule.

        Parameters:
            state (tuple): The current state
            stateNext (tuple): The next state
            actionIndex (int): The index of the chosen action
            reward (int): The reward of the chosen action

        Returns:
            (pd.DataFrame): The updated q_table. It doesn't need to be returned, but I return it here
                            so I can keep track of Q-tables of high-performing episodes during training.
        """
        # Get the current q value
        current_q = self.get(state)[actionIndex]

        # Get the max future q value
        max_future_q = np.max(self.get(stateNext))

        # Update the current q value
        new_q = current_q + self.lr*(reward + self.discount*max_future_q - current_q)
        self.q_table.loc[state, str(self.actions[actionIndex])] = new_q

        return self.q_table
    
    def gameStep(self, ale, screen, ram, gameover, screens):
        # get number of lives
        livesBeforeAction = ale.lives()
        
        # get current state
        ale.getScreen(screen)
        screens.append(copy.deepcopy(screen))
        screen_proc = preprocess(screen)
        state = getState(screen_proc)
        
        # get action
        action = self.getAction(state)
        
# Testing function
def tester(ale, agent, numEpisodes=2, reward_factor=0.6, epsilon=0.5, actions=[2,3,4,5], decay=True, lr=0.2, verbose=False):
    """
    Trains a given agent for a given number of episodes, altering the q_value given to each step
    to facilitate my specific definition of rewards.

    Parameters:
        ale (ALE Object): The ALE object to use for each episode
        agent (class): Which learner to use for training
        numEpisodes (int): The number of episodes to train for
        reward_factor (float): A factor used to discount rewards. Chosen experimentally.
        epislon (float): Epsilon, used to define degree of random exploration
        actions (list of int): The actions available to the agent
        decay (bool): Whether or not to implement epsilon decay
        lr (float): The learning rate of the learner
        verbose (bool): Whether to print output after each episode

    Returns:
        all_episodes (list of dict): A list of all the info across the episodes run
        q_tables (list of pd.DataFrame): A list of all the final q_tables across the episodes run
    """
    agent = agent(epsilon=epsilon, actions=actions, lr=lr)
    all_episodes = []
    all_scores = []
    q_tables = []
    for ep in range(numEpisodes):
        # reset ale
        ale.reset_game()
        total_reward = 0
        
        # variables for screen/ram/gameover
        screen = np.zeros(ale.getScreenDims(),dtype=np.uint8)
        ram = np.zeros(ale.getRAMSize(),dtype=np.uint8)
        gameover = ale.game_over()
        
        screens = []
        t_start = time.time()
        while not gameover:
            # get current state
            ale.getScreen(screen)
            screens.append(copy.deepcopy(screen))
            screen_proc = preprocess(screen)
            state = getState(screen_proc)
            livesBeforeAction = ale.lives()
            
            # choose an action and take it
            actionIndex = agent.getAction(state, decay)
            reward = ale.act(actions[actionIndex])
            
            # wait for RAM condition to know when to act again
            ale.getRAM(ram)
            while(not (ram[0]==0 and ram[-1]&1==1)):
                # until ram condition is true, collect rewards from noops
                reward += ale.act(0) # noop
                ale.getRAM(ram)
            
            # ram condition true now, get new state 
            ale.getScreen(screen)
            screen_proc = preprocess(screen)
            nextState = getState(screen_proc)

            # multiplying by reward factor
            q = reward*reward_factor

            # heavily downweight losing a life
            livesAfterAction = ale.lives()
            q -= (livesBeforeAction-livesAfterAction)*1000

            # generalization step
            # difference = np.sum(np.abs(np.array(nextState)-np.array(state)))
            difference = np.sum(np.abs(np.array(nextState[:2])-np.array(state[:2])))
            q -= 1/max(1, difference)

            # update q_table and collect rewards
            q_table = agent.update(state, nextState, actionIndex, q)
            total_reward += reward
            gameover = ale.game_over()

        # when episode is over, collect info
        currEp = {
            'ep_num': ep,
            'frames': ale.getEpisodeFrameNumber(),
            'time_taken': time.time()-t_start,
            'end_score': total_reward
        }
        all_episodes.append(currEp)

        # if total_score is not our highest, don't save screens
        if(len(all_scores)>0):
            if(total_reward <= np.max(all_scores)):
                # delete screens from memory
                screens = None
        all_scores.append(total_reward)

        # save final q_table of this episode
        # q_tables.append(q_table)

        # logging
        if(verbose):
            print(currEp)

        # save screens after logging
        currEp['screens'] = screens
    
    return (all_episodes, q_tables)




# ----------------- Running a test -----------------
if(__name__=="__main__"):
    total_start = time.time()

    # params for test
    # EDIT THESE PARAMETERS TO CONFIGURE A TEST
    lr = 0.2
    reward_factor = 0.6
    numEpisodes = 500
    epsilon = 0.5
    decay = True
    actions = [2,3,4,5] # up, right, left, down

    # run test
    ale = setupALE('mspacman.BIN')
    (all_episodes, _) = tester(ale,
                            agent=QLearner,
                            numEpisodes=numEpisodes,
                            reward_factor=reward_factor,
                            epsilon=epsilon,
                            actions=actions,
                            decay=decay,
                            lr=lr,
                            verbose=False)
    print(f'Time elapsed: {stringTime(time.time()-total_start)}')
    print('')
    # displaySingleScreen(all_episodes[0]['screens'][0])
    # displaySingleScreen(all_episodes[0]['screens'][-1])

    # create dataframe of all episodes
    df = pd.DataFrame(all_episodes).drop(['screens'], axis=1).set_index(['ep_num'])

    # extract stats
    scores = np.asarray(df['end_score'])
    mean_score = np.mean(scores)
    bestEpisodeIndex = np.argmax([x['end_score'] for x in all_episodes])
    bestEpisode = all_episodes[bestEpisodeIndex]
    # animateScreens(bestEpisode['screens'], display=False)

    # Logging
    print(f'Mean end score overall: {mean_score:.2f}')
    print('Best episode: ', (bestEpisode['ep_num']+1), 'out of', numEpisodes)
    print('Best ep score: ', bestEpisode['end_score'])
    print('')
    n = 10 if numEpisodes>=10 else numEpisodes
    print(f'Last {n} scores')
    for i in range(n, 0, -1):
        print(f'{len(scores)-i+1}: {scores[-i]}')

    # plot scores across all episodes
    d_str = 'decay' if decay else 'no decay'
    plt.plot(scores)
    plt.title(f'{numEpisodes} Eps, Mean: {mean_score:.2f}, epsilon={epsilon}, lr={lr}, {d_str}')
    plt.xlabel('Episode #')
    plt.ylabel('End Score')
    plt.grid()
    now = getTime()
    save_path = f'./plots/{now.strftime("%Y%m%d-%H%M-%S")}.jpg'
    plt.savefig(save_path, format='jpg', dpi=500, bbox_inches='tight')
    plt.show()
