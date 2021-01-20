import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np


class WindyGridWorld:

    def __init__(self, stochastic_wind=False):

        # Default game parameters
        self.grid_h = 7
        self.grid_w = 10
        self.start_loc = (self.grid_h // 2, 0)
        self.goal_loc = (self.grid_h // 2, self.grid_w - 3)
        self.very_windy_cols = range(6, 8)
        self.windy_cols = range(3, 9)
        self.stochastic = stochastic_wind
        self.rand_generator = np.random.RandomState(1)

        # 'Live' game parameters
        self.agent_loc = None
        self.reward_state_term = [None] * 3

    def state(self, loc):
        """The state is the one dimensional state representation
        of the agent location.

        Returns:
            one-dimensional state representation
        """
        grid = np.arange(self.grid_h * self.grid_w)
        grid = grid.reshape(self.grid_h, self.grid_w)
        return grid[loc]

    def env_start(self):
        """The first method called when the episode starts, called
        before the agent starts.

        Returns:
            The first state from the environment.
        """
        reward = 0
        self.agent_loc = self.start_loc
        state = self.state(self.agent_loc)
        termination = False
        self.reward_state_term = (reward, state, termination)

        return self.reward_state_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state,
                                     and boolean indicating if it's terminal.
        """

        x, y = self.agent_loc

        # Agent actions
        if action == 0:  # UP
            x -= 1
        elif action == 1:  # LEFT
            y -= 1
        elif action == 2:  # DOWN
            x += 1
        else:  # RIGHT
            y += 1

        # Wind effect
        if y in self.windy_cols:

            if self.stochastic:

                if y in self.very_windy_cols:
                    wind_values = range(1, 4)

                else:
                    wind_values = range(3)

                wind_effect = self.rand_generator.choice(wind_values)

            else:

                if y in self.very_windy_cols:
                    wind_effect = 2

                else:
                    wind_effect = 1

            x -= wind_effect  # "Southerly" wind pushing agent UP

        if not self.is_in_bounds(x, y):
            x = np.clip(x, 0, self.grid_h - 1)
            y = np.clip(y, 0, self.grid_w - 1)

        self.agent_loc = (x, y)

        reward = -1
        terminal = True if self.agent_loc == self.goal_loc else False

        self.reward_state_term = (reward, self.state(self.agent_loc), terminal)
        return self.reward_state_term

    def is_in_bounds(self, x, y):
        row_bool = self.grid_h > x >= 0
        col_bool = self.grid_w > y >= 0
        return row_bool and col_bool


class TDAgent:

    def __init__(self, agent_info={}):
        # Default agent parameters
        self.rand_generator = np.random.RandomState(1)
        self.policy = agent_info.get("policy")
        self.epsilon = agent_info.get("epsilon")
        self.discount = agent_info.get("discount")
        self.step_size = agent_info.get("step_size")
        self.values = np.zeros(self.policy.shape)

        # 'Live' agent parameters
        self.last_state = None
        self.last_action = None

    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start
                                 function.
        Returns:
            The first action the agent takes.
        """
        # The policy can be represented as a (# States, # Actions) array.
        # So, we can use the second dimension here when choosing an action.
        action = self.rand_generator.choice(range(self.policy.shape[1]),
                                            p=self.policy[state])
        self.last_state = state
        self.last_action = action
        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                            terminal state.
        """
        q = self.values[self.last_state, self.last_action]
        q_prime = 0
        td_error = reward + self.discount * q_prime - q
        self.values[
            self.last_state, self.last_action] += self.step_size * td_error

    def agent_update_policy(self):

        q_values = self.values[self.last_state]
        greedy_action = self.argmax(q_values)

        n_actions = self.policy.shape[1]
        new_state_policy = np.ones(n_actions) * self.epsilon / n_actions
        new_state_policy[greedy_action] += 1 - self.epsilon

        self.policy[self.last_state] = new_state_policy

    def argmax(self, array):
        """
        Takes in an array or a list and returns the index of the item
        with the highest value. Breaks ties randomly.
        returns: int - the index of the highest value in array
        """
        top_value = float("-inf")
        ties = []

        for i in range(len(array)):
            value = array[i]

            if value > top_value:
                top_value = value
                ties = [i]

            elif value == top_value:
                ties.append(i)

            else:
                continue

        return self.rand_generator.choice(ties)

    def plot_grid(self, height, width):

        # Prepare grid
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        major_y_ticks = np.arange(0, height + 1)
        major_x_ticks = np.arange(0, width + 1)

        ax.set_xticks(major_x_ticks)
        ax.set_yticks(major_y_ticks)

        ax.grid(which='major')
        ax.grid(which='major', alpha=0.7)

        # Highlight start and end squares
        patches, colours = [], []
        start1 = np.array([[0, 3], [1, 3], [0, 4]])
        start2 = np.array([[1, 4], [1, 3], [0, 4]])
        end1 = np.array([[8, 3], [7, 4], [8, 4]])
        end2 = np.array([[8, 3], [7, 4], [7, 3]])
        coords = (start1, start2, end1, end2)

        for c, coord in enumerate(coords):
            polygon = Polygon(coord)
            patches.append(polygon)
            colour = c // 2
            colours.append(colour)

        p = PatchCollection(patches, alpha=0.4)
        p.set_array(np.array(colours))
        ax.add_collection(p)

        for c, i in enumerate(self.policy):
            if not all(i == 0.25):  # state has been visited
                row = 6.5 - (c // 10)
                col = 0.5 + (c % 10)
                coords = [col, row, col, row]

                argmax = self.argmax(i)
                if argmax == 0:
                    coords[1] += 0.45
                elif argmax == 1:
                    coords[0] -= 0.45
                elif argmax == 2:
                    coords[1] -= 0.45
                elif argmax == 3:
                    coords[0] += 0.45

                ax.annotate("",
                            xy=(coords[0], coords[1]), xycoords='data',
                            xytext=(coords[2], coords[3]), textcoords='data',
                            arrowprops=dict(arrowstyle="->",
                                            connectionstyle="arc3"))

            else:  # state not visited
                pass

        plt.tick_params(axis='x', which='both', bottom=False,
                        top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', right=False,
                        left=False, labelleft=False)


class Sarsa(TDAgent):

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step after
                                 the last action, i.e., where the agent ended up
                                 after the last action.
        Returns:
            The next action the agent is taking.
        """

        action = self.rand_generator.choice(range(self.policy.shape[1]),
                                            p=self.policy[state])
        last_state = self.last_state
        last_action = self.last_action

        q = self.values[last_state, last_action]
        q_prime = self.values[state, action]
        td_error = reward + self.discount * q_prime - q
        self.values[last_state, last_action] += self.step_size * td_error

        # derive policy from self.values (Q) (epsilon greedy)
        self.agent_update_policy()

        self.last_state = state
        self.last_action = action

        return action


class QLearner(TDAgent):

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step after
                                 the last action, i.e., where the agent ended up
                                 after the last action.
        Returns:
            The next action the agent is taking.
        """

        last_state = self.last_state
        last_action = self.last_action

        q = self.values[last_state, last_action]
        q_prime = np.max(self.values[state])
        td_error = reward + self.discount * q_prime - q
        self.values[last_state, last_action] += self.step_size * td_error

        # derive policy from self.values (Q) (epsilon greedy)
        self.agent_update_policy()

        action = self.rand_generator.choice(range(self.policy.shape[1]),
                                            p=self.policy[state])
        self.last_state = state
        self.last_action = action

        return action


class ExpectedSarsa(TDAgent):

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step after
                                 the last action, i.e., where the agent ended up
                                 after the last action.
        Returns:
            The next action the agent is taking.
        """

        last_state = self.last_state
        last_action = self.last_action

        q = self.values[last_state, last_action]
        q_prime = np.sum(self.values[state] * self.policy[state])
        td_error = reward + self.discount * q_prime - q
        self.values[last_state, last_action] += self.step_size * td_error

        # derive policy from self.values (Q) (epsilon greedy)
        self.agent_update_policy()

        action = self.rand_generator.choice(range(self.policy.shape[1]),
                                            p=self.policy[state])
        self.last_state = state
        self.last_action = action

        return action

