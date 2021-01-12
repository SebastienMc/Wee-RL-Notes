import numpy as np


def run_episode():
    actions = [-1, 1]  # Left, right - uniform random policy
    action_sequence = np.array([])

    while True:

        # Draw 30 random actions
        seq = np.random.choice(actions, size=30, replace=True)
        action_sequence = np.concatenate([action_sequence, seq], axis=0)
        cumsum = action_sequence.cumsum()

        # Cumsum == 3 means right-hand terminal reached
        right_terminal = np.where(cumsum == 3, True, 0)
        end_right_arr = np.argwhere(right_terminal)

        # Cumsum == -3 means left-hand terminal reached
        left_terminal = np.where(cumsum == -3, True, 0)
        end_left_arr = np.argwhere(left_terminal)

        # Check whether either terminal state has been reached
        reached_right = end_right_arr.size > 0
        reached_left = end_left_arr.size > 0

        if reached_right or reached_left:
            break
        else:
            continue

    # Which terminal state?
    if reached_right and reached_left:
        end_right = end_right_arr[0, 0] < end_left_arr[0, 0]

    elif reached_right:
        end_right = True

    else:
        end_right = False

    # Allocate reward
    if end_right:
        end_idx = end_right_arr[0, 0] + 1
        sequence = action_sequence[:end_idx]
        reward = np.zeros_like(sequence)
        reward[-1] = 1

    else:
        end_idx = end_left_arr[0, 0] + 1
        sequence = action_sequence[:end_idx]
        reward = np.zeros_like(sequence)

    states = np.array([3.])  # Start state
    states = np.concatenate([states, sequence.cumsum() + 3], axis=0)
    states = states[:-1]

    return states, sequence, reward


class Agent:

    true_state_values = [0.167, 0.333, 0.5, 0.667, 0.833]

    def __init__(self, gamma=1., alpha=1.):
        state_values = np.zeros(7)  # Initialisation
        state_values[1:-1] = 0.5

        self.state_values = state_values
        self.N = np.zeros(7)
        self.gamma = gamma
        self.alpha = alpha
        self.rmse_log = []

    def log_rmse(self):
        error = self.rmse()
        self.rmse_log.append(error)

    def rmse(self):

        state_values = np.array(self.state_values[1:-1])
        sq_error = (self.true_state_values - state_values) ** 2

        return np.mean(np.sqrt(sq_error))

    def update_state_values(self, episode):
        pass


class MonteCarlo(Agent):

    def __init__(self, gamma=1., alpha=1., first_visit=True):
        super().__init__(gamma, alpha)
        self.first_visit = first_visit

    def update_state_values(self, episode):

        states, actions, rewards = episode
        returns = self.get_returns(episode)
        iterable = tuple(zip(states, returns))

        # only relevant if first_visit=True
        unique = np.unique(states).tolist()
        seen = []

        for i in iterable:
            state, est_return = i

            try:
                assert state not in seen

            except AssertionError:
                continue

            else:
                if self.first_visit:
                    seen.append(state)

            finally:
                idx = int(state)

                self.N[idx] += 1
                step_size = 1 / self.N[idx]

                error = est_return - self.state_values[idx]
                self.state_values[idx] += self.alpha * error

                if self.first_visit and unique == sorted(seen):
                    break

    def get_returns(self, episode):
        returns = []
        s, a, r = episode

        for step in range(len(s)):
            idx = step + 1
            gammas = np.array([self.gamma] * idx)
            powers = np.array([x for x in range(idx)])
            discounted_gammas = gammas ** powers
            rewards = np.array(r[-idx:])
            returns.append(np.sum(discounted_gammas * rewards))

        return returns


class TemporalDifference(Agent):

    def update_state_values(self, episode):

        n_steps = len(episode[1])

        for step in range(n_steps):
            s, a, r = tuple(map(lambda x: x[step], episode))
            idx = int(s)
            next_s_idx = int(idx + a)
            next_s_value = self.state_values[next_s_idx]

            td_error = r + self.gamma * next_s_value - self.state_values[idx]
            self.state_values[idx] += self.alpha * td_error

