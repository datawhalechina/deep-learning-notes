from enum import IntEnum
from typing import NamedTuple

import numpy as np

__all__ = ['RiverSwimEnvr']


class Action(IntEnum):
    """An enumeration representing the possible actions in the RiverSwim environment."""

    LEFT = 0
    RIGHT = 1


class RiverCurrent(IntEnum):
    """An enumeration representing the strength of the river current."""

    WEAK = 1
    MEDIUM = 2
    STRONG = 3


class RiverSwimModel(NamedTuple):
    """A named tuple representing the RiverSwim MDP model."""

    T: np.ndarray  # Transition function
    R: np.ndarray  # Reward function


class RiverSwimResult(NamedTuple):
    """A named tuple representing the result of running an algorithm on the RiverSwim MDP."""

    policy: np.ndarray  # The optimal policy found
    val_func: np.ndarray  # The value function corresponding to the optimal policy


class RiverSwimProblem:
    """Defines the RiverSwim MDP environment for the CS234 assignment 1."""

    def __init__(self, current: RiverCurrent, seed: int = 1234):
        """Initialize the RiverSwim environment.

        Args:
            current (RiverCurrent): The strength of the river current.
            seed (int, default: 1234): Random seed for reproducibility.
        """
        self.num_states = 6
        self.num_actions = 2  # 0 <=> LEFT, 1 <=> RIGHT

        # Larger current makes it harder to swim up the river
        self.current = current.value

        # Configure reward function
        R = np.zeros((self.num_states, self.num_actions))
        R[0, 0] = 0.005
        R[5, 1] = 1.0

        # Configure transition function
        T = np.zeros((self.num_states, self.num_actions, self.num_states))

        # Encode initial and rewarding state transitions
        T[0, 0, 0] = 1.0
        T[0, 1, 0] = 0.6
        T[0, 1, 1] = 0.4

        T[5, 1, 5] = 0.6
        T[5, 1, 4] = 0.4
        T[5, 0, 4] = 1.0

        # Encode intermediate state transitions
        for s in range(1, self.num_states - 1):
            l, r = 0, 1

            # Going left always succeeds
            T[s, l, s - 1] = 1.0

            # Going right sometimes succeeds
            T[s, r, s] = 0.6
            T[s, r, s - 1] = 0.09 * self.current
            T[s, r, s + 1] = 0.4 - T[s, r, s - 1]

            # Make sure the transition probabilities sum to 1
            assert np.isclose(np.sum(T[s, l]), 1.0)
            assert np.isclose(np.sum(T[s, r]), 1.0)

        self.R = np.asarray(R)
        self.T = np.asarray(T)
        self.R.setflags(write=False)
        self.T.setflags(write=False)

        # Agent always starts at the opposite end of the river
        self.init_state = 0
        self.curr_state = self.init_state

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get_model(self) -> RiverSwimModel:
        """Return a copy of the model (transition and reward functions)."""
        return RiverSwimModel(self.T.copy(), self.R.copy())

    def reset(self):
        """Reset the environment to the initial state."""
        self.curr_state = self.init_state

    def step(self, a: Action) -> tuple[float, int]:
        """Take a step in the environment given an action."""
        # Get the reward for the current state and action
        reward = self.R[self.curr_state, a.value]

        # Choose the next state based on the transition probabilities
        p = self.T[self.curr_state, a.value]
        next_state = int(self.rng.choice(range(self.num_states), p=p))

        self.curr_state = next_state
        return reward


class RiverSwimEnvr(RiverSwimProblem):
    """A subclass of RiverSwim that provides a method to solve the MDP."""

    def bellman_backup(
        self, state: int, action: int, V: np.ndarray, gamma: float
    ) -> float:
        r"""Performs a Bellman backup for a given state and action.

        The Bellman backup is defined as:

        .. math::
            Q(s, a) = R(s, a) + \gamma \sum_{s'} T(s, a, s') V(s')

        Args:
            state (int): The current state.
            action (int): The action taken in the current state.
            V (np.ndarray): The value function of shape `(num_states,)`.
            gamma (float): The discount factor.

        Returns:
            Q (float): The updated value for the given state and action.
        """
        assert 0 <= gamma < 1, 'Discount factor must be in [0, 1).'
        return self.R[state, action] + gamma * np.dot(self.T[state, action], V)

    def policy_evaluation(
        self, policy: np.ndarray, gamma: float, tol: float = 1e-3
    ) -> np.ndarray:
        r"""Perform policy evaluation for a given **deterministic** policy.

        The value function for a policy is defined as:

        .. math::
            V(s) = R(s, \pi(s)) + \gamma \sum_{s'} T(s, \pi(s), s') V(s')

            V(s) = Q(s, \pi(s))

        Args:
            policy (np.ndarray): The policy to evaluate.
            gamma (float): The discount factor.
            tol (float): The tolerance for convergence.

        Returns:
            V (np.ndarray): The value function for the given policy.

        Example:
            >>> env = RiverSwimEnvr(Current.WEAK)
            >>> policy = np.array([Action.RIGHT] * env.num_states)
            >>> V = env.policy_evaluation(policy, gamma=0.9)
        """
        assert 0 <= gamma < 1, 'Discount factor must be in [0, 1).'
        old_V = np.zeros(self.num_states)

        while True:
            new_V = np.array(
                [
                    self.bellman_backup(state, policy[state], old_V, gamma)
                    for state in range(self.num_states)
                ]
            )

            if np.max(np.abs(new_V - old_V)) < tol:
                return new_V

            old_V = new_V

    def policy_improvement(self, V_policy: np.ndarray, gamma: float) -> np.ndarray:
        r"""Compute the improved policy given the value function of the current policy.

        .. math::
            \pi'(s) = \arg\max_a Q(s, a)

        Args:
            V_policy (np.ndarray): The value function of the current policy.
            gamma (float): The discount factor.

        Returns:
            new_policy (np.ndarray): The improved policy.

        Example:
            >>> env = RiverSwimEnvr(Current.WEAK)
            >>> policy = np.array([Action.RIGHT] * env.num_states)
            >>> V_policy = env.policy_evaluation(policy, gamma=0.9)
            >>> new_policy = env.policy_improvement(V_policy, gamma=0.9)
        """
        assert 0 <= gamma < 1, 'Discount factor must be in [0, 1).'
        new_policy = np.zeros(self.num_states, dtype=np.int8)

        for state in range(self.num_states):
            action_values = [
                self.bellman_backup(state, action, V_policy, gamma)
                for action in range(self.num_actions)
            ]
            new_policy[state] = np.argmax(action_values)

        return new_policy

    def policy_iteration(self, gamma: float, tol: float = 1e-3) -> RiverSwimResult:
        r"""Perform policy iteration to find the optimal policy for the RiverSwim MDP.
        The algorithm alternates between policy evaluation and policy improvement until
        convergence.

        Args:
            gamma (float): The discount factor.
            tol (float, default: 1e-3): The tolerance for convergence in policy evaluation.

        Returns:
            result (RiverSwimResult): The optimal policy found by policy iteration.

        Example:
            >>> env = RiverSwimEnvr(Current.WEAK)
            >>> result = env.policy_iteration(gamma=0.9)
        """
        assert 0 <= gamma < 1, 'Discount factor must be in [0, 1).'
        old_policy = np.zeros(self.num_states, dtype=np.int8)

        while True:
            V_policy = self.policy_evaluation(old_policy, gamma, tol)
            new_policy = self.policy_improvement(V_policy, gamma)

            if np.array_equal(new_policy, old_policy):
                break

            old_policy = new_policy

        return RiverSwimResult(policy=new_policy, val_func=V_policy)

    def value_iteration(self, gamma: float, tol: float = 1e-3) -> RiverSwimResult:
        r"""Perform value iteration to find the optimal policy for the RiverSwim MDP.
        The algorithm iteratively updates the value function until convergence, and then
        derives the optimal policy from the final value function.

        .. math::
            Q(s, a) = R(s, a) + \gamma \sum_{s'} T(s, a, s') V(s')

            V(s) = \max_a Q(s, a)

        Args:
            gamma (float): The discount factor.
            tol (float, default: 1e-3): The tolerance for convergence in value iteration.

        Returns:
            result (RiverSwimResult): The optimal policy found by value iteration.

        Example:
            >>> env = RiverSwimEnvr(Current.WEAK)
            >>> result = env.value_iteration(gamma=0.9)
        """
        assert 0 <= gamma < 1, 'Discount factor must be in [0, 1).'
        old_V = np.zeros(self.num_states)

        while True:
            new_V = np.array(
                [
                    max(
                        self.bellman_backup(state, action, old_V, gamma)
                        for action in range(self.num_actions)
                    )
                    for state in range(self.num_states)
                ]
            )

            if np.max(np.abs(new_V - old_V)) < tol:
                old_V = new_V
                break

            old_V = new_V

        # One more policy improvement step to get the optimal policy.
        policy = self.policy_improvement(new_V, gamma)
        return RiverSwimResult(policy=policy, val_func=new_V)


if __name__ == '__main__':
    seed = 1234
    current = RiverCurrent.WEAK

    env = RiverSwimEnvr(current, seed)
    discount_factor = 0.99

    print('\n' + '-' * 25 + '\nBeginning Policy Iteration\n' + '-' * 25)

    pi = env.policy_iteration(discount_factor, tol=1e-3)
    print(pi.val_func)
    print([['L', 'R'][a] for a in pi.policy])

    print('\n' + '-' * 25 + '\nBeginning Value Iteration\n' + '-' * 25)

    vi = env.value_iteration(discount_factor, tol=1e-3)
    print(vi.val_func)
    print([['L', 'R'][a] for a in vi.policy])
