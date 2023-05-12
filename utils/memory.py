# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Kai Arulkumaran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# ==============================================================================
from __future__ import division

import numpy as np

import torch

from collections import namedtuple

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
blank_trans = Transition(0, torch.zeros(84, 84, dtype=torch.uint8), None, 0, False)


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        # Initialize fixed size tree with all (priority) zeros
        self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)
        self.data = np.array([None] * size)  # Wrap-around cyclic buffer
        self.max = 1  # Initial max value to return (1 = 1^w)

    # Propagates value up tree given a tree index
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]

        if parent != 0:
            self._propagate(parent, value)

    # Updates value given a tree index
    def update(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate(index, value)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2

        if left >= len(self.sum_tree):
            return index

        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)

        else:
            return self._retrieve(right, value - self.sum_tree[left])

    # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        index = self._retrieve(0, value)  # Search for index of item from root
        data_index = index - self.size + 1

        return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

    # Return data given a data index
    def get(self, data_index):

        return self.data[data_index % self.size]

    def total(self):

        return self.sum_tree[0]


class ReplayMemory():
    def __init__(self, args, capacity):
        self.args = args
        self.capacity = capacity
        self.history = args.history_length
        # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_weight = args.priority_weight
        self.t = 0  # Internal episode timestep counter
        # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
        self.transitions = SegmentTree(capacity)

    # Add state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, terminal):
        # Only store last frame and discretize to save memory
        state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))
        # Store new transition with maximum priority
        self.transitions.append(Transition(self.t, state, action, reward, not terminal), self.transitions.max)
        self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

    # Returns a transition with blank states where appropriate
    def _get_transition(self, idx):
        transition = np.array([None] * (self.history + self.args.multi_step))
        transition[self.history - 1] = self.transitions.get(idx)

        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0

            if transition[t + 1].timestep == 0:
                transition[t] = blank_trans  # If future frame has timestep 0

            else:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)

        for t in range(self.history, self.history + self.args.multi_step):  # e.g. 4 5 6

            if transition[t - 1].nonterminal:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)

            else:
                transition[t] = blank_trans  # If prev (next) frame is terminal

        return transition

    # Returns a valid sample from a segment
    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            # Uniformly sample an element from within a segment
            sample = np.random.uniform(i * segment, (i + 1) * segment)
            # Retrieve sample from tree with un-normalized probability
            prob, idx, tree_idx = self.transitions.find(sample)

            # Resample if transition straddled current index or probability 0
            if (self.transitions.index - idx) % self.capacity > self.args.multi_step and \
               (idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx)

        # Create un-discretized state and nth next state
        state = torch.stack(
            [trans.state for trans in transition[:self.history]]
        ).to(device=self.args.cuda).to(dtype=torch.float32).div_(255)
        next_state = torch.stack(
            [trans.state for trans in transition[self.args.multi_step:self.args.multi_step + self.history]]
        ).to(device=self.args.cuda).to(dtype=torch.float32).div_(255)

        # Discrete action to be used as index
        action = torch.tensor([transition[self.history - 1].action], dtype=torch.int64, device=self.args.cuda)

        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1
        # (note that invalid nth next states have reward 0)
        R = torch.tensor([sum(self.args.gamma ** n * transition[self.history + n - 1].reward
                              for n in range(self.args.multi_step))],
                         dtype=torch.float32, device=self.args.cuda)

        # Mask for non-terminal nth next states
        nonterminal = torch.tensor([transition[self.history + self.args.multi_step - 1].nonterminal],
                                   dtype=torch.float32, device=self.args.cuda)

        return prob, idx, tree_idx, state, action, R, next_state, nonterminal

    def sample(self, batch_size):
        # Retrieve sum of all priorities (used to create a normalized probability distribution)
        p_total = self.transitions.total()

        # Batch size number of segments, based on sum over all probabilities
        segment = p_total / batch_size

        # Get batch of valid samples
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
        states, next_states = torch.stack(states), torch.stack(next_states)
        actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)

        # Calculate normalized probabilities
        probs = np.array(probs, dtype=np.float32) / p_total

        capacity = self.capacity if self.transitions.full else self.transitions.index

        # Compute importance-sampling weights w
        weights = (capacity * probs) ** -self.priority_weight

        # Normalize by max importance-sampling weight from batch
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.args.cuda)

        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.args.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0

        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration

        # Create stack of states
        state_stack = [None] * self.history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep

        for t in reversed(range(self.history - 1)):

            if prev_timestep == 0:
                state_stack[t] = blank_trans.state  # If future frame has timestep 0

            else:
                state_stack[t] = self.transitions.data[self.current_idx + t - self.history + 1].state
                prev_timestep -= 1

        # Agent will turn into batch
        state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.args.cuda).div_(255)
        self.current_idx += 1

        return state

    next = __next__  # Alias __next__ for Python 2 compatibility
