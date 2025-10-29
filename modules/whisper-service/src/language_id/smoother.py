#!/usr/bin/env python3
"""
LID Smoothing with HMM/Viterbi

Per FEEDBACK.md line 37: "Smooth with Viterbi or hysteresis"

This provides additional smoothing beyond the median filter in FrameLevelLID.
Uses Hidden Markov Model (HMM) with Viterbi decoding for optimal path finding.

Language switching has a cost (transition penalty), so the Viterbi algorithm
will prefer staying in the same language unless there's strong evidence for a switch.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SmoothedLIDResult:
    """Result of Viterbi smoothing"""
    language: str
    smoothed_probabilities: Dict[str, float]
    raw_probabilities: Dict[str, float]
    confidence: float
    transition_cost: float


class LIDSmoother:
    """
    HMM-based smoothing for language identification using Viterbi algorithm.

    Per FEEDBACK.md line 37: "Smooth with Viterbi or hysteresis"

    The HMM model has:
    - States: Languages (e.g., 'en', 'zh')
    - Observations: LID frame probabilities
    - Transition costs: Penalty for switching languages
    - Emission probabilities: LID frame probabilities

    Args:
        languages: List of target languages
        transition_cost: Cost for switching languages (0.0-1.0, default 0.3)
        window_size: Number of frames for Viterbi window (default 5)
    """

    def __init__(
        self,
        languages: List[str],
        transition_cost: float = 0.3,
        window_size: int = 5
    ):
        self.languages = languages
        self.transition_cost = transition_cost
        self.window_size = window_size

        # History for Viterbi smoothing
        self.probability_history: List[Dict[str, float]] = []
        self.smoothed_history: List[str] = []

        # Initialize transition matrix (uniform with self-transition preference)
        self.transition_matrix = self._init_transition_matrix()

        logger.info(
            f"LIDSmoother initialized: languages={languages}, "
            f"transition_cost={transition_cost}, window={window_size}"
        )

    def _init_transition_matrix(self) -> np.ndarray:
        """
        Initialize transition probability matrix.

        Diagonal (staying in same language): 1 - transition_cost
        Off-diagonal (switching): transition_cost / (num_languages - 1)

        Example for 2 languages with cost=0.3:
        [[0.7, 0.3],   # From en: stay=0.7, switch=0.3
         [0.3, 0.7]]   # From zh: switch=0.3, stay=0.7
        """
        n = len(self.languages)
        matrix = np.zeros((n, n))

        # Fill diagonal with staying probability
        np.fill_diagonal(matrix, 1.0 - self.transition_cost)

        # Fill off-diagonal with switching probability
        if n > 1:
            switch_prob = self.transition_cost / (n - 1)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        matrix[i, j] = switch_prob

        return matrix

    def smooth(
        self,
        lid_probs: Dict[str, float],
        timestamp: float
    ) -> SmoothedLIDResult:
        """
        Smooth LID probabilities using Viterbi algorithm.

        Args:
            lid_probs: Raw LID probabilities from FrameLevelLID
            timestamp: Current timestamp (not used, for future extensions)

        Returns:
            SmoothedLIDResult with Viterbi-smoothed language
        """
        # Add to history
        self.probability_history.append(lid_probs.copy())

        # Keep only window_size frames
        if len(self.probability_history) > self.window_size:
            self.probability_history.pop(0)

        # If we have insufficient history, return raw result
        if len(self.probability_history) < 2:
            top_lang = max(lid_probs, key=lid_probs.get)
            return SmoothedLIDResult(
                language=top_lang,
                smoothed_probabilities=lid_probs.copy(),
                raw_probabilities=lid_probs.copy(),
                confidence=lid_probs[top_lang],
                transition_cost=0.0
            )

        # Run Viterbi over window
        smoothed_path = self._viterbi_decode(self.probability_history)

        # Get current smoothed language (last in path)
        smoothed_lang = smoothed_path[-1]

        # Calculate transition cost
        trans_cost = 0.0
        if len(self.smoothed_history) > 0:
            prev_lang = self.smoothed_history[-1]
            if prev_lang != smoothed_lang:
                trans_cost = self.transition_cost

        # Update smoothed history
        self.smoothed_history.append(smoothed_lang)
        if len(self.smoothed_history) > self.window_size:
            self.smoothed_history.pop(0)

        # Create smoothed probabilities (boost selected language)
        smoothed_probs = lid_probs.copy()
        smoothed_probs[smoothed_lang] = min(
            1.0,
            smoothed_probs.get(smoothed_lang, 0.0) + 0.1
        )

        # Normalize
        total = sum(smoothed_probs.values())
        if total > 0:
            smoothed_probs = {k: v / total for k, v in smoothed_probs.items()}

        return SmoothedLIDResult(
            language=smoothed_lang,
            smoothed_probabilities=smoothed_probs,
            raw_probabilities=lid_probs.copy(),
            confidence=smoothed_probs[smoothed_lang],
            transition_cost=trans_cost
        )

    def _viterbi_decode(
        self,
        observations: List[Dict[str, float]]
    ) -> List[str]:
        """
        Viterbi algorithm to find most likely language sequence.

        Args:
            observations: List of LID probability dictionaries

        Returns:
            List of most likely languages (one per observation)
        """
        n_states = len(self.languages)
        n_obs = len(observations)

        # Convert observations to matrix (n_obs x n_states)
        obs_matrix = np.zeros((n_obs, n_states))
        for i, obs in enumerate(observations):
            for j, lang in enumerate(self.languages):
                obs_matrix[i, j] = obs.get(lang, 1e-10)  # Avoid log(0)

        # Log probabilities for numerical stability
        log_obs = np.log(obs_matrix + 1e-10)
        log_trans = np.log(self.transition_matrix + 1e-10)

        # Initialize Viterbi tables
        viterbi = np.zeros((n_obs, n_states))
        backpointer = np.zeros((n_obs, n_states), dtype=int)

        # Initialize first observation (uniform prior)
        viterbi[0] = log_obs[0] + np.log(1.0 / n_states)

        # Forward pass
        for t in range(1, n_obs):
            for s in range(n_states):
                # Calculate probability of each previous state + transition
                trans_probs = viterbi[t-1] + log_trans[:, s]

                # Best previous state
                backpointer[t, s] = np.argmax(trans_probs)

                # Viterbi value
                viterbi[t, s] = np.max(trans_probs) + log_obs[t, s]

        # Backward pass to find best path
        path = np.zeros(n_obs, dtype=int)
        path[-1] = np.argmax(viterbi[-1])

        for t in range(n_obs - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]

        # Convert state indices to language codes
        language_path = [self.languages[state_idx] for state_idx in path]

        return language_path

    def reset(self):
        """Reset smoother state"""
        self.probability_history.clear()
        self.smoothed_history.clear()
        logger.debug("LID smoother reset")

    def get_statistics(self) -> Dict:
        """Get smoothing statistics"""
        # Count transitions in smoothed history
        transitions = 0
        if len(self.smoothed_history) > 1:
            for i in range(1, len(self.smoothed_history)):
                if self.smoothed_history[i] != self.smoothed_history[i-1]:
                    transitions += 1

        return {
            'window_size': self.window_size,
            'transition_cost': self.transition_cost,
            'history_length': len(self.probability_history),
            'smoothed_transitions': transitions,
            'current_language': (
                self.smoothed_history[-1] if self.smoothed_history else None
            )
        }
