#################################
# Your name: Jonathan Yahav
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        rng = np.random.default_rng()
        x_values = rng.uniform(0.0, 1.0, m)
        y_values = np.array([generate_label(x) for x in x_values])
        return np.vstack((x_values, y_values))


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        # TODO: Implement the loop
        pass

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # TODO: Implement the loop
        pass


    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me
        pass

#################################
# Place for additional methods


def generate_label(x):
    rng = np.random.default_rng()
    random = rng.uniform(0.0, 1.0) # use a random number for probabilistic mapping
    return int(random <= 0.8) if is_in_positive_interval(x) else int(random <= 0.1)


def is_in_positive_interval(x):
    positive_intervals = [0 <= x <= 0.2, 0.4 <= x <= 0.6, 0.8 <= x <= 1]
    return any(positive_intervals)

def compute_true_error(interval_list):
    positive_intervals = [[0, 0.2], [0.4, 0.6], [0.8, 1]]
    negative_intervals = [[0.2, 0.4], [0.6, 0.8]]
    positive_overlap = compute_overlap_length(interval_list, positive_intervals)
    negative_overlap = compute_overlap_length(interval_list, negative_intervals)
    positive_no_overlap, negative_no_overlap = 0.6 - positive_overlap, 0.4 - negative_overlap
    return 0.2 * positive_overlap + 0.1 * negative_no_overlap + 0.8 * positive_no_overlap + 0.9 * negative_overlap


def compute_overlap_length(lst1, lst2):
    i, j, overlap_length = 0, 0, 0
    while i < len(lst1) and j < len(lst2):
        left1, left2, right1, right2 = lst1[i][0], lst2[j][0], lst1[i][1], lst2[j][1]
        low, high = max(left1, left2), min(right1, right2)
        if low <= high:
            overlap_length += high - low
        i += right1 <= right2
        j += right1 >= right2
    return overlap_length

#################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.cross_validation(1500)

