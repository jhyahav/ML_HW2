#################################
# Your name: Jonathan Yahav
#################################
import numpy
import numpy as np
import matplotlib.pyplot as plt
import intervals as intv


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
        x_values.sort()
        y_values = np.array([self.generate_label(x) for x in x_values])
        return np.column_stack((x_values, y_values))


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
        x_axis = np.arange(m_first, m_last + step, step)
        empirical_errors, true_errors = [], []
        for n in range(m_first, m_last + step, step):
            empirical_avg, true_avg = 0, 0
            for i in range(T):
                sample = self.sample_from_D(n)
                intervals, besterror = intv.find_best_interval(sample[:, 0], sample[:, 1], k)
                empirical_avg += besterror / (n * T)
                true_avg += self.compute_true_error(intervals) / T
            empirical_errors.append(empirical_avg)
            true_errors.append(true_avg)
        title = "Average Error as a Function of n"
        [empirical, true] = [np.array(empirical_errors), np.array(true_errors)]
        self.graph(title, x_axis, [empirical, true], "n", ["Empirical Error", "True Error"])
        return np.column_stack((empirical, true))

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        x_axis = np.arange(k_first, k_last + step, step)
        sample = self.sample_from_D(m)
        empirical_errors, true_errors = [], []
        min_empirical_error, best_k = 2, -1
        for k in range(k_first, k_last + step, step):
            intervals, besterror = intv.find_best_interval(sample[:, 0], sample[:, 1], k)
            empirical_error = besterror / m
            if empirical_error < min_empirical_error:
                min_empirical_error = empirical_error
                best_k = k
            empirical_errors.append(empirical_error)
            true_errors.append(self.compute_true_error(intervals))
        [empirical, true] = [np.array(empirical_errors), np.array(true_errors)]
        title = "Error as a Function of k"
        self.graph(title, x_axis, [empirical, true], "k", ["Empirical Error", "True Error"])
        return best_k

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        x_axis = np.arange(1, 11, 1)
        empirical_errors = []
        min_empirical_error, best_k, best_intervals = 2, -1, None
        split_point = int(0.8 * m)
        for k in range(1, 11):
            sample = self.sample_from_D(m)
            np.random.shuffle(sample)
            training_set, holdout_set = sample[:split_point], sample[split_point:]
            training_set = training_set[np.argsort(training_set[:, 0])]
            intervals, training_error = intv.find_best_interval(training_set[:, 0], training_set[:, 1], k)
            empirical_error = self.compute_empirical_error(intervals, holdout_set)
            if empirical_error < min_empirical_error:
                min_empirical_error = empirical_error
                best_k = k
                best_intervals = intervals
            empirical_errors.append(empirical_error)
        # print(best_intervals)
        empirical = np.array(empirical_errors)
        title = "Error as a Function of k with Holdout Validation"
        self.graph(title, x_axis, [empirical], "k", ["Empirical Error"])
        return best_k

    #################################
    # Place for additional methods

    positive_intervals = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
    negative_intervals = [(0.2, 0.4), (0.6, 0.8)]

    def generate_label(self, x):
        rng = np.random.default_rng()
        random = rng.uniform(0.0, 1.0) # use a random number for probabilistic mapping
        return int(random <= 0.8) if self.is_in_positive_interval(x) else int(random <= 0.1)

    def compute_true_error(self, interval_list):
        positive_overlap = self.compute_overlap_length(interval_list, self.positive_intervals)
        negative_overlap = self.compute_overlap_length(interval_list, self.negative_intervals)
        positive_no_overlap, negative_no_overlap = 0.6 - positive_overlap, 0.4 - negative_overlap
        return 0.2 * positive_overlap + 0.1 * negative_no_overlap + 0.8 * positive_no_overlap + 0.9 * negative_overlap

    @staticmethod
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

    def compute_empirical_error(self, hypothesis_intervals, data):
        mislabeled_count = 0
        length = np.size(data, 0)
        for i in range(length):
            mislabeled_count += (self.is_in_interval(data[i, 0], hypothesis_intervals) != bool(data[i, 1]))
        return mislabeled_count / length

    def is_in_positive_interval(self, x):
        return self.is_in_interval(x, self.positive_intervals)

    @staticmethod
    def is_in_interval(x, interval_lst):
        for interval in interval_lst:
            if interval[0] <= x <= interval[1]:
                return True
        return False

    @staticmethod
    def graph(title, xData, yData, xLabel, yLabels):
        plt.xlabel(xLabel)
        colors = ["red", "black"]
        for i in range(len(yData)):
            plt.plot(xData, yData[i], label=yLabels[i], color=colors[i])
        plt.legend()
        plt.title(title)
        plt.show()

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.cross_validation(1500)

