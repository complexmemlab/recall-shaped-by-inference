import numpy as np


class FeatureRL:
    def __init__(self, eta, beta, st, data):
        self.eta = eta
        self.beta = beta
        self.st = st  # response stickiness
        self.data = data
        if "trial_number_per_run" in data.columns:
            self.trial_num_per_run = data["trial_number_per_run"].values
        self.W = np.zeros((2, 2))
        self.Ws = np.zeros((2, 2, self.data.shape[0]))
        self.Vs = np.zeros((self.data.shape[0]))
        self.n_trials = len(data)
        self.rpe = np.zeros((self.n_trials))
        self.loglik = 0
        self.trial_by_trial_loglik = np.zeros((self.n_trials))
        self.value_chosen = 0.5
        self.value_not_chosen = 0.5
        self.uncertainties = np.zeros((self.n_trials))
        self.P = np.array(
            [0.25, 0.25, 0.25, 0.25]
        )  # uniform distribution over potential targets

    def construct_stimulus(self):
        stim = np.array(
            (
                # self.data['word'].values,
                (
                    self.data["item_rule_idx0"].values,
                    self.data["item_rule_idx1"].values,
                ),
                (
                    self.data["inv_item_rule_idx0"].values,
                    self.data["inv_item_rule_idx1"].values,
                ),
            )
        )
        return stim

    def update_weights(self, stim_chosen_dim1, stim_chosen_dim2, reward, i):
        if self.trial_num_per_run[i] == 1:
            self.W = np.zeros((2, 2))
        else:
            self.W[0, stim_chosen_dim1] += self.eta * (reward - self.value_chosen)
            self.W[1, stim_chosen_dim2] += self.eta * (reward - self.value_chosen)

    def softmax(self, value_chosen, value_not_chosen):
        max_value = np.max([value_chosen, value_not_chosen])
        exp_value_chosen = np.exp(self.beta * (value_chosen - max_value))
        exp_value_not_chosen = np.exp(self.beta * (value_not_chosen - max_value))
        return exp_value_chosen / (exp_value_chosen + exp_value_not_chosen)

    def sim(self):  # TODO write a simulate function and then refactor what is in fit
        pass

    def uncertainty(self):
        """
        Takes in the weights, uses laplace smoothing, converts to 1x4 vector of probabilities and compares to
        uniform distribution to compute KL divergence
        """
        weights = self.W.flatten().copy()
        weights -= weights.min()
        weights += 1e-8
        weights /= weights.sum()
        return np.sum(self.P * np.log(self.P / weights))

    def uncertainty_shen(self, value_chosen, value_not_chosen):
        """
        Uses the decision uncertainty from Shen et al. 2022 $(1 - (P^highest - P^lowest))$
        Here we can just use value_chosen and value not_chosen because there are only 2 options unlike the 3 in their study
        """
        max_value = np.max([value_chosen, value_not_chosen])
        exp_value_chosen = np.exp(self.beta * (value_chosen - max_value))
        exp_value_not_chosen = np.exp(self.beta * (value_not_chosen - max_value))
        p_chosen = exp_value_chosen / (exp_value_chosen + exp_value_not_chosen)
        p_unchosen = exp_value_not_chosen / (exp_value_chosen + exp_value_not_chosen)
        p_highest = np.max([p_chosen, p_unchosen])
        p_lowest = np.min([p_chosen, p_unchosen])
        return 1 - (p_highest - p_lowest)

    def fit(self):
        stimulus_array = self.construct_stimulus()
        choices = self.data["resp_numeric"].values
        rewards = self.data["points"].values

        for i in range(self.n_trials):
            if choices[i] == -1:
                self.rpe[i] = np.nan
                R = np.array([0, 1])
                continue
            else:
                if i == 0:
                    R = np.array([0, 1])
                elif choices[i] == choices[i - 1]:
                    R = np.array([1, 0])
                stim_chosen_dim1 = stimulus_array[choices[i]][0][i]
                stim_chosen_dim2 = stimulus_array[choices[i]][1][i]
                stim_not_chosen_dim1 = 1 - stim_chosen_dim1
                stim_not_chosen_dim2 = 1 - stim_chosen_dim2

                self.value_chosen = (
                    self.W[0, stim_chosen_dim1] + self.W[1, stim_chosen_dim2]
                )
                self.value_not_chosen = (
                    self.W[0, stim_not_chosen_dim1] + self.W[1, stim_not_chosen_dim2]
                )
                self.update_weights(stim_chosen_dim1, stim_chosen_dim2, rewards[i], i)
                self.Ws[:, :, i] = self.W
                self.Vs[i] = self.value_chosen

                p_chosen = self.softmax(
                    self.value_chosen + self.st * R[0],
                    self.value_not_chosen + self.st * R[1],
                )
                # p_chosen = self.softmax(self.value_chosen, self.value_not_chosen)

                self.loglik += np.log(p_chosen)
                self.rpe[i] = rewards[i] - self.value_chosen
                self.trial_by_trial_loglik[i] = np.log(p_chosen)
                self.uncertainties[i] = self.uncertainty_shen(
                    self.value_chosen, self.value_not_chosen
                )

        if np.isnan(self.loglik):
            return np.inf
        return self.loglik


class DecayFeatureRL(FeatureRL):
    def __init__(self, eta, beta, decay, st, data):
        super().__init__(eta, beta, st, data)
        self.decay = decay  # rate at which the value of non-chosen feature decays

    def update_weights(self, stim_chosen_dim1, stim_chosen_dim2, reward, i):
        super().update_weights(stim_chosen_dim1, stim_chosen_dim2, reward, i)
        stim_not_chosen_dim1 = 1 - stim_chosen_dim1
        stim_not_chosen_dim2 = 1 - stim_chosen_dim2

        self.W[0, stim_not_chosen_dim1] *= self.decay
        self.W[1, stim_not_chosen_dim2] *= self.decay
