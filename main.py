class Bayes(object):

    def __init__(self, hypos, priors, obs, likelihood):
        self.hypos = hypos
        self.priors = priors
        self.obs = obs
        self.likelihood_matrix = likelihood

    def likelihood(self, o, h):
        index_o = self.obs.index(o)
        index_h = self.hypos.index(h)

        return self.likelihood_matrix[index_h][index_o]

    def norm_constant(self, o):
        return sum(prior * self.likelihood(o, h) for prior, h in zip(self.priors, self.hypos))

    def single_posterior_update(self, o, prs):
        self.priors = prs
        normalization = self.norm_constant(o)
        posteriors = []
        for prior, h in zip(self.priors, self.hypos):
            posterior = self.likelihood(o, h) * prior / normalization
            posteriors.append(posterior)
        return posteriors

    def compute_posterior(self, observations):
        prs = self.priors
        for o in observations:
            posteriors = self.single_posterior_update(o, prs)
            prs = posteriors
        return prs


def cookie_problem(output_file):
    hypos = ["Bowl1", "Bowl2"]
    priors = [0.5, 0.5]
    obs = ["chocolate", "vanilla"]
    # e.g. likelihood[0][1] corresponds to the likehood of Bowl1 and vanilla, or 35/50
    likelihood = [[15 / 50, 35 / 50], [30 / 50, 20 / 50]]

    b = Bayes(hypos, priors, obs, likelihood)

    l = b.likelihood("chocolate", "Bowl1")
    print("likelihood(chocolate, Bowl1) = %s " % l)

    n_c = b.norm_constant("vanilla")
    print("normalizing constant for vanilla: %s" % n_c)

    p_1 = b.single_posterior_update("vanilla", [0.5, 0.5])
    print("vanilla - posterior: %s" % p_1)

    # Question 1
    output_file.write(f'{p_1[0]}\n')

    p_2 = b.compute_posterior(["chocolate", "vanilla"])
    print("chocolate, vanilla - posterior: %s" % p_2)

    # Question 2
    output_file.write(f'{p_2[1]}\n')

def archery_problem(output_file):
    hypos = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
    priors = [0.25, 0.25, 0.25, 0.25]
    obs = ['yellow', 'red', 'blue', 'black', 'white']
    likelihood = [[0.05, 0.1, 0.4, 0.25, 0.2],
                  [0.1, 0.2, 0.4, 0.2, 0.1],
                  [0.2, 0.4, 0.25, 0.1, 0.05],
                  [0.3, 0.5, 0.125, 0.05, 0.025]]

    b = Bayes(hypos, priors, obs, likelihood)

    observations = ['yellow', 'white', 'blue', 'red', 'red', 'blue']
    posteriors = b.compute_posterior(observations)

    for level, probability in zip(hypos, posteriors):
        print(f'{level}: {probability:.3f}')

    # Question 3
    output_file.write(f'{posteriors[1]}\n')  # probability its intermediate (second entry)

    # Question 4
    maximum = -float('inf')
    maximum_level = None
    for p, h in zip(posteriors, hypos):
        if p > maximum:
            maximum = p
            maximum_level = h
    output_file.write(f'{maximum_level}\n')

if __name__ == '__main__':
    with open('group_25.txt', 'w') as output_file:
        cookie_problem(output_file)
        print()
        archery_problem(output_file)

