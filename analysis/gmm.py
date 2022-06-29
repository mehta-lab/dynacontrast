"""
Script for clustering cell embeddings from multiple time points and conditions using Gaussian Mixture Model (GMM).
Unlike the common GMM where each Gaussian component only has one weight coefficient, the GMM model here allows
each Gaussian component to have different weight coefficients across different times & conditions to model the
change of each cell state over time and across conditions. The embedding dimension is reduced using UMAP before
GMM fit. Optimal number of clusters is determined using Bayesian Information Criterion (BIC).
"""
import numpy as np
import pickle
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans
from sklearn import mixture
import os
import pickle
import cv2
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

class GMM:
    """ Gaussian Mixture Model

    Parameters
    -----------
        k: int , number of gaussian distributions

        seed: int, will be randomly set if None

        max_iter: int, number of iterations to run algorithm, default: 200

    Attributes
    -----------
       centroids: array, k, number_features

       cluster_labels: label for each data point

    """

    def __init__(self, C, n_runs):
        self.C = C  # number of Guassians/clusters
        self.n_runs = n_runs


    def get_params(self):
        return (self.mu, self.pi, self.sigma)

    def calculate_mean_covariance(self, X, prediction):
        """Calculate means and covariance of different
            clusters from k-means prediction

        Parameters:
        ------------
        prediction: cluster labels from k-means

        X: N*d numpy array data points 

        Returns:
        -------------
        intial_means: for E-step of EM algorithm

        intial_cov: for E-step of EM algorithm

        """
        d = X.shape[1]
        comp_ids = np.unique(prediction)
        self.condi_uniq = np.unique(self.conditions)
        self.initial_means = np.zeros((self.C, d))
        self.initial_cov = np.zeros((self.C, d, d))
        self.initial_pi = np.zeros((self.C, len(self.condi_uniq)))

        for i, comp_id in enumerate(comp_ids):
            for j, condi in enumerate(self.condi_uniq):
                ids = (prediction == comp_id) & (self.conditions == condi)  # returns indices
                self.initial_pi[i, j] = sum(ids) / sum(self.conditions == condi)
            ids = (prediction == comp_id)
            self.initial_means[i, :] = np.mean(X[ids], axis=0)
            de_meaned = X[ids] - self.initial_means[i, :]
            Nk = sum(ids) # number of data points in current gaussian
            self.initial_cov[i, :, :] = np.dot(de_meaned.T, de_meaned) / Nk
            initial_pi_sum = np.sum(self.initial_pi, axis=0)
            np.ones((len(self.condi_uniq),))
        # assert (initial_pi_sum == np.ones((len(self.condi_uniq), ))).all()

        return (self.initial_means, self.initial_cov, self.initial_pi)

    def _initialise_parameters(self, X, seed=None):
        """Implement k-means to find starting
            parameter values.
            https://datascience.stackexchange.com/questions/11487/how-do-i-obtain-the-weight-and-variance-of-a-k-means-cluster

        Parameters:
        ------------
        X: numpy array of data points

        Returns:
        ----------
        tuple containing initial means and covariance

        _initial_means: numpy array: (C*d)

        _initial_cov: numpy array: (C,d*d)


        """
        N = X.shape[0]
        self.gamma = np.zeros((N, self.C))
        n_clusters = self.C
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=500, algorithm='auto', random_state=seed)
        fitted = kmeans.fit(X)
        prediction = kmeans.predict(X)
        self._initial_means, self._initial_cov, self._initial_pi = self.calculate_mean_covariance(X, prediction)
        self.pi = self._initial_pi.copy()
        self.mu = self._initial_means.copy()
        self.sigma = self._initial_cov.copy()

    def _weighted_gaussian_prob(self, X):
        N = X.shape[0]
        data_prob = np.zeros((N, self.C))
        for c in range(self.C):
            for j, condi in enumerate(self.condi_uniq):
                ids = self.conditions == condi
                data_prob[ids, c] = self.pi[c, j] * mvn.pdf(X[ids], self.mu[c, :], self.sigma[c])
        return data_prob


    def _e_step(self, X):
        """Performs E-step on GMM model

        Parameters:
        ------------
        X: (N x d), data points, m: no of features
        pi: (C), weights of mixture components
        mu: (C x d), mixture component means
        sigma: (C x d x d), mixture component covariance matrices

        Returns:
        ----------
        gamma: (N x C), probabilities of clusters for objects
        """
        self.gamma = self._weighted_gaussian_prob(X)

        # normalize across columns to make a valid probability
        gamma_norm = np.sum(self.gamma, axis=1)[:, np.newaxis]
        self.gamma /= gamma_norm

    def _m_step(self, X):
        """Performs M-step of the GMM
        We need to update our priors, our means
        and our covariance matrix.

        Parameters:
        -----------
        X: (N x d), data 
        gamma: (N x C), posterior distribution of lower bound 

        Returns:
        ---------
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)
        """
        # N = X.shape[0]  # number of objects
        C = self.gamma.shape[1]  # number of clusters
        # d = X.shape[1]  # dimension of each object
        gamma_c = np.sum(self.gamma, axis=0)[:, np.newaxis] # sum of sample weights over each component
        # responsibilities for each gaussian
        for j, condi in enumerate(self.condi_uniq):
            ids = self.conditions == condi  # returns indices
            self.pi[:, j] = np.mean(self.gamma[ids], axis=0)
        # pi_sum = np.sum(self.pi, axis=0)
        self.mu = np.dot(self.gamma.T, X) / gamma_c

        for c in range(C):
            x = X - self.mu[c, :]  # (N x d)

            gamma_diag = np.diag(self.gamma[:, c])
            x_mu = np.matrix(x)
            gamma_diag = np.matrix(gamma_diag)

            sigma_c = x.T * gamma_diag * x
            self.sigma[c, :, :] = (sigma_c) / gamma_c[c]

        # return self.pi, self.mu, self.sigma

    def _compute_loss_function(self, X):
        """Computes lower bound loss function

        Parameters:
        -----------
        X: (N x d), data 

        Returns:
        ---------
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)
        """
        N = X.shape[0]
        self.loss = np.zeros((N, self.C))

        for c in range(self.C):
            dist = mvn(self.mu[c], self.sigma[c], allow_singular=True)
            for j, condi in enumerate(self.condi_uniq):
                ids = self.conditions == condi  # returns indices
                self.loss[ids, c] = self.gamma[ids, c] * (
                            np.log(self.pi[c, j] + 0.00001) + dist.logpdf(X[ids]) - np.log(self.gamma[ids, c] + 0.000001))
        self.loss = np.sum(self.loss)

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.mu.shape
        cov_params = self.C * n_features * (n_features + 1) / 2.
        mean_params = n_features * self.C
        # degree of freedom = number of component -1 due to normalization constraint
        weight_params = (self.C - 1) * len(self.condi_uniq)
        return int(cov_params + mean_params + weight_params)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        data_prob = self._weighted_gaussian_prob(X)
        data_prob_sum = np.sum(data_prob, axis=1)
        log_data_prob = np.sum(np.log(np.sum(data_prob, axis=1)), axis=0)
        n_parameters = self._n_parameters()
        return (-2 * log_data_prob +
                self._n_parameters() * np.log(X.shape[0]))

    def fit(self, X, conditions=None, seed=None):
        """Compute the E-step and M-step and
            Calculates the lowerbound

        Parameters:
        -----------
        X: (N x d), data 

        Returns:
        ----------
        instance of GMM

        """
        if conditions is None:
            conditions = np.zeros(X.shape[0], dtype=np.int32)
        self.conditions = conditions
        self._initialise_parameters(X, seed=seed)

        # try:
        for run in range(self.n_runs):
            self._e_step(X)
            self._m_step(X)
            self._compute_loss_function(X)

            if (run + 1) % 5 == 0:
                print("Iteration: %d Loss: %0.6f" % (run + 1, self.loss))

        return self

    def predict(self, X):
        """Returns predicted labels using Bayes Rule to
        Calculate the posterior distribution

        Parameters:
        -------------
        X: ?*d numpy array

        Returns:
        ----------
        labels: predicted cluster based on 
        highest responsibility gamma.

        """

        posterior = self.predict_prob(X)
        labels = posterior.argmax(1)

        return labels

    def predict_prob(self, X):
        """Returns predicted labels

        Parameters:
        -------------
        X: N*d numpy array

        Returns:
        ----------
        labels: predicted cluster based on 
        highest responsibility gamma.

        """
        post_prob = np.zeros((X.shape[0], self.C))

        for c in range(self.C):
            for j, condi in enumerate(self.condi_uniq):
                ids = self.conditions == condi  # returns indices
            # Posterior Distribution using Bayes Rule, try and vectorise
                post_prob[ids, c] = self.pi[c, j] * mvn.pdf(X[ids], self.mu[c, :], self.sigma[c])

        return post_prob

def umap_grid(df_meta, fig_name, hue=None, col=None, row=None, alpha=0.02, title=None):
    sns.set_context("poster", font_scale=1.3)
    x_col = 'UMAP 1'
    y_col = 'UMAP 2'

    g = sns.relplot(data=df_meta, x=x_col, y=y_col, col=col, row=row, hue=hue,
                    alpha=alpha, facet_kws={'margin_titles': True})
    # g.set_titles('')
    # g._margin_titles=True
    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    if title is not None:
        g.fig.suptitle(title)
    g.tight_layout()
    plt.savefig(os.path.join(umap_dir, '{}.png'.format(fig_name)), dpi=100, bbox_inches='tight')
#%%
if __name__ == '__main__':
    umap_dir = '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield_tif/2021_04_20_HEK_OC43_widefield_registered/MOI2_rep1/dnm_train/CM+kidney+A549+HEK_ResNet50_moretrj_patchnorm_fullrot_jit_crop_tr'
    data_dirs = [
        '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield_tif/2021_04_20_HEK_OC43_widefield_registered/Mock_rep0',
        '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield_tif/2021_04_20_HEK_OC43_widefield_registered/Mock_rep1',
        '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield_tif/2021_04_20_HEK_OC43_widefield_registered/MOI0.25_rep0',
        '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield_tif/2021_04_20_HEK_OC43_widefield_registered/MOI0.25_rep1',
        '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield_tif/2021_04_20_HEK_OC43_widefield_registered/MOI1_rep0',
        '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield_tif/2021_04_20_HEK_OC43_widefield_registered/MOI1_rep1',
        '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield_tif/2021_04_20_HEK_OC43_widefield_registered/MOI2_rep0',
        '/CompMicro/projects/HEK/2021_04_20_HEK_OC43_63x_04NA_Widefield_tif/2021_04_20_HEK_OC43_widefield_registered/MOI2_rep1', ]

    conditions = [
        #                 'CM mock', 'CM infected',
        #                       'kidney tissue',
        #                       'A549 Mock 24h 60X', 'A549 Mock 48h 60X', 'A549 RSV 24h 60X', 'A549 RSV 48h 60X',
        #                         'A549 MOCK IFNA 48 40X', 'A549 RSV IFNA 24 40X', 'A549 RSV IFNA 48 40X', 'A549 RSV IFNL 24 40X',
        'Mock_rep0', 'Mock_rep1', 'MOI0.25_rep0', 'MOI0.25_rep1', 'MOI1_rep0', 'MOI1_rep1', 'MOI2_rep0', 'MOI2_rep1',
    ]
    pcas = []
    umap, labels = pickle.load(open(os.path.join(umap_dir, 'umap_nbr50_a1.58_b0.9_HEK_long.pkl'), 'rb'))
    # umap, labels = pickle.load(open(os.path.join(umap_dir, 'umap_nbr15_a1.58_b0.9_long.pkl'), 'rb'))
    # umap, labels = pickle.load(open(os.path.join(umap_dir, 'umap_50_nbr.pkl'), 'rb'))
    labels_str = [str(x) for x in labels]
    labels = np.array(labels).astype(np.int32)
    cell_ids = np.arange(len(labels))
    df_meta = pd.DataFrame()
    for data_dir, condition in zip(data_dirs, conditions):
        meta_path = os.path.join(data_dir, 'dnm_supp', 'im-supps', 'patch_meta.csv')
        df_meta_condi = pd.read_csv(meta_path, index_col=0, converters={
            'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
        condi, rep = condition.split('_')
        df_meta_condi['condition'] = condi
        df_meta_condi['replicate'] = rep
        df_meta = df_meta.append(df_meta_condi, ignore_index=True)
    df_meta.reset_index(drop=True, inplace=True)
    df_meta['UMAP 1'] = umap[:, 0]
    df_meta['UMAP 2'] = umap[:, 1]
    df_meta['time (h)'] = df_meta['time'].map(lambda x: '{}h'.format((x * 4) + 10))
    df_meta['condition & time'] = df_meta['condition'] + df_meta['time (h)']
#%%
    from sklearn.decomposition import PCA

    fraction = 1
    seed = 0
    vector_list = []

    label = 0
    pca_frac = 0.5
    model_name = os.path.basename(umap_dir)
    for data_dir in data_dirs:
        embed_dir = os.path.join(data_dir, 'dnm_input', model_name)
        vec = pickle.load(open(os.path.join(embed_dir, 'im_latent_space.pkl'), 'rb'))
        vector_list.append(vec.reshape(vec.shape[0], -1))
    vectors = np.concatenate(vector_list, axis=0)
    pca = PCA(pca_frac, svd_solver='auto')
    print('Fitting PCA model ...')
    pcas = pca.fit_transform(vectors)
    print('vectors shape', vectors.shape)
    print('pcas shape', pcas.shape)

    if fraction != 1:
        np.random.seed(seed)
        sample_ids = np.random.choice(len(df_meta), int(fraction * len(df_meta)), replace=False)
        sample_ids.sort()
        df_meta_sub = df_meta.loc[sample_ids, :]
        vectors_sub = vectors[sample_ids]
        umap_sub = umap[sample_ids]
        pcas_sub = pcas[sample_ids]
    else:
        df_meta_sub = df_meta.copy()
        vectors_sub = vectors
        umap_sub = umap
        pcas_sub = pcas
    print(vectors.shape)
    print(umap_sub.shape)
    x_dict = {'raw': vectors_sub, 'pca': pcas_sub, 'umap': umap_sub}
#%%
    from sklearn import preprocessing
    range_n_clusters = list(range(1, 11))
    bics = []
    data_key = 'umap'
    row = 'condition'
    col = 'time (h)'
    # data_key = 'pca'
    # n_clusters = 5
    X = x_dict[data_key]
    le = preprocessing.LabelEncoder()
    le.fit(df_meta['condition & time'])
    gmm_labels = le.transform(df_meta_sub['condition & time'])
    for n_clusters in range_n_clusters:
        model = GMM(C=n_clusters, n_runs=60)
        fitted_values = model.fit(X, gmm_labels)
        labels = model.predict(X)
        bic = model.bic(X)
        bics.append(bic)
        data_prob = model._weighted_gaussian_prob(X)
        labels_str = labels.astype(str)
        df_meta_sub['cluster'] = labels_str
        fig_name = 'umap_grid_gmm_home_{}_c{}_frac{}'.format(data_key, n_clusters, fraction)
        umap_grid(df_meta_sub, fig_name, hue='cluster', alpha=0.02 / fraction, col=col, row=row)
#%%
    sns.set_context("talk", font_scale=1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(range_n_clusters, bics, linewidth=5,color='b', linestyle='-', alpha=0.5)
    ax.set_xticks(range_n_clusters)
    ax.set_xlabel('N(clusters)')
    ax.set_ylabel('BIC')
    # fig_name = 'bic_2dumap'
    # fig_name = 'bic_pca_{}var_frac{}'.format(pca_frac, fraction)
    fig_name = 'bic_home_{}_frac{}'.format(data_key, fraction)
    plt.savefig(os.path.join(umap_dir, '{}.png'.format(fig_name)), dpi=200, bbox_inches='tight')