import numpy as np
import matplotlib.pyplot as plt


class ReactionDiffusion():
	'''
	:param gsize:
	:param dA:
	:param dB:
	:param feed:
	:param k:
	:param delta_t:
	'''
	def __init__(self, gsize, dA, dB, feed, k, delta_t):

		self.gsize = gsize
		self.mA, self.mB = self.set_initialization(self.gsize)
		self.dA = dA
		self.dB = dB
		self.feed = feed
		self.k = k
		self.delta_t = delta_t

	def set_initialization(self, N, random_influence=0.2):
		'''
		:param N: grid size. (N x N)
		:returns A, B: Matrices of the two virtual chemicals A & B.
		'''
		A = (1 - random_influence) * np.ones((N,N)) + random_influence * np.random.random((N,N))

		B = random_influence * np.random.random((N,N))

		N2 = N // 2
		radius = r = int(N / 10.0)

		A[N2 - r:N2 + r, N2 - r:N2 + r] = 0.50
		B[N2 - r:N2 + r, N2 - r:N2 + r] = 0.25

		return (A, B)


	def discrete_laplacian(self, M):
		'''
		Compute the discrete laplacian of matrix M.
		This is the convolution step within the function.
		'''
		L = -4*M
		L += np.roll(M, (0,-1), (0,1)) # right neighbor
		L += np.roll(M, (0,+1), (0,1)) # left neighbor
		L += np.roll(M, (-1,0), (0,1)) # top neighbor
		L += np.roll(M, (+1,0), (0,1)) # bottom neighbor

		return (L)

	def gray_scott_update(self):
		'''
		Updates a concentration configuration according to a Gray-Scott model.
		'''
		dlA = self.discrete_laplacian(self.mA)
		dlB = self.discrete_laplacian(self.mB)

		update_A = (self.dA * dlA - self.mA * self.mB**2 + self.feed * (1 - self.mA)) * self.delta_t
		update_B = (self.dB * dlB + self.mA * self.mB**2 - (self.k + self.feed) * self.mB) * self.delta_t

		self.mA += update_A
		self.mB += update_B

	def visualize(self):
		'''
		'''
		fig, ax = plt.subplots(1, 2, figsize=(6,4))
		ax[0].imshow(self.mA, cmap='Greys')
		ax[1].imshow(self.mB, cmap='Greys')
		ax[0].set_title('A')
		ax[1].set_title('B')
		ax[0].axis('off')
		ax[1].axis('off')
		plt.show()

	def run(self, sim_steps=1000):
		'''
		'''
		for i in range(sim_steps):
			self.gray_scott_update()
