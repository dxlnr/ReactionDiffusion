from model import ReactionDiffusion

def main():
	'''
	Main function.
	'''
	delta_t = 1.0
	# Diffusion coefficients
	dA = 0.16
	dB = 0.08
	# define feed/kill rates
	f = 0.060
	k = 0.062
	# grid size
	N = 200
	# simulation steps
	n_steps = 10000

	model = ReactionDiffusion(N, dA, dB, f, k, delta_t)
	model.run(n_steps)
	model.visualize()

if __name__ == "__main__":
	main()
