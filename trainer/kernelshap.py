# CODE BURROWED FROM https://github.com/iancovert/shapley-regression/blob/master/shapreg/shapley.py
import numpy as np
import pdb


def calculate_result(A, b, total):
	'''Calculate the regression coefficients.'''
	num_players = A.shape[1]
	try:
		if len(b.shape) == 2:
			A_inv_one = np.linalg.solve(A, np.ones((num_players, 1)))
		else:
			A_inv_one = np.linalg.solve(A, np.ones(num_players))
		A_inv_vec = np.linalg.solve(A, b)
		values = (
			A_inv_vec -
			A_inv_one * (np.sum(A_inv_vec, axis=0, keepdims=True) - total)
			/ np.sum(A_inv_one))
	except np.linalg.LinAlgError:
		raise ValueError('singular matrix inversion. Consider using larger '
						 'variance_batches')
	return values
