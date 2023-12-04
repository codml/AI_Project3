import numpy as np

np.set_printoptions(precision=2)
grid = np.zeros([7, 7])
action = np.array([[-1, -1, -100, -1, -1, -1, -1],
                [-1, -1, -100, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -100, -100, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -100, -100, -1, -1, 0]])
policy = np.array([[[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True]],
                    [[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True]],
                    [[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True]],
					[[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True]],
					[[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True]],
					[[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True]],
					[[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True], [True, True, True, True], [True, True, True, True],
					[True, True, True, True]]])

def get_policy(i, j):
	cnt = 0
	for k in range(4):
		if (policy[i, j, k]):
			cnt += 1
	p = 1 / cnt
	return p

def get_action_value(i, j, k, p):
	ret = 0
	if (k == 0): # up
		v = grid[i - 1][j] if (i != 0) else grid[i][j]
		r = action[i - 1][j] if (i != 0) else action[i][j]
	elif (k == 1): # down
		v = grid[i + 1][j] if (i != 6) else grid[i][j]
		r = action[i + 1][j] if (i != 6) else action[i][j]
	elif (k == 2): # left
		v = grid[i][j - 1] if (j != 0) else grid[i][j]
		r = action[i][j - 1] if (j != 0) else action[i][j]
	elif (k == 3): # right
		v = grid[i][j + 1] if (j != 6) else grid[i][j]
		r = action[i][j + 1] if (j != 6) else action[i][j]
	return p * (r + v)

print("policy evaluation\n\n")
for i in range(5000):
	next_state = np.zeros([7, 7], dtype=float)
	for j in range(7):
		for k in range(7):
			p = get_policy(j, k)
			if (j == k and j == 6):
				next_state[j][k] = 0
			else:
				for l in range(4):
					next_state[j][k] += get_action_value(j, k, l, p)
	grid = next_state
	if (i < 3 or (i + 1) % 500 == 0):
		print("Iteration: ", i + 1)
		print(grid)
		print('\n\n')

for n in range(3):
	for j in range(7):
		for k in range(7):
			if (j != k or j != 6):
				actions = np.zeros(4)
				if (j != 0): # up
					actions[0] = grid[j - 1][k] - 1
				else:
					actions[0] = grid[j][k] - 1
				if (j != 6): # down
					actions[1] = grid[j + 1][k] - 1
				else:
					actions[1] = grid[j][k] - 1
				if (k != 0): # left
					actions[2] = grid[j][k - 1] - 1
				else:
					actions[2] = grid[j][k] - 1
				if (k != 6): # right
					actions[3] = grid[j][k + 1] - 1
				else:
					actions[3] = grid[j][k] - 1
				max_action = np.max(actions)
				policy[j][k] = [False, False, False, False]
				for i in np.where(actions == max_action)[0]:
					policy[j][k][i] = True	
	next_state = np.zeros([7, 7], dtype=float)
	for j in range(7):
		for k in range(7):
			p = get_policy(j, k)
			if (j == k and j == 6):
				next_state[j][k] = 0
			else:
				for l in range(4):
					next_state[j][k] += get_action_value(j, k, l, p)
	grid = next_state
	print("policn improvement: ", n + 1, '\n')
	for j in range(7):
		for k in range(7):
			if (j != k or j != 6):
				print('[', end='')
				if (policy[j][k][0]):
					print('Up ', end='')
				if (policy[j][k][1]):
					print('Down ', end='')
				if (policy[j][k][2]):
					print('Left ', end='')
				if (policy[j][k][3]):
					print('Right ', end='')
				print('] ', end='')
		print('')
	print('\n')

grid = np.zeros([7, 7])
for i in range(20):
	next_state = np.zeros([7, 7], dtype=float)
	for j in range(7):
		for k in range(7):
			if (j == k and j == 6):
				next_state[j][k] = 0
			else:
				up = get_action_value(j, k, 0, 1)
				down = get_action_value(j, k, 1, 1)
				left = get_action_value(j, k, 2, 1)
				right = get_action_value(j, k, 3, 1)
				max_value = np.max([up, down, left, right])
				next_state[j][k] = max_value
	grid = next_state
	print(grid, '\n')