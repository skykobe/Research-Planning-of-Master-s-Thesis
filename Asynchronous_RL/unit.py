import skimage
from skimage import transform, color, exposure
import time
import matplotlib.pyplot as plt
import numpy as np

PRE_time = None
TIMES = 0

def preprocess(image):
	image = skimage.color.rgb2gray(image)
	image = skimage.transform.resize(image, (80, 80), mode = 'constant')
	image = skimage.exposure.rescale_intensity(image, out_range=(0,255))
	image = image.reshape(1, image.shape[0], image.shape[1], 1)
	return image

def time_count(t):
	global TIMES, PRE_time
	if TIMES == t:
		return 3  # for end the running
	if PRE_time == None:
		PRE_time = time.time()
	if abs(PRE_time - time.time()) >= 3600:
		PRE_time = time.time()
		TIMES += 1
		return 2 # for save the model and the training reward
	else:
		return 1 # no operation

def save_model(saver, sess):
	saver.save(sess, 'flappybird_EE_a3c.ckpt')

def load_model(saver, sess, check):
	path = check.model_checkpoint_path
	saver.restore(sess, path)

def state_transform(state):
    length = 4
    tmp = []
    new_state = []
    for img in state:
        if len(tmp) < length:
            tmp.append(img)
        if len(tmp) == length:
            scrn = np.concatenate((tmp[0], tmp[1], tmp[2], tmp[3]), axis=3)
            new_state.append(scrn)
            tmp = []
    if len(tmp) == 1:
        scrn = np.concatenate((tmp[0], tmp[0], tmp[0], tmp[0]), axis=3)
        new_state.append(scrn)
    if len(tmp) == 2:
        scrn = np.concatenate((tmp[0], tmp[0], tmp[1], tmp[1]), axis=3)
        new_state.append(scrn)
    if len(tmp) == 3:
        scrn = np.concatenate((tmp[0], tmp[1], tmp[2], tmp[2]), axis=3)
        new_state.append(scrn)
    return new_state

def action_transform(actions):
	length = 4
	count = length
	new_actions = []
	for a in actions:
		if count == length:
			new_actions.append(a)
			count = count - 1
			continue
		count = count - 1
		if count == 0:
			count = length
	return new_actions

def reward_transform(reward):
	length = 4
	tmp = []
	new_reward = []
	for r in reward:
		if len(tmp) < length:
			tmp.append(r)
		if len(tmp) == length:
			tmp = np.array(tmp)
			new_reward.append(tmp.sum())
			tmp = []
	if len(tmp) > 0:
		tmp = np.array(tmp)
		new_reward.append(tmp.sum())
	return new_reward

def get_state_index(state, memory):
	index = 0
	for i in memory:
	    x = state == i
	    x = x.all()
	    if x:
	        return index
	    index += 1
	return -1
