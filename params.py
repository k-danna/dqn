
#environment list for ez switching
envs = ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 
        'BipedalWalker-v2', 'Pong-v4', 'SpaceInvaders-v0', 
        'Breakout-v0']
env_name = envs[0] 

#training
train_episodes = 1000
train_print_interval = 100
train_max_steps = 500
save_interval = 10 #episode interval to save the model

#testing
test_episodes = 10
test_print_interval = 1
test_max_steps = 500
test_records = 4 #number of episodes to dump to video

#hyper
mem_max_size = 50000 #max size for replay memory
train_batch = 64 #replay memory batch size
reward_decay = 0.9 #gamma discount for future rewards
learn_rate = 1e-4
explore = 0.14 #epsilon greedy chance to explore
#reward_offset = -1 #scalar added to raw environment rewards
#done_reward = 1000 #scalar for reaching done state

#misc
seed = 42
out_dir = './logs' #base folder for model, any recordings, etc
downsample = 'slow' #slow, fast, none. 'fast' sacrifices quality for speed
recover = False


