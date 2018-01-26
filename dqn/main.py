from __future__ import print_function

import time
import gym
import gym_ple

from agent import Agent

def run(episode=100000000, is_training=True):
    env = gym.make('FlappyBird-v0')
    agent = Agent(env)
    # agent.load_net('./tb/checkpoints/2190000')
    for e in range(episode):
        ob = env.reset()
        ob = agent.preproc(ob)
        done = False
        score = step = 0
        start_time = time.time()
        while not done:
            if is_training is False:
                env.render()
            ac = agent.act(ob, is_training)
            next_ob, rew, done, _ = env.step(ac)
            # if rew == 0:
            #     rew = 0.1
            if is_training:
                ob = agent.memory(ob, ac, next_ob, rew, done)
            else:
                ob = agent.preproc(next_ob)
            score += rew
            step += 1

        agent.get_score(score)
        print('episode: {} | score: {} | fps: {}'.format(e, score, step/(time.time() - start_time)))

if __name__ == '__main__':
    run()
