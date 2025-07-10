from BanditEnv import BanditEnv
from Agent import Agent
import numpy as np
from matplotlib import pyplot as plt
import tqdm

def game(env, agent, game_step, N = 2000):
    rewards = np.zeros(game_step)
    opt_actions = np.zeros(game_step)
    for i in tqdm.tqdm(range(N),desc=f'Experiment',leave=False):
        env.reset()
        agent.reset()
        for j in range(game_step):
            action = agent.select_action()
            reward = env.step(action)
            agent.update_q(action,reward)
        action_history, reward_history = env.export_history()
        rewards += reward_history
        opt_actions += (action_history == np.array(env.opt_actions))
    avg_rewards = rewards / float(N)
    opt_actions = opt_actions / float(N)
    return avg_rewards, opt_actions

def plot_graph(results,exp):
    labels = ['Epsilon = 0.0','Epsilon = 0.01','Epsilon = 0.1']
    x_label_name = 'times'
    y_label_names = ['Average Reward','Percentage of Optimal Action Selection']
    figure_names = [exp+'_rewards.png',exp+'_opt_action.png']
    for i in range(2):
        fig = plt.figure()
        plt.plot(results[0][i],'b-',label=labels[0])
        plt.plot(results[1][i],'r-',label=labels[1])
        plt.plot(results[2][i],'g-',label=labels[2])
        plt.xlabel(x_label_name)
        plt.ylabel(y_label_names[i])
        plt.legend()
        plt.grid(True)
        plt.savefig(figure_names[i])

def Experiment(env_stationary,agent_constant,k,game_step,name):
    env = BanditEnv(k,stationary=env_stationary)
    agents = [Agent(k,0.0,agent_constant), Agent(k,0.01,agent_constant), Agent(k,0.1,agent_constant)]
    results = [game(env,agent,game_step=game_step) for agent in agents]
    plot_graph(results,exp=name)
    print(f'{name} End')

def main():
    # Experiment1 (Part 3)
    Experiment(True,None,10,1000,'exp1')
    # Experiment2 (Part 5)
    Experiment(False,None,10,10000,'exp2')
    # Experiment3 (Part 7)
    Experiment(False,0.1,10,10000,'exp3')

    # Experiment4
    Experiment(False,0.05,10,10000,'exp4')
    # Experiment5
    Experiment(False,0.15,10,10000,'exp5')
    # Experiment6
    Experiment(False,0.2,10,10000,'exp6')
    # Experiment7
    Experiment(False,0.25,10,10000,'exp7')


if __name__ == '__main__':
    main()
    
