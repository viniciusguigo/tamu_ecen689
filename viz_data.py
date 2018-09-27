import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

def parse_episode(log_data, epi_n):
    """ Return states, actions, rewards from a specific episode.

    Args:
        epi_n : episode numnber

    Return:
        states : position, velocity, acceleration  (linear and angular)
        actions : controls applied
        rewards : rewards received by the agent

    """
    # parse specific episode
    steps = np.where(log_data[:,1] == epi_n)
    steps = np.squeeze(np.asarray(steps))


    # TODO: automatically identify action_level, so columns are not hard-coded
    states = log_data[steps,11:]
    actions = log_data[steps,3:7]
    rewards = log_data[steps,7]

    return states, actions, rewards

def main(data_addr):
    # load log
    log_data = np.loadtxt(data_addr, delimiter=',', skiprows=1)
    n_episodes = int(np.max(log_data[:,1]))
    print('Found {} episodes parsing {}'.format(n_episodes, data_addr))

    # parse log
    epi_n = np.random.randint(0,n_episodes)
    print('Plotting data from episode #{}.'.format(epi_n))
    states, actions, rewards = parse_episode(log_data, epi_n)

    # plot data
    plt.figure()
    plt.title('Rewards: Episode #{}'.format(epi_n))
    plt.plot(rewards)
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.title('Controls: Episode #{}'.format(epi_n))
    plt.plot(actions[:,0], label='Roll Rate')
    plt.plot(actions[:,1], label='Pitch Rate')
    plt.plot(actions[:,2], label='Throttle Rate')
    plt.plot(actions[:,3], label='Yaw Rate')
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Vehicle Position: Episode #{}'.format(epi_n))
    ax.plot(
        xs=states[:,0], ys=states[:,1], zs=-states[:,2], label='Vehicle')
    ax.plot(
        xs=[0], ys=[0], zs=[-1],
        color='r', marker='o', label='Landing Pad')
    pad = plt.Circle((0, 0), 0.75, color='r', alpha=.35)
    ax.add_patch(pad)
    art3d.pathpatch_2d_to_3d(pad, z=-1, zdir="z")
    # check distance on x or y axis. use it to scale XY plane
    max_x = np.max(np.abs(states[:,0]))
    max_y = np.max(np.abs(states[:,1]))
    max_pos = np.max([max_x, max_y]) + 0.5
    ax.set_xlim([-max_pos,max_pos])
    ax.set_ylim([-max_pos,max_pos])
    plt.legend(loc='best')

    fig = plt.figure()
    plt.title('Vehicle Attitude: Episode #{}'.format(epi_n))
    plt.plot(states[:,3], label='Pitch')
    plt.plot(states[:,4], label='Roll')
    plt.plot(states[:,5], label='Yaw')
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    # DATA LABELS INSIDE FILE
    data_addr = './data/log_v0.csv'
    main(data_addr)
