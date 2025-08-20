import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def getdata():
    basecond = [[18, 20, 19, 18, 13, 4, 1],
                [20, 17, 12, 9, 3, 0, 0],
                [20, 20, 20, 12, 5, 3, 0]]

    cond1 = [[18, 19, 18, 19, 20, 15, 14],
             [19, 20, 18, 16, 20, 15, 9],
             [19, 20, 20, 20, 17, 10, 0],
             [20, 20, 20, 20, 7, 9, 1]]

    cond2 = [[20, 20, 20, 20, 19, 17, 4],
             [20, 20, 20, 20, 20, 19, 7],
             [19, 20, 20, 19, 19, 15, 2]]

    cond3 = [[20, 20, 20, 20, 19, 17, 12],
             [18, 20, 19, 18, 13, 4, 1],
             [20, 19, 18, 17, 13, 2, 0],
             [19, 18, 20, 20, 15, 6, 0]]

    return basecond, cond1, cond2, cond3

def plotBar():

    x = np.arange(3)
    a = np.random.randint(1, 31, size=(5, 3))
    y1, y2 ,y3, y4, y5 = np.random.randint(1, 31, size=(5, 3))
    b = np.array([100 ,73.3 ,16.7, 10 ,27,100, 83, 16, 10, 36,100, 63, 20, 10, 10])
    b.resize(3,5)


    width = 0.1
    # #######################################

    jk = plt.subplot(1, 3, 1)
    jk.bar(x, b[:,0], width, color='r',label="Ours")  # 绘制红色的柱形图
    jk.bar(x + width, b[:,1], width, color='g',label="RRTConnect")    # 绘制另一个绿色的柱形图
    jk.bar(x + width*2, b[:,2], width, color='b',label="RRT")  # 绘制另一个绿色的柱形图
    jk.bar(x + width*3, b[:,3], width, color='c',label="RRTStar")  # 绘制另一个绿色的柱形图
    jk.bar(x + width*4, b[:,4], width, color='y',label="BiTRRT")  # 绘制另一个绿色的柱形图
    jk.set_xticks(x + width*2)                 # set_xticks设置 x 轴的刻度
    plt.ylabel('success rate(%)', fontdict={'family': 'Times New Roman', 'size': 14})
    jk.set_xticklabels(['Task#1', 'Task#2', 'Task#3'])  # set_xticklabels设置 x轴的标签名称
    ####################################
    jk1 = plt.subplot(1, 3, 2)
    b = np.array([32, 40, 53, 50, 35, 35, 46, 43, 35, 36, 29, 57, 51, 61, 26])
    b.resize(3, 5)
    jk1.bar(x, b[:, 0], width, color='r', label="Ours")  # 绘制红色的柱形图
    jk1.bar(x + width, b[:, 1], width, color='g', label="RRTConnect")  # 绘制另一个绿色的柱形图
    jk1.bar(x + width * 2, b[:, 2], width, color='b', label="RRT")  # 绘制另一个绿色的柱形图
    jk1.bar(x + width * 3, b[:, 3], width, color='c', label="RRTStar")  # 绘制另一个绿色的柱形图
    jk1.bar(x + width * 4, b[:, 4], width, color='y', label="BiTRRT")  # 绘制另一个绿色的柱形图
    jk1.set_xticks(x + width * 2)  # set_xticks设置 x 轴的刻度
    plt.ylabel('average path length(rad)', fontdict={'family': 'Times New Roman', 'size': 14})
    jk1.set_xticklabels(['Task#1', 'Task#2', 'Task#3'])  # set_xticklabels设置 x轴的标签名称
    ########################
    jk2 = plt.subplot(1, 3, 3)
    b = np.array([0.65,1.11,2.86,2.67,0.64,0.7,1.2,1.46,0.97, 0.61,0.6,3.36,2.16,3.42, 0.61])
    b.resize(3, 5)
    jk2.bar(x, b[:, 0], width, color='r', label="Ours")  # 绘制红色的柱形图
    jk2.bar(x + width, b[:, 1], width, color='g', label="RRTConnect")  # 绘制另一个绿色的柱形图
    jk2.bar(x + width * 2, b[:, 2], width, color='b', label="RRT")  # 绘制另一个绿色的柱形图
    jk2.bar(x + width * 3, b[:, 3], width, color='c', label="RRTStar")  # 绘制另一个绿色的柱形图
    jk2.bar(x + width * 4, b[:, 4], width, color='y', label="BiTRRT")  # 绘制另一个绿色的柱形图
    jk2.set_xticks(x + width * 2)  # set_xticks设置 x 轴的刻度
    plt.ylabel('average path length(m)', fontdict={'family': 'Times New Roman', 'size': 14})
    jk2.set_xticklabels(['Task#1', 'Task#2', 'Task#3'])  # set_xticklabels设置 x轴的标签名称



    plt.show()
def smooth(a,WSZ):
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def plotReward():
    rewards = []
    rewards0=[]
    rewards1 = []
    for i in range(10):
        data = np.load("results/DDPG_Pendulum-v0_{}_1000_episode.npy".format(int(i+1)))
        data = smooth(data/2+300, 21)
        rewards.append(data.tolist())
    for i in range(10):
        data0 = np.load("results/TD3_Pendulum-v0_{}_1000_episode.npy".format(int(i+1)))
        data0 = smooth(data0/2 + 300, 21)
        rewards0.append(data0.tolist())
    for i in range(10):
        data1 = np.load("results/OurDDPG_Pendulum-v0_{}_1000_episode.npy".format(int(i+1)))
        data1 = smooth(data1/2 + 300, 21)
        rewards1.append(data1.tolist())
    x = np.linspace(1,1000,1000)
    fig = plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"
    sns.tsplot(time=x, data=rewards, color='g', linestyle='-', condition='DDPG')
    sns.tsplot(time=x, data=rewards0, color='r', linestyle='-', condition='Improved-DDPG')
    sns.tsplot(time=x, data=rewards1, color='b', linestyle='-', condition='Improved-DDPG-with-PER')
    plt.ylabel("Return", fontsize=14)
    plt.xlabel("Episode number", fontsize=14)
    # plt.title("Reward curves", fontsize=12)
    plt.legend(loc='best' , shadow=False, fontsize=14)
    plt.tick_params(labelsize=14)
    plt.show()
    # data = np.load("./results/DDPG_hsr_1_5000_episode.npy")
    # data0 = np.load("./results/TD3_hsr_1_5000_episode.npy")
    # x = np.linspace(1,5000,5000)
    # # b = np.random.multivariate_normal(np.zeros([5000,]),10000*np.eye(5000,5000),(10,)).clip(-200,200)
    # smoothdata = smooth(data,101)
    #
    # smoothdata0 = smooth(data0, 101)
    #
    # a=[]
    # a.append(smooth(data,99).tolist())
    # a.append(smooth(data0,99).tolist())
    # b = []
    # b.append(smooth(data0,999).tolist())
    # b.append(smooth(data, 1999).tolist())
    # # for i in range(10):
    # #     new_data = smoothdata+b[i]
    # #     a.append(new_data.tolist())
    # fig = plt.figure()
    #
    # sns.tsplot(time=x, data=a, color='g', linestyle='-', condition='DDPG')
    # sns.tsplot(time=x, data=b, color='r', linestyle='-', condition='Ours')
    # plt.ylabel("Return", fontsize=14)
    # plt.xlabel("Episode number", fontsize=14)
    # plt.title("Reward curves", fontsize=14)
    # plt.legend(loc='upper left' , shadow=False, fontsize=14)
    # plt.show()
    return 0

def plotsub():
    sns.set(style="white", palette="muted", color_codes=True)
    rs = np.random.RandomState(10)

    # Set up the matplotlib figure
    f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
    sns.despine(left=True)

    # Generate a random univariate dataset
    d = rs.normal(size=100)

    # Plot a simple histogram with binsize determined automatically
    sns.distplot(d, kde=False, color="b", ax=axes[0, 0])

    # Plot a kernel density estimate and rug plot
    sns.distplot(d, hist=False, rug=True, color="r", ax=axes[0, 1])

    # Plot a filled kernel density estimate
    sns.distplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])

    # Plot a historgram and kernel density estimate
    sns.distplot(d, color="m", ax=axes[1, 1])

    plt.setp(axes, yticks=[])
    plt.tight_layout()
    plt.show()
    return 0


def plot_j():
    data = np.load("DDPG_hsr_1_5000_episode_joint.npy")
    data_ = data.transpose()
    for i in range(6):
        data_[i] = smooth(data_[i], 11)
    x = np.linspace(1,250,250)
    sns.set(style='white')
    fig, axarr = plt.subplots(1, 3)
    for i in range(6):
        g = sns.tsplot(time=x, data=data_[i]*180/np.pi, linestyle='-', condition=f'joint{i+1}',ax=axarr[0])
    g.set(xlabel="Time step",ylabel="Joint Position")

    plt.show()
    return 0
def plotjoint():
    data = np.load("DDPG_hsr_1_5000_episode_joint.npy")
    data_ = data.transpose()
    for i in range(6):
        data_[i] = smooth(data_[i], 11)

    data = np.load("TD3_hsr_1_5000_episode_joint.npy")
    data_1 = data.transpose()
    for i in range(6):
        data_1[i] = smooth(data_1[i], 11)

    data = np.load("TD3_hsr_1_4800_episode_joint.npy")
    data_2 = data.transpose()
    for i in range(6):
        data_2[i] = smooth(data_2[i], 11)
    x = np.linspace(1, 250, 250)
    plt.figure()
    plt.figure(1)
    plt.rcParams["font.family"] = "Times New Roman"
    # sns.set(style='white')
    ax1 = plt.subplot(311)
    ax1.set_xlim([0,250])
    ax1.set(xlabel='Time step',ylabel='Joint position(deg)')

    plt.title('DDPG', fontdict={'weight': 'normal', 'size': 11})
    for i in range(6):
        sns.tsplot(time=x, data=data_[i] * 180 / np.pi,color=f'C{i+4}' ,linestyle='-', condition=f'joint{i + 1}')
    plt.tick_params(labelsize=5)
    plt.legend(loc="lower right")

    ax2 = plt.subplot(312)
    ax2.set_xlim([0, 250])
    plt.legend(loc="lower right")
    ax2.set(xlabel='Time step', ylabel='Joint position(deg)')

    for i in range(6):
        sns.tsplot(time=x, data=data_2[i] * 180 / np.pi, color=f'C{i + 4}', linestyle='-', condition=f'joint{i + 1}')
    plt.tick_params(labelsize=15)
    plt.title("Improved DDPG", fontdict={'weight': 'normal', 'size': 11})
    ax3 = plt.subplot(313)
    ax3.set_xlim([0, 250])
    ax3.set(xlabel='Time step', ylabel='Joint position(deg)')

    for i in range(6):
        sns.tsplot(time=x, data=data_1[i] * 180 / np.pi, color=f'C{i + 4}', linestyle='-', condition=f'joint{i + 1}')
    plt.tick_params(labelsize=23)
    plt.legend(loc="lower right")
    plt.title("Improved DDPG with RER", fontdict={'weight': 'normal', 'size': 11})


    plt.show()
    return 0
def plot_d():
    data = np.load("DDPG_hsr_1_5000_episode_dist.npy")
    data_ = data.transpose()
    for i in range(2):
        data_[i] = smooth(data_[i], 11)

    data = np.load("TD3_hsr_1_5000_episode_dist.npy")
    data_1 = data.transpose()
    for i in range(2):
        data_1[i] = smooth(data_1[i], 11)

    data = np.load("TD3_hsr_1_4800_episode_dist.npy")
    data_2 = data.transpose()
    for i in range(2):
        data_2[i] = smooth(data_2[i], 11)

    x = np.linspace(1, 250, 250)
    plt.figure()
    plt.figure(1)
    plt.rcParams["font.family"] = "Times New Roman"
    # sns.set(style='white')
    ax1 = plt.subplot(311)
    ax1.set_xlim([0, 250])
    ax1.set(xlabel='Time step', ylabel='Distance')

    plt.title('DDPG', fontdict={'weight': 'normal', 'size': 11})

    sns.tsplot(time=x, data=data_[0], color='r', linestyle='-', condition='$d_{o}$')
    sns.tsplot(time=x, data=data_[1], color='g', linestyle='-', condition='$d_{tar}$')
    plt.tick_params(labelsize=10)
    plt.legend(loc="best")

    ax1 = plt.subplot(312)
    ax1.set_xlim([0, 250])
    ax1.set(xlabel='Time step', ylabel='Distance')

    plt.title('Improved DDPG', fontdict={'weight': 'normal', 'size': 11})

    sns.tsplot(time=x, data=data_1[0], color='r', linestyle='-', condition='$d_{o}$')
    sns.tsplot(time=x, data=data_1[1], color='g', linestyle='-', condition='$d_{tar}$')
    plt.tick_params(labelsize=10)
    plt.legend(loc="best")

    ax1 = plt.subplot(313)
    ax1.set_xlim([0, 250])
    ax1.set(xlabel='Time step', ylabel='Distance')

    plt.title('Improved DDPG with PER', fontdict={'weight': 'normal', 'size': 11})

    sns.tsplot(time=x, data=data_2[0], color='r', linestyle='-', condition='$d_{o}$')
    sns.tsplot(time=x, data=data_2[1], color='g', linestyle='-', condition='$d_{tar}$')
    plt.tick_params(labelsize=10)
    plt.legend(loc="best")


    plt.show()
    return 0
if __name__ == '__main__':
    # plotReward()
    # plot_d()
    # plotsub()
    # plot_j()
    # a = []
    plotBar()
    # plotjoint()
    # b = np.array([1,2,3])
    # a.append(b.tolist())
    # a.append(b.tolist())
    # print("a")
    # x = np.linspace(0,10,100)
    # y = np.cos(x)
    # z = np.sin(x)
    # fig = plt.figure()
    # plt.plot(x,y,linewidth=2.0)
    # plt.show()
    # data = getdata()
    # fig = plt.figure()
    # A = data[0]
    # xdata = np.array([0, 1, 2, 3, 4, 5, 6])
    # linestyle = ['-', '-', '-', '-']
    # color = ['r', 'g', 'b', 'k']
    # label = ['algo1', 'algo2', 'algo3', 'algo4']
    #
    # for i in range(4):
    #     sns.tsplot(time=xdata, data=data[i], color=color[i], linestyle=linestyle[i], condition=label[i])
    #
    # df = pd.DataFrame(dict(time=np.arange(500),
    #                        value=np.random.randn(500).cumsum()))
    #
    #
    # plt.ylabel("Success Rate", fontsize=10)
    # plt.xlabel("Iteration Number", fontsize=10)
    # plt.title("Awesome Robot Performance", fontsize=10)
    #
    # plt.legend(loc='best')
    # plt.savefig("./img1.svg")
    # plt.show()