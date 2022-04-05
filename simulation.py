import numpy as np
import matplotlib.pyplot as plt
def result_plot(threshold, result_dic, ay):
    
    # 推移をプロット
    plt.rcParams["font.size"] = 18
    fig = plt.figure() #親グラフと子グラフを同時に定義
    ax = fig.add_subplot()
    ax.set_xlabel('threshold')
    ax.set_ylabel(ay)
    x = np.array(threshold)
    
    ours = np.array(result_dic['ours'])
    top = np.array(result_dic['top'])
    random = np.array(result_dic['random'])
    full_irt = np.array(result_dic['full_irt'])

    ours_std = np.array(result_dic['ours_std'])
    top_std = np.array(result_dic['top_std'])
    random_std = np.array(result_dic['random_std'])
    full_irt_std = np.array(result_dic['full_irt_std'])
    
    ax.plot(x, ours, color='red', label='ours')
    ax.plot(x, top, color='blue', label='top')
    ax.plot(x, random, color='green', label='random')
    ax.plot(x, full_irt, color='purple', label='IRT')
    if ay == 'accuracy':
        ax.plot(x, x, color='orange', linestyle="dashed")
    print(x)
    a = 0.15
    plt.fill_between(x, ours - ours_std, ours + ours_std, facecolor='r', alpha=a)
    plt.fill_between(x, top - top_std, top + top_std, facecolor='b', alpha=a)
    plt.fill_between(x, random - random_std, random + random_std, facecolor='g', alpha=a)
    plt.fill_between(x, full_irt - full_irt_std, full_irt + full_irt_std, facecolor='purple', alpha=a)
    fig.legend(bbox_to_anchor=(0.280, 0.400), loc='upper left')
    return plt
