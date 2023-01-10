import matplotlib.pyplot as plt
import numpy as np

def plot_scores(scores, episode, score, avg_score, epsilon):
    plt.clf()
    plt.plot(scores)
    plt.title(f'Episode {episode}, score {score}, avg score {avg_score}, eps {epsilon:.3f}')
    
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

plt.ion()

def plot_learning_curve(scores):
    # plot learning curve
    x = [i+1 for i in range(len(scores))]
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-1000):(i+1)])
        
    # make a trend line
    z = np.polyfit(x, running_avg, 1)
    p = np.poly1d(z)

    # show formula for trend line on plot
    plt.plot(x, running_avg)
    plt.plot(x,p(x),"r")  
    plt.legend([f"y={z[0]:.20f}x+{z[1]:.2f}"])
    plt.title("Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()
    plt.waitforbuttonpress()