import matplotlib.pyplot as plt

def PlotStats(avg_fitness_history, max_fitness_history, min_fitness_history):
    plt.plot(avg_fitness_history, label = 'Avg')
    plt.plot(max_fitness_history, label = 'Max')
    # plt.plot(min_fitness_history, label = 'Min')
    plt.title('Fitness over time, max is 100')
    plt.legend()
    plt.show()