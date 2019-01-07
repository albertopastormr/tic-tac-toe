import matplotlib.pyplot as plt
import numpy as np

def draw_end_game(x, y=None, categorical=False):
    # data preparation
    cell_x = x.reshape(3,3)
    if not categorical:
        convert_to_game_cell = lambda x : 'x' if x == 1 else ('o' if x == 0 else ' ')
        cell_x = np.vectorize(convert_to_game_cell)(cell_x)
    else:
        cell_x = np.vectorize(lambda x : ' ' if x == 'b' else x)(cell_x)

    if y is not None:
        if not categorical:
            game_result = 'x' if y == 1 else 'o'
        else:
            game_result = y
    # plot config
    plt.rcParams.update({'font.size': 6})
    plt.figure(figsize=(0.5,0.7), dpi=180, frameon=False)
    plt.axis('off')            

    tab = plt.table(cellText=cell_x,
                #rowLabels=range(len(cell_x)),
                #colLabels=range(len(cell_x)),
                #rowColours=[0.65] * len(cell_x),
                #colColours=[0.65] * len(cell_x),
                #cellColours= [[0.95] * len(cell_x)] * len(cell_x),
                cellLoc='center',
                loc='upper left')
    plt.title( f"Game end board; Won '{game_result}'" if y is not None else "Game end board")
    plt.show()