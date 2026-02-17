from acoustools.Utilities import TRANSDUCERS, get_convert_indexes, TOP_BOARD, BOTTOM_BOARD

if __name__ == "__main__":

    board = TRANSDUCERS

    
    IDX = get_convert_indexes(board.shape[0],'top')
    # print(IDX)
    flip = board[IDX]
    for i,row in enumerate(flip):
        print(i+1, row)