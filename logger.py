
class Logger:
    def __init__(self) -> None:
        pass

    '''
    gets wins of current player 1 and player 2
    '''
    def read_from_file(filename):
        with open(filename, "r") as f:
            data = f.readline().split(",")
        p1_data, p2_data = data

        return int(p1_data.split(':')[1]), int(p2_data.split(':')[1])
    
    '''
    updates wins of current player 1 and player 2
    '''
    def update_file(filename, p1_wins, p2_wins):
        # p1 : 0, p2 : 1
        result_string = f"p1 : {p1_wins}, p2 : {p2_wins}"
        with open(filename, "w") as f:
            f.write(result_string)

