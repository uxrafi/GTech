from martingale import *

def main():
    print("Testing Martingale...")
    print("Author: ", author())
    print("Id:", gtid())
    print("Study Group:", study_group())

    win_prob = .88

    result = get_spin_result(win_prob)
    print("Testing get_spin_result with win probability:", result)

    winnings = simulate_episode()
    print("Testing simulate_episode with winnings:", winnings)

    for win in winnings:
        print("Testing win:", win)
        if win > 0:
            print("Win detected, doubling bet.")
        else:
            print("Loss detected, resetting bet.")
if __name__ == "__main__":
    main()