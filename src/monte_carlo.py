import numpy as np

from src.probability_analysis import *


def transition_2_cards_monte_carlo(epoch: object = DEFAULT_EPOCH, save_csv_as: object = None) -> object:
    deck = BJDeck()
    df = get_empty_2_to_many_table()
    for _ in range(epoch):
        hand = BJHand()
        hand.add(deck.draw())
        hand.add(deck.draw())
        while hand.get_soft_value() < 21:
            from_state = BJHandState.of_ignore_pair(hand)
            hand.add(deck.draw())
            to_state = BJHandState.of_ignore_pair(hand)
            df.at[from_state, to_state] += 1
    make_row_conditional_prob(df)
    df.to_csv(save_csv_as) if save_csv_as else None
    return df


def dealer_monte_carlo(stand_if_reach=17, epoch=DEFAULT_EPOCH, save_csv_as=None) -> pd.DataFrame:
    df = get_empty_dealer_table()
    deck = BJDeck()
    for _ in range(epoch):
        hand = BJHand()
        first = deck.draw()
        hand.add(first)
        hand.add(deck.draw())
        if hand.is_blackjack():
            df.at['BJ', 'BJ'] += 1
            continue
        df.at['BJ', 'NotBJ'] += 1
        while hand.get_soft_value() < stand_if_reach:
            hand.add(deck.draw())
        value = hand.get_soft_value()
        df.at[first, value if value <= 21 else 'BUSTED'] += 1

    make_row_conditional_prob(df)
    df.to_csv(save_csv_as) if save_csv_as else None
    return df


def read_strategy(filepath_or_buffer) -> pd.DataFrame:
    table = pd.read_csv(filepath_or_buffer, index_col=0)
    table.columns = pd.Index([BJCard[c] for c in table.columns])
    table.index = pd.Index([BJHandState[i] for i in table.index])
    return table


def get_optimal_actions(strategy: pd.DataFrame, dealer_open: BJCard, player_hand: BJHandState) -> List[BJAction]:
    actions = strategy.at[player_hand, dealer_open]
    return [BJAction[a] for a in actions.split(' | ')]


def get_legal_best_action(strategy: pd.DataFrame, dealer_open: BJCard, player_hand: BJHandState,
                          legal_actions: List[BJAction]) -> BJAction:
    best_actions = get_optimal_actions(strategy, dealer_open, player_hand)
    for action in best_actions:
        if action in legal_actions:
            return action
    if player_hand.name[0] == 'P':
        return get_legal_best_action(strategy, dealer_open, player_hand.to_non_pair(), legal_actions)
    else:
        raise Exception('Legal action not found.')


def strategy_diff(actual: pd.DataFrame, expected: pd.DataFrame, save_csv_as=None):
    diff_table = pd.DataFrame(index=actual.index, columns=actual.columns)
    for i in actual.index:
        for c in actual.columns:
            if actual.at[i, c] != expected.at[i, c]:
                print('In Dealer {} and Player {}, actual: {}, expected: {}'.format(c, i, actual.at[i, c],
                                                                                    expected.at[i, c]))
                diff_table.at[i, c] = 'A {} ~ E {}'.format(actual.at[i, c], expected.at[i, c])
    if save_csv_as:
        diff_table.to_csv(save_csv_as)


def run_monte_carlo(strategy: pd.DataFrame, num_rounds=10000):
    Blackjack.default_settings()
    bj = Blackjack()
    rewards = np.zeros(num_rounds, dtype=float)
    records = list()
    for r in range(num_rounds):
        curr_round_reward = 0
        bj.new_round()
        while not bj.is_round_ended():
            dealer_open = bj.dealer_open
            player_hand = BJHandState.of(bj.player_hands[bj.curr_dealing_hand])
            legal_actions = bj.get_legal_actions()
            if BJAction.NO_ACTION in legal_actions:
                action = BJAction.NO_ACTION
            else:
                action = get_legal_best_action(strategy, dealer_open, player_hand, legal_actions)
            _, reward = bj.take_action(action)
            curr_round_reward += reward
            records.append([dealer_open, player_hand, action, reward])
        rewards[r] = curr_round_reward
    np.savetxt("../out/mc_rewards.csv", rewards, delimiter=",")
    df_records = pd.DataFrame(np.array(records), columns=['dealer_open', 'player_hand', 'action', 'reward'])
    df_records.to_csv('../out/mc_records.csv')


if __name__ == '__main__':
    # using mc to confirm result from probability analysis
    transition_2_cards_monte_carlo(save_csv_as='../out/mc_2_to_many.csv')
    dealer_monte_carlo(save_csv_as='../out/mc_dealer_s17.csv')
    dealer_monte_carlo(stand_if_reach=18, save_csv_as='../out/mc_dealer_h17.csv')

    # my_strategy = read_strategy('../tables/basic_strategy.csv')
    exp_strategy = read_strategy('../resources/basic_strategy_reference.csv')
    # strategy_diff(actual=my_strategy, expected=exp_strategy, save_csv_as='../tables/basic_strategy_diff.csv')
    run_monte_carlo(exp_strategy)
