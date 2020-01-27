"""
Analyze how a given hand will be transiting to another if it takes one more card.
"""
from typing import Deque

import pandas as pd

from src.blackjack import *
from src.util import NestedDict

DEFAULT_EPOCH = 100000
_DEFAULT_CARDS_PROB = {BJCard.R_A: 1 / 13, BJCard.R_2: 1 / 13, BJCard.R_3: 1 / 13, BJCard.R_4: 1 / 13,
                       BJCard.R_5: 1 / 13, BJCard.R_6: 1 / 13, BJCard.R_7: 1 / 13, BJCard.R_8: 1 / 13,
                       BJCard.R_9: 1 / 13, BJCard.R_10: 4 / 13}


def get_empty_2_to_many_table():
    df = pd.DataFrame(index=BJHandState.hard_hands() + BJHandState.soft_hands(),
                      columns=BJHandState.hard_hands() + BJHandState.soft_hands() + BJHandState.terminal_hands(),
                      dtype=float)
    df.fillna(0, inplace=True)
    return df


def get_empty_1_to_2_table():
    df = pd.DataFrame(index=BJHandState.single_card_hands(),
                      columns=BJHandState,
                      dtype=float)
    df.fillna(0, inplace=True)
    return df


def get_empty_dealer_table() -> pd.DataFrame:
    df = pd.DataFrame(index=list(BJCard) + ['BJ'],
                      columns=list(range(1, 22)) + ['BUSTED', 'BJ', 'NotBJ'],
                      dtype=float)
    df.fillna(0, inplace=True)
    return df


def make_row_conditional_prob(df):
    for row in df.index:
        df.loc[row, :] /= sum(df.loc[row, :])
    df.fillna(0, inplace=True)


def transition_2_cards_probability(cards_prob=_DEFAULT_CARDS_PROB, save_csv_as=None) -> pd.DataFrame:
    df = get_empty_2_to_many_table()
    # hard hand
    for hard_hand in range(4, 21):
        from_state = BJHandState['H' + str(hard_hand)]
        for card, card_prob in cards_prob.items():
            if card is BJCard.R_A:
                value = hard_hand + 11
                if value > 21:
                    to_state = BJHandState['H' + str(value - 10)]
                else:
                    to_state = BJHandState['S' + str(value)]
            else:
                value = hard_hand + card.value
                to_state = BJHandState['H' + str(value) if value <= 21 else 'BUSTED']
            df.at[from_state, to_state] += card_prob
    # soft hand
    for soft_hand in range(12, 21):
        from_state = BJHandState['S' + str(soft_hand)]
        for card, card_prob in cards_prob.items():
            value = soft_hand + card.value
            if value > 21:
                to_state = BJHandState['H' + str(value - 10)]
            else:
                to_state = BJHandState['S' + str(value)]
            df.at[from_state, to_state] += card_prob
    # get cond prob for each row
    make_row_conditional_prob(df)
    df.to_csv(save_csv_as) if save_csv_as else None
    return df


def transition_1_card_probability(cards_prob=_DEFAULT_CARDS_PROB, save_csv_as=None) -> pd.DataFrame:
    df = get_empty_1_to_2_table()
    for first, first_prob in cards_prob.items():
        for second, second_prob in cards_prob.items():
            hand = BJHand()
            hand.add(first)
            from_state = BJHandState.of(hand)
            hand.add(second)
            to_state = BJHandState.of(hand)
            df.at[from_state, to_state] += first_prob * second_prob
    # get cond prob for each row
    make_row_conditional_prob(df)
    df.to_csv(save_csv_as) if save_csv_as else None
    return df


def dealer_probability(cards_prob=_DEFAULT_CARDS_PROB, stand_if_reach=17, save_csv_as=None) -> pd.DataFrame:
    df = get_empty_dealer_table()
    hands: Deque[Tuple[BJHand, float]]
    hands = deque()
    for card, card_prob in cards_prob.items():
        hand = BJHand()
        hand.add(card)
        hands.append((hand, card_prob))
    while len(hands) > 0:
        hand, hand_prob = hands.popleft()
        value = hand.get_soft_value()
        if hand.is_blackjack():
            df.at['BJ', 'BJ'] += hand_prob
        elif value >= stand_if_reach:
            df.at['BJ', 'NotBJ'] += hand_prob
            df.at[hand.cards[0], value if value <= 21 else 'BUSTED'] += hand_prob
        else:
            for card, card_prob in cards_prob.items():
                hands.append((BJHand(*hand.cards + [card]), hand_prob * card_prob))
    make_row_conditional_prob(df)
    df.to_csv(save_csv_as) if save_csv_as else None
    return df

def initial_2_cards_probability(cards_prob=_DEFAULT_CARDS_PROB, save_csv_as=None) -> pd.Series:
    s = pd.Series(0, index=BJHandState, dtype=float)
    for first, first_prob in cards_prob.items():
        for second, second_prob in cards_prob.items():
            hand = BJHand()
            hand.add(first)
            hand.add(second)
            s[BJHandState.of(hand)] += first_prob * second_prob
    s.to_csv(save_csv_as) if save_csv_as else None
    return s

def initial_dealer_probability(cards_prob=_DEFAULT_CARDS_PROB, save_csv_as=None) -> pd.Series:
    s = pd.Series(0, index=list(BJCard) + ['BJ'], dtype=float)
    for first, first_prob in cards_prob.items():
        for second, second_prob in cards_prob.items():
            hand = BJHand()
            hand.add(first)
            hand.add(second)
            if hand.is_blackjack():
                s['BJ'] += first_prob * second_prob
            else:
                s[first] += first_prob * second_prob
    s.to_csv(save_csv_as) if save_csv_as else None
    return s


def get_EV_table(cards_prob=_DEFAULT_CARDS_PROB, dealer_stand_if_reach=17) -> NestedDict:
    df_dealer = dealer_probability(cards_prob=cards_prob, stand_if_reach=dealer_stand_if_reach)
    df_1_to_2 = transition_1_card_probability(cards_prob=cards_prob)
    df_2_to_many = transition_2_cards_probability(cards_prob=cards_prob)
    output = NestedDict()
    for action in [BJAction.STAND, BJAction.DOUBLE_DOWN, BJAction.HIT, BJAction.SPLIT]:
        for dealer_open in BJCard:
            for player_hand in BJHandState:
                if player_hand is not BJHandState.H0:  # ignore H0
                    _get_EV_for_Action(output, dealer_open, player_hand, action, df_dealer, df_1_to_2, df_2_to_many)
    return output

def _get_EV_for_Action(EV_table: NestedDict,
                       dealer_open: BJCard,
                       player_hand: BJHandState,
                       action: BJAction,
                       df_dealer: pd.DataFrame,
                       df_1_to_2: pd.DataFrame,
                       df_2_to_many: pd.DataFrame):
    if player_hand.is_terminal_hand():
        return _get_EV_for_terminal_hand(EV_table, dealer_open, player_hand, df_dealer)

    if action is BJAction.STAND:
        return _get_EV_for_stand(EV_table, dealer_open, player_hand, df_dealer)
    elif action is BJAction.DOUBLE_DOWN:
        return _get_EV_for_DD(EV_table, dealer_open, player_hand, df_dealer, df_2_to_many)
    elif action is BJAction.HIT:
        return _get_EV_for_hit(EV_table, dealer_open, player_hand, df_dealer, df_2_to_many)
    elif action is BJAction.SPLIT:
        return _get_EV_for_split(EV_table, dealer_open, player_hand, df_dealer, df_1_to_2, df_2_to_many)


def _get_EV_for_terminal_hand(EV_table: NestedDict,
                              dealer_open: BJCard,
                              player_hand: BJHandState,
                              df_dealer: pd.DataFrame) -> float:
    if EV_table.contains(dealer_open, player_hand, BJAction.STAND):
        return EV_table[dealer_open, player_hand, BJAction.STAND]

    if player_hand is BJHandState.BJ:
        EV_table[dealer_open, player_hand, BJAction.STAND] = 1
    elif player_hand is BJHandState.BUSTED:
        EV_table[dealer_open, player_hand, BJAction.STAND] = -1
    elif player_hand is BJHandState.H21 or player_hand is BJHandState.S21:
        win_prob = 1 - df_dealer.loc[dealer_open, 21]
        EV_table[dealer_open, player_hand, BJAction.STAND] = win_prob
    else:
        raise Exception('Not a terminal hand: {}'.format(player_hand))
    return EV_table[dealer_open, player_hand, BJAction.STAND]


def _get_EV_for_stand(EV_table: NestedDict,
                      dealer_open: BJCard,
                      player_hand: BJHandState,
                      df_dealer: pd.DataFrame) -> float:
    if EV_table.contains(dealer_open, player_hand, BJAction.STAND):
        return EV_table[dealer_open, player_hand, BJAction.STAND]
    if player_hand.is_pair() and not player_hand.is_soft():
        EV_table[dealer_open, player_hand, BJAction.STAND] = EV_table[dealer_open, player_hand.to_non_pair(), BJAction.STAND]
        return EV_table[dealer_open, player_hand, BJAction.STAND]

    dealer_final_prob = df_dealer.loc[dealer_open]
    value = player_hand.hand_value()
    # P(dealer value < player value) + P(dealer busted)
    win = functools.reduce(lambda acc, x: acc + dealer_final_prob[x], range(1, value), 0) + dealer_final_prob['BUSTED']
    # P(loss) = P(dealer_value > player_value)
    loss = functools.reduce(lambda acc, x: acc + dealer_final_prob[x], range(value + 1, 22), 0)

    ev = win - loss
    EV_table[dealer_open, player_hand, BJAction.STAND] = ev
    return ev


def _get_EV_for_hit(EV_table: NestedDict,
                    dealer_open: BJCard,
                    player_hand: BJHandState,
                    df_dealer: pd.DataFrame,
                    df_2_to_many: pd.DataFrame) -> float:
    if EV_table.contains(dealer_open, player_hand, BJAction.HIT):
        return EV_table[dealer_open, player_hand, BJAction.HIT]
    if player_hand.is_terminal_hand():
        return -float('inf')
    if player_hand.is_pair() and not player_hand.is_soft():
        EV_table[dealer_open, player_hand, BJAction.STAND] = EV_table[dealer_open, player_hand.to_non_pair(), BJAction.STAND]
        return EV_table[dealer_open, player_hand, BJAction.STAND]

    # iterate through all possible hands after hit and probability of getting each of those hands
    ev = 0
    for hand, hand_prob in df_2_to_many.loc[player_hand, df_2_to_many.loc[player_hand, :] > 0].items():
        if hand.is_terminal_hand():
            hand_ev = _get_EV_for_terminal_hand(EV_table, dealer_open, hand, df_dealer)
        else:
            hand_ev = max(_get_EV_for_stand(EV_table, dealer_open, hand, df_dealer),
                          _get_EV_for_hit(EV_table, dealer_open, hand, df_dealer, df_2_to_many))
        ev += hand_ev * hand_prob
    EV_table[dealer_open, player_hand, BJAction.HIT] = ev
    return ev


def _get_EV_for_DD(EV_table: NestedDict,
                   dealer_open: BJCard,
                   player_hand: BJHandState,
                   df_dealer: pd.DataFrame,
                   df_2_to_many: pd.DataFrame) -> float:
    if EV_table.contains(dealer_open, player_hand, BJAction.DOUBLE_DOWN):
        return EV_table[dealer_open, player_hand, BJAction.HIT]
    if player_hand.is_terminal_hand():
        return -float('inf')
    if player_hand.is_pair() and not player_hand.is_soft():
        EV_table[dealer_open, player_hand, BJAction.STAND] = EV_table[dealer_open, player_hand.to_non_pair(), BJAction.STAND]
        return EV_table[dealer_open, player_hand, BJAction.STAND]
    #
    ev = 0
    for hand, hand_prob in df_2_to_many.loc[player_hand, df_2_to_many.loc[player_hand, :] > 0].items():
        if hand.is_terminal_hand():
            hand_ev = 2 * _get_EV_for_terminal_hand(EV_table, dealer_open, hand, df_dealer)
        else:
            hand_ev = 2 * _get_EV_for_stand(EV_table, dealer_open, hand, df_dealer)
        ev += hand_ev * hand_prob
    EV_table[dealer_open, player_hand, BJAction.DOUBLE_DOWN] = ev
    return ev


def _get_EV_for_split(EV_table: NestedDict,
                      dealer_open: BJCard,
                      player_hand: BJHandState,
                      df_dealer: pd.DataFrame,
                      df_1_to_2: pd.DataFrame,
                      df_2_to_many: pd.DataFrame) -> float:
    if EV_table.contains(dealer_open, player_hand, BJAction.SPLIT):
        return EV_table[dealer_open, player_hand, BJAction.SPLIT]
    if player_hand.is_terminal_hand() or not player_hand.is_pair():
        return -float('inf')

    # assuming player will never stand right after split since that is never optimal
    # all the possible hands if hit/DD after split
    split_hand = player_hand.split()
    no_pair_hands_ev = 0  # Σ P(hand)E(hand) ∀ hand that is not a pair
    all_hands_ev = 0  # Σ P(hand)E(hand) ∀ hand
    DD_ev = 0
    # for each possible hand if we choose to hit/double down after split
    for hand, hand_prob in df_1_to_2.loc[split_hand, df_1_to_2.loc[split_hand, :] > 0].items():
        non_pair_hand = hand.to_non_pair()
        hit_stand_ev = _get_EV_for_stand(EV_table, dealer_open, non_pair_hand, df_dealer)
        hit_hit_ev = _get_EV_for_hit(EV_table, dealer_open, non_pair_hand, df_dealer, df_2_to_many)

        #  1. the EV assuming the first card after split is from Double Down
        DD_ev += 4 * hit_stand_ev * hand_prob

        #  2. the EV assuming the first card after split is from Hit
        hit_ev = max(hit_stand_ev, hit_hit_ev)
        all_hands_ev += hit_ev * hand_prob
        if not hand.is_pair():
            no_pair_hands_ev += hit_ev * hand_prob

    # assuming you can resplit up to 3 times (i.e., 1 time for the original hand, and 1 time for each of the split hand
    # to find the EV of the original hand so we always choose to split when possible
    # Let EV_k be the EV if the split_hand can split k times
    # EV_0 = 2 * all_hands_ev, i.e., the split hand will not be splitting again
    # EV_1 = 2 * (no_pair_hands_ev + P(resplit) * 2 * EV_0)
    resplit_prob = df_1_to_2.loc[split_hand, player_hand]
    HIT_ev = 2 * (no_pair_hands_ev + resplit_prob * 2 * all_hands_ev)

    ev = max(DD_ev, HIT_ev)
    EV_table[dealer_open, player_hand, BJAction.SPLIT] = ev
    return ev


def get_optimal_Action_EV(action_ev_table: Dict[BJAction, float], legal_actions=set(BJAction)):
    opt_action = None
    opt_ev = -float('inf')
    for a, e in action_ev_table.items():
        if e > opt_ev and a in legal_actions:
            opt_action = a
            opt_ev = e
    return opt_action, opt_ev


def get_optimal_Action_EV_except_DD(action_ev_table: Dict[BJAction, float]):
    opt_action = None
    opt_ev = -float('inf')
    for a, e in action_ev_table.items():
        if a is not BJAction.DOUBLE_DOWN and e > opt_ev:
            opt_action = a
            opt_ev = e
    return opt_action, opt_ev


def get_basic_strategy(EV_table: NestedDict, save_as_csv=None) -> pd.DataFrame:
    df = pd.DataFrame(index=EV_table[BJCard.R_A].keys(), columns=EV_table.keys(), dtype=str)
    for card, d1 in EV_table.items():
        for hand, d2 in d1.items():
            df.at[hand, card] = get_optimal_Action_EV(d2)[0]
    df.to_csv(path_or_buf=save_as_csv) if save_as_csv else None
    return df


def get_ev_table_for_action(EV_table: NestedDict, action: BJAction, save_as_csv=None) -> pd.DataFrame:
    df = pd.DataFrame(index=EV_table[BJCard.R_A].keys(), columns=EV_table.keys(), dtype=float)
    for card, d1 in EV_table.items():
        for hand, d2 in d1.items():
            if EV_table.contains(card, hand, action):
                df.at[hand, card] = EV_table[card, hand, action]
    df.to_csv(path_or_buf=save_as_csv) if save_as_csv else None
    return df



if __name__ == '__main__':
    transition_2_cards_probability(save_csv_as='../out/prob_2_to_many.csv')
    transition_1_card_probability(save_csv_as='../out/prob_1_to_2.csv')
    dealer_probability(save_csv_as='../out/prob_dealer_s17.csv')
    player_initial = initial_2_cards_probability(save_csv_as='../out/prob_initial.csv')
    dealer_initial = initial_dealer_probability(save_csv_as='../out/prob_dealer_initial.csv')

    EV = get_EV_table(dealer_stand_if_reach=17)
    get_basic_strategy(EV, save_as_csv='../out/basic_strategy.csv')
    get_ev_table_for_action(EV, BJAction.HIT, save_as_csv='../out/hit_ev.csv')
    get_ev_table_for_action(EV, BJAction.STAND, save_as_csv='../out/stand_ev.csv')
    get_ev_table_for_action(EV, BJAction.DOUBLE_DOWN, save_as_csv='../out/dd_ev.csv')
    get_ev_table_for_action(EV, BJAction.SPLIT, save_as_csv='../out/sp_ev.csv')
    overall_ev = 0
    for dealer_open, open_prob in dealer_initial.items():
        if dealer_open == 'BJ':
            overall_ev -= open_prob
        else:
            for hand, hand_prob in player_initial.loc[player_initial.values > 0].items():
                Action, ev = get_optimal_Action_EV(EV[dealer_open, hand])
                overall_ev += ev * open_prob * hand_prob
    print('Overall ev is ' + str(overall_ev))
    pass

