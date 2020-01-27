import pandas as pd

from src.blackjack import *


class QValueTable:
    table: Dict[object, Dict[object, float]]  # key: state, value: dictionary of action and reward
    count: Dict[object, Dict[object, int]]  # key: state, value: dictionary of action of count
    alpha: float  # learning rate
    discount: float  # time discount

    def __init__(self, alpha, discount=1):
        self.alpha = alpha
        self.discount = discount
        self.table = {}
        self.count = {}

    def set_learning_rate(self, alpha):
        assert 0 <= alpha <= 1
        self.alpha = alpha

    def get_Q(self, state, action) -> float:
        """
        Get the q-value of the given state and action
        >>> qs = QValueTable(0.5)
        >>> qs.table['1'] = {'a':1,'b':2}
        >>> qs.get_Q('1','a')
        1
        >>> qs.get_Q('1','c')
        0
        >>> qs.get_Q('2','a')
        0
        """
        return self.table.get(state).get(action, 0) if state in self.table else 0

    def get_count(self, state, action) -> int:
        return self.count.get(state).get(action, 0) if state in self.count else 0

    def get_Q_max(self, state) -> Tuple[object, float]:
        """
        Get the v-value of the given state
        >>> qs = QValueTable(0.5)
        >>> qs.table['1'] = {'a':1,'b':2,'c':-1}
        >>> qs.get_Q_max('1')
        ('b', 2)
        >>> qs.get_Q_max('2')
        (None, 0)
        """
        if state in self.table and len(self.table.get(state)) > 0:
            state_table = self.table.get(state)
            max_action = max(state_table, key=lambda i: state_table[i])
            return max_action, state_table[max_action]
        else:
            return None, 0

    def update(self, orig_state, action, next_state, reward):
        """
        Update the q-value given the original state, action, resulting state, and reward.
        >>> qs = QValueTable(0.5)
        >>> qs.update('2', 'a', '3', 1)
        >>> qs.get_Q('2', 'a')
        0.5
        >>> qs.update('2', 'a', '4', 2)
        >>> qs.get_Q('2', 'a')
        1.25
        >>> qs.update('2', 'b', '5', 3)
        >>> qs.get_Q('2', 'b')
        1.5
        >>> qs.update('1', 'a', '2', 0)
        >>> qs.get_Q('1', 'a')
        0.75
        """
        _, next_state_Q_max = self.get_Q_max(next_state)
        new_q = (1 - self.alpha) * self.get_Q(orig_state, action) \
                + self.alpha * (reward + self.discount * next_state_Q_max)
        if orig_state not in self.table:
            self.table[orig_state] = {}
            self.table[orig_state][action] = new_q

            self.count[orig_state] = {}
            self.count[orig_state][action] = 1
        else:
            self.table[orig_state][action] = new_q
            self.count[orig_state][action] = self.get_count(orig_state, action) + 1

class BlackjackQValueTable(QValueTable):
    table: Dict[BJGameState, Dict[BJAction, float]]

    def to_csv(self, file_path):
        data = {}
        for dealer in list(BJCard):
            actions = []
            for hand_state in BJHandState.hard_hands() + BJHandState.soft_hands():
                game_state = BJGameState(BJStage.DEALING_PLAYER, dealer, 0, hand_state)
                action, _ = self.get_Q_max(game_state)
                actions.append(action) if action else actions.append(BJAction.NO_ACTION)
            data[dealer] = actions
        df = pd.DataFrame(data, index=BJHandState.hard_hands() + BJHandState.soft_hands())
        df.to_csv(path_or_buf=file_path)

    def count_to_csv(self, file_path):
        data = {}
        for dealer in list(BJCard):
            counts = []
            for hand_state in list(BJHandState):
                game_state = BJGameState(BJStage.DEALING_PLAYER, dealer, 0, hand_state)
                counts.append(sum(self.get_count(game_state, action) for action in list(BJAction)))
            data[dealer] = counts
        df = pd.DataFrame(data, index=list(BJHandState))
        df.to_csv(path_or_buf=file_path)

if __name__ == '__main__':

    bj = Blackjack()

    Blackjack.no_split_training()

    q_values = BlackjackQValueTable(0.2)

    for _ in range(1000000):
        bj.new_round()
        game_state = bj.get_game_state()
        while not bj.is_round_ended():
            legal_actions = bj.get_legal_actions()
            if BJAction.NO_ACTION in legal_actions:
                pick = BJAction.NO_ACTION
            else:
                pick = random.sample(legal_actions, 1)[0]
            next_state, reward = bj.take_action(pick)
            q_values.update(game_state, pick, next_state, reward)
            game_state = next_state

    q_values.to_csv('q_values.csv')
    q_values.count_to_csv('counts.csv')