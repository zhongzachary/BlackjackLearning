import functools
import json
import logging
import random
from collections import deque
from enum import Enum, unique
from typing import List, Tuple, Dict, Union

logging.basicConfig(filename='blackjack.log',  filemode='w', level=logging.INFO)

class BJCard(Enum):
    """
    Representing cards in a Blackjack game.

    In Blackjack, the suit of a card is neglected and only its rank matters. Therefore, a card can be represented
    only with its rank. Note that all 10, Q, J, and K have the same values and are considered the same (even under
    situation like splitting), all four cards are representing by R_10.

    >>> BJCard.R_A.value
    1
    """
    R_A = 1
    R_2 = 2
    R_3 = 3
    R_4 = 4
    R_5 = 5
    R_6 = 6
    R_7 = 7
    R_8 = 8
    R_9 = 9
    R_10 = 10

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

class BJDeck(object):
    """
    Represents a deck of cards in Blackjack. It can be composed using an arbitrary number of standard 52-card deck.
    """
    _cards: deque
    num_decks: int
    reshuffle: bool
    _card_counter: Dict[BJCard, int]

    @staticmethod
    def get_ordered_deck() -> List[BJCard]:
        deck = []
        for _ in range(4):
            deck.extend([card for card in BJCard])  # add Ace to 10 to the deck
            deck.extend([BJCard.R_10, BJCard.R_10, BJCard.R_10])  # add J, Q, K to the deck
        return deck

    def __init__(self, num_decks=6, reshuffle=True):
        """
        Create a deck with given number of decks.
        :param reshuffle: if true, the deck will reshuffle now and everything it is repleted
        :type reshuffle: bool
        :param num_decks: the number of standard 52-card decks used to create such deck
        :type num_decks: int
        """
        self.num_decks = num_decks
        self.reshuffle = reshuffle
        self._replete_deck()

    def __iter__(self):
        yield from self._cards

    def __str__(self):
        return [c.__str__() for c in self._cards].__str__()

    def __len__(self):
        return len(self._cards)

    def _replete_deck(self):
        """
        Replete self._card given the number of deck and whether it will reshuffle
        """
        self._cards = deque()
        self._card_counter = {}
        deck = BJDeck.get_ordered_deck()
        [self._cards.extend(deck) for _ in range(self.num_decks)]
        if self.reshuffle:
            random.shuffle(self._cards)

    def draw(self):
        """
        Draw the next card. If there is no card left, the deck will be repleted and then draw.
        :rtype: Card
        >>> deck = BJDeck(2, False)
        >>> deck.draw() == BJCard.R_A
        True
        >>> deck.num_card_drawn(BJCard.R_A) == 1
        True
        """
        if len(self._cards) == 0:
            self._replete_deck()
        card = self._cards.popleft()
        self._card_counter[card] = self._card_counter.get(card, 0) + 1
        return card

    def num_card_drawn(self, card: BJCard) -> int:
        """
        Returns the number of times the given card was drawn from the current deck.
        >>> deck = BJDeck(reshuffle=False)
        >>> (); deck.draw(); () # doctest: +ELLIPSIS
        (...)
        >>> deck.num_card_drawn(BJCard.R_A)
        1
        >>> deck.num_card_drawn(BJCard.R_2)
        0
        """
        return self._card_counter.get(card, 0)


class BJHand(object):
    """
    Represents a hand in a blackjack game. It is a collection of cards.
    """

    cards: List[BJCard]
    doubled: bool

    def __init__(self, *cards):
        """
        Create a new instance of BJHand with the given cards.
        """
        self.cards = []
        self.doubled = False
        for card in cards:
            if isinstance(card, BJCard):
                self.cards.append(card)
            else:
                raise Exception("Expect a BlackjackCard object, actual " + type(card))

    def __len__(self):
        return len(self.cards)

    def __str__(self):
        return self.cards.__str__()

    def add(self, card: BJCard):
        self.cards.append(card)

    def double_down(self, card: BJCard):
        """
        Marking this hand as a double-down hand with the given card, hence, doubling the gain or loss
        """
        self.doubled = True
        self.add(card)

    def split(self):
        """
        Split this hand, returns the newly generated hand
        :rtype: BJHand
        """
        card = self.cards.pop(len(self) - 1)
        return BJHand(card)

    def is_pair(self):
        """
        Returns true if this hand is a pair.
        >>> BJHand(BJCard.R_A, BJCard.R_A).is_pair()
        True
        >>> BJHand(BJCard.R_A, BJCard.R_2).is_pair()
        False
        """
        return len(self.cards) == 2 and self.cards[0] == self.cards[1]

    def is_blackjack(self):
        """
        Returns true if this hand is a blackjack.
        >>> BJHand(BJCard.R_A, BJCard.R_10).is_blackjack()
        True
        >>> BJHand(BJCard.R_10, BJCard.R_10, BJCard.R_A).is_blackjack()
        False
        """
        return len(self.cards) == 2 and self.get_soft_value() == 21

    def get_value(self):
        """
        The (hard) value of the hand.
        >>> hand = BJHand(BJCard.R_A, BJCard.R_4)
        >>> hand.get_value()
        5
        """
        return functools.reduce(lambda acc, card: acc + card.value, self.cards, 0)

    def get_soft_value(self):
        """
        Get the soft value of this hand.
        Ace will be counted as 11 as long as the soft value of the hand doesn't exceed 21.
        >>> BJHand(BJCard.R_A, BJCard.R_4).get_soft_value()
        15
        >>> BJHand(BJCard.R_A, BJCard.R_A, BJCard.R_10).get_soft_value()
        12
        """
        numAce = functools.reduce(lambda acc, card: acc + (1 if card is BJCard.R_A else 0), self.cards, 0)
        soft_value = self.get_value()
        while numAce > 0 and soft_value + 10 <= 21:
            soft_value += 10
            numAce -= 1
        return soft_value

    def is_soft(self):
        return not self.get_value() == self.get_soft_value()


@unique
class BJHandState(Enum):
    """
    Represents a hand. It captures whether the hand is a pair, whether it is a soft hand, and what its (hard) value is.
    """

    # special
    BUSTED = 'BUSTED'
    BJ = 'BJ'
    # soft hand
    S11 = (11, True)
    S12 = (12, True)  # AA, the only soft pair
    S13 = (13, True)
    S14 = (14, True)
    S15 = (15, True)
    S16 = (16, True)
    S17 = (17, True)
    S18 = (18, True)
    S19 = (19, True)
    S20 = (20, True)
    S21 = (21, True)
    # hard hand
    H0 = (0, False, False)
    H2 = (2, False, False)
    H3 = (3, False, False)
    H4 = (4, False, False)
    H5 = (5, False, False)
    H6 = (6, False, False)
    H7 = (7, False, False)
    H8 = (8, False, False)
    H9 = (9, False, False)
    H10 = (10, False, False)
    H11 = (11, False, False)
    H12 = (12, False, False)
    H13 = (13, False, False)
    H14 = (14, False, False)
    H15 = (15, False, False)
    H16 = (16, False, False)
    H17 = (17, False, False)
    H18 = (18, False, False)
    H19 = (19, False, False)
    H20 = (20, False, False)
    H21 = (21, False, False)
    # hard pair
    P4 = (4, False, True)
    P6 = (6, False, True)
    P8 = (8, False, True)
    P10 = (10, False, True)
    P12 = (12, False, True)
    P14 = (14, False, True)
    P16 = (16, False, True)
    P18 = (18, False, True)
    P20 = (20, False, True)

    @staticmethod
    def soft_hands():
        return [BJHandState.S11, BJHandState.S12, BJHandState.S13, BJHandState.S14, BJHandState.S15, BJHandState.S16,
                BJHandState.S17, BJHandState.S18, BJHandState.S19, BJHandState.S20]

    def is_soft(self):
        return self.name[0] == 'S'

    @staticmethod
    def hard_hands():
        return [BJHandState.H2, BJHandState.H3, BJHandState.H4, BJHandState.H5, BJHandState.H6,
                BJHandState.H7, BJHandState.H8, BJHandState.H9, BJHandState.H10, BJHandState.H11, BJHandState.H12,
                BJHandState.H13, BJHandState.H14, BJHandState.H15, BJHandState.H16, BJHandState.H17, BJHandState.H18,
                BJHandState.H19, BJHandState.H20]

    @staticmethod
    def pair_hands():
        return [BJHandState.S12, BJHandState.P4, BJHandState.P6, BJHandState.P8, BJHandState.P10, BJHandState.P12,
                BJHandState.P14, BJHandState.P16, BJHandState.P18, BJHandState.P20]

    def is_pair(self):
        return self is BJHandState.S12 or self.name[0] == 'P'

    def to_non_pair(self):
        if self.name[0] == 'P':
            return BJHandState['H' + self.name[1:]]
        else:
            return self

    @staticmethod
    def terminal_hands():
        return [BJHandState.BUSTED, BJHandState.BJ, BJHandState.H21, BJHandState.S21]

    def is_terminal_hand(self):
        return isinstance(self.value, str) or self.value[0] == 21

    @staticmethod
    def single_card_hands():
        return [BJHandState.S11, BJHandState.H2, BJHandState.H3, BJHandState.H4, BJHandState.H5, BJHandState.H6,
                BJHandState.H7, BJHandState.H8, BJHandState.H9, BJHandState.H10]

    @staticmethod
    def of(hand: BJHand):
        """
        >>> BJHandState.of(BJHand(BJCard.R_7, BJCard.R_7)) is BJHandState.P14
        True
        >>> BJHandState.of(BJHand(BJCard.R_6, BJCard.R_8)) is BJHandState.H14
        True
        >>> BJHandState.of(BJHand(BJCard.R_6, BJCard.R_8, BJCard.R_7)) is BJHandState.H21
        True
        >>> BJHandState.of(BJHand(BJCard.R_A, BJCard.R_10)) is BJHandState.BJ
        True
        >>> BJHandState.of(BJHand(BJCard.R_A, BJCard.R_5, BJCard.R_5)) is BJHandState.S21
        True
        >>> BJHandState.of(BJHand(BJCard.R_A, BJCard.R_5)) is BJHandState.S16
        True
        >>> BJHandState.of(BJHand(BJCard.R_A, BJCard.R_A)) is BJHandState.S12
        True
        >>> BJHandState.of(BJHand(BJCard.R_A, BJCard.R_10, BJCard.R_10)) is BJHandState.H21
        True
        >>> BJHandState.of(BJHand(BJCard.R_2, BJCard.R_10, BJCard.R_10)) is BJHandState.BUSTED
        True
        >>> BJHandState.of(BJHand()) is BJHandState.H0
        True
        """
        return BJHandState._of(hand, False)

    def hand_value(self):
        if self is BJHandState.BUSTED:
            return 0
        elif self is BJHandState.BJ:
            return 21
        else:
            return self.value[0]

    @staticmethod
    def of_ignore_pair(hand: BJHand):
        """
        >>> BJHandState.of_ignore_pair(BJHand(BJCard.R_7, BJCard.R_7)) is BJHandState.H14
        True
        >>> BJHandState.of_ignore_pair(BJHand(BJCard.R_6, BJCard.R_8)) is BJHandState.H14
        True
        >>> BJHandState.of_ignore_pair(BJHand(BJCard.R_6, BJCard.R_8, BJCard.R_7)) is BJHandState.H21
        True
        >>> BJHandState.of_ignore_pair(BJHand(BJCard.R_A, BJCard.R_10)) is BJHandState.BJ
        True
        >>> BJHandState.of_ignore_pair(BJHand(BJCard.R_A, BJCard.R_5, BJCard.R_5)) is BJHandState.S21
        True
        >>> BJHandState.of_ignore_pair(BJHand(BJCard.R_A, BJCard.R_5)) is BJHandState.S16
        True
        >>> BJHandState.of_ignore_pair(BJHand(BJCard.R_A, BJCard.R_A)) is BJHandState.S12
        True
        >>> BJHandState.of_ignore_pair(BJHand(BJCard.R_A, BJCard.R_10, BJCard.R_10)) is BJHandState.H21
        True
        >>> BJHandState.of_ignore_pair(BJHand(BJCard.R_2, BJCard.R_10, BJCard.R_10)) is BJHandState.BUSTED
        True
        """
        return BJHandState._of(hand, True)

    @staticmethod
    def _of(hand: BJHand, ignore_pair: bool):
        """
        >>> BJHandState.of(BJHand(BJCard.R_7, BJCard.R_7)) is BJHandState.P14
        True
        """
        if hand.get_soft_value() > 21:
            return BJHandState.BUSTED
        elif hand.is_blackjack():
            return BJHandState.BJ
        elif hand.is_soft():
            return BJHandState((hand.get_soft_value(), True))
        else:
            return BJHandState((hand.get_soft_value(), False, False if ignore_pair else hand.is_pair()))

    def split(self):
        if self is BJHandState.S12:
            return BJHandState.S11
        elif type(self.value) == tuple and len(self.value) == 3 and self.value[2]:
            return BJHandState((self.value[0] / 2, False, False))
        else:
            raise Exception('Not a pair: ' + str(self))

    def __str__(self):
        """
        >>> print(BJHandState.of(BJHand(BJCard.R_7, BJCard.R_7)))
        P14
        """
        return self.name

    def __repr__(self):
        return self.__str__()


class BJJSONEncoder(json.JSONEncoder):
    """
    Uses to JSON encode any data containing BJHandState.
    """

    def default(self, o):
        return o.__str__() if isinstance(o, BJHandState) else o


class BJStage(Enum):
    """
    Describes stages of a single round of Blackjack game.
    It has the following stages:
    - BEFORE_CHECKING_NATURAL: right after all the initial cards are dealt and before dealer checks if it has a natural
                                blackjack, surrender (ES) and insurance can happen in this stage
    - AFTER_CHECKING_NATURAL: right after the dealer checks if it has a natural blackjack. If not, surrender (LS) can
                                happen in this stage
    """
    BEFORE_CHECKING_NATURAL = 0
    AFTER_CHECKING_NATURAL = 1
    DEALING_PLAYER = 2
    END_OF_ROUND = 3

    def get_next_stage(self):
        return BJStage(self.value + 1 % 4)


class BJAction(Enum):
    NO_ACTION = 'NO_ACTION'
    SURRENDER = 'SURRENDER'
    INSURANCE = 'INSURANCE'
    DOUBLE_DOWN = 'DOUBLE_DOWN'
    SPLIT = 'SPLIT'
    HIT = 'HIT'
    STAND = 'STAND'

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class AbstractBJGameState(object):
    """
    Represents a game state in a round of Blackjack game.

    It is one of:
    - BJEndOfGameState: represents a terminal game state (hence a new round is needed to be started)
    - BJGameState: represents a non-terminal game state
    """
    _hand_state_factory_method = BJHandState.of

    @classmethod
    def ignore_pair(cls, on: bool):
        if on:
            cls._hand_state_factory_method = BJHandState.of_ignore_pair
        else:
            cls._hand_state_factory_method = BJHandState.of

    def __repr__(self):
        return self.__str__()


class BJEndOfGameState(AbstractBJGameState):
    """
    Represents a terminal game state.

    It captures the following information:
    - game stage (always 3)
    - dealer's hand
    - a tuple of player's hands
    """

    _stage_dealer_handStates: Tuple[int, str, Tuple[BJHandState, ...]]

    def __init__(self, dealer_hand: BJHand, *hands: BJHand):
        hand_states = tuple(AbstractBJGameState._hand_state_factory_method(hand) for hand in hands)
        dealer_repr = "Busted" if dealer_hand.get_value() > 21 else "BJ" if dealer_hand.is_blackjack() else str(
            dealer_hand.get_soft_value())
        self._stage_dealer_handStates = (BJStage.END_OF_ROUND.value, dealer_repr, hand_states)

    def __str__(self):
        return json.dumps(self._stage_dealer_handStates, cls=BJJSONEncoder)

    def __eq__(self, o):
        return isinstance(o,
                          BJEndOfGameState) and self._stage_dealer_handStates == o._stage_dealer_handStates

    def __hash__(self):
        return self._stage_dealer_handStates.__hash__()


class BJGameState(AbstractBJGameState):
    """
    Represent a state of in a Blackjack round. It captures the following information:
    - the open card of dealer, representing by the (hard) value of the card as an integer
    - whether dealer has natural blackjack, representing by a bool or None (if it is not checked yet)
    - the hands this agent possesses
    """
    _stage_open_curr_hands: Tuple[int, int, int, Tuple[BJHandState, ...]]

    def __init__(self, stage: BJStage, dealer_open: BJCard, curr: int, *hands: Union[BJHand, BJHandState]):
        """
        >>> print(BJGameState(BJStage.BEFORE_CHECKING_NATURAL, BJCard.R_A, 0, BJHand(BJCard.R_A, BJCard.R_10)))
        [0, 1, 0, ["BJ"]]
        >>> print(BJGameState(BJStage.BEFORE_CHECKING_NATURAL, BJCard.R_A, 0, BJHand(BJCard.R_A, BJCard.R_10), BJHand(BJCard.R_7, BJCard.R_7)))
        [0, 1, 0, ["BJ", "P14"]]
        >>> AbstractBJGameState.ignore_pair(True)
        >>> print(BJGameState(BJStage.BEFORE_CHECKING_NATURAL, BJCard.R_A, 0, BJHand(BJCard.R_A, BJCard.R_10), BJHand(BJCard.R_7, BJCard.R_7)))
        [0, 1, 0, ["BJ", "H14"]]
        """
        def to_hand_state(x):
            return x if isinstance(x, BJHandState) else AbstractBJGameState._hand_state_factory_method(x)
        hand_states = tuple(to_hand_state(hand) for hand in hands)
        self._stage_open_curr_hands = (stage.value,
                                       dealer_open.value,
                                       curr,
                                       hand_states)

    def __str__(self):
        return json.dumps(self._stage_open_curr_hands, cls=BJJSONEncoder)

    def __eq__(self, o):
        """
        >>> state1 = BJGameState(BJStage.BEFORE_CHECKING_NATURAL, BJCard.R_A, 0, BJHand(BJCard.R_A, BJCard.R_10))
        >>> state2 = BJGameState(BJStage.BEFORE_CHECKING_NATURAL, BJCard.R_A, 0, BJHand(BJCard.R_A, BJCard.R_10))
        >>> state1 == state2
        True
        """
        return isinstance(o, BJGameState) and self._stage_open_curr_hands == o._stage_open_curr_hands

    def __hash__(self):
        return self._stage_open_curr_hands.__hash__()


class BJSurrenderMode(Enum):
    NOT_ALLOWED = 0  # surrender is not allowed
    ES = 1  # surrender is allowed even before dealer has checked for blackjack
    LS = 2  # surrender is allowed only after dealer has checked it doesn't have blackjack


class BJDoubleDownMode(Enum):
    NOT_ALLOWED = 0  # double down is not allowed
    ALLOW_BEFORE_SPLIT = 1  # double down is allowed before any split (double down after split is not allowed)
    ALLOW_AFTER_SPLIT = 2  # double down is allowed before or after split


class BJDealerMode(Enum):
    S17 = 0  # stand on soft 17
    H17 = 1  # hit on soft 17


class Blackjack:
    """
    Represents a game of blackjack.

    It is used for training a blackjack_model and hence there is only 1 agent/player.
    To run blackjack game,
    1. create a new instance.
    2. use new_round() to start a new round
    3. use get_legal_actions() to get all the available action now
    4. use take_action(BJAction) to take an action, it will tells you the resulting game state and reward
    5. use is_round_ended() to check if the current round is ended. If so, follow step 1 again.
    """

    # class variables for game variation parameters
    ALLOW_INSURANCE: bool = True
    SURRENDER_MODE: BJSurrenderMode = BJSurrenderMode.LS
    SPLIT_LIMIT: int = 3
    DOUBLE_DOWN_MODE: BJDoubleDownMode = BJDoubleDownMode.ALLOW_AFTER_SPLIT
    BLACKJACK_PAYS: float = 1.5
    DEALER_MODE: BJDealerMode = BJDealerMode.S17

    @classmethod
    def default_settings(cls):
        cls.ALLOW_INSURANCE = True
        cls.SURRENDER_MODE = BJSurrenderMode.LS
        cls.SPLIT_LIMIT = 3
        cls.DOUBLE_DOWN_MODE = BJDoubleDownMode.ALLOW_AFTER_SPLIT
        cls.BLACKJACK_PAYS = 1.5
        cls.DEALER_MODE = BJDealerMode.S17

        AbstractBJGameState.ignore_pair(False)

    @classmethod
    def no_split_training(cls):
        cls.SPLIT_LIMIT = 0

        AbstractBJGameState.ignore_pair(True)

    # overall game stats
    round_elapsed: int
    player_total_rewards: float
    deck: BJDeck

    # current round data
    stage: BJStage
    dealer_open: BJCard  # the open card of dealer
    dealer_hand: BJHand  # the hand of dealer
    player_hands: List[BJHand]  # the list of hands player possesses
    curr_dealing_hand: int  # index of the currently dealing hand, initialized to be 0 every round

    def __init__(self):
        self.round_elapsed = 0
        self.player_total_rewards = 0
        self.deck = BJDeck(6, True)

    def new_round(self):
        self.round_elapsed += 1
        self.stage = BJStage.BEFORE_CHECKING_NATURAL
        self.player_hands = [BJHand(self.deck.draw(), self.deck.draw())]
        self.curr_dealing_hand = 0
        self.dealer_open = self.deck.draw()
        self.dealer_hand = BJHand(self.dealer_open, self.deck.draw())
        logging.info('Current round #: {}. Dealer: {}'.format(self.round_elapsed, BJHandState.of_ignore_pair(self.dealer_hand)))

    def get_legal_actions(self):
        """
        Return a list of legal actions for the agent
        :rtype: List[BJAgentAction]
        >>> # checking ES
        >>> Blackjack.SURRENDER_MODE = BJSurrenderMode.ES
        >>> bj = Blackjack()
        >>> bj.deck = BJDeck(1, False)
        >>> bj.new_round()
        >>> Blackjack.SURRENDER_MODE = BJSurrenderMode.ES
        >>> set(bj.get_legal_actions()) == {BJAction.NO_ACTION, BJAction.SURRENDER}
        True
        >>> bj.dealer_open = BJCard.R_A
        >>> set(bj.get_legal_actions()) == {BJAction.NO_ACTION, BJAction.SURRENDER, BJAction.INSURANCE}
        True
        >>> bj.take_action(BJAction.INSURANCE)[1]
        -0.5
        >>> bj.stage == BJStage.DEALING_PLAYER
        True
        >>> set(bj.get_legal_actions()) == {BJAction.STAND, BJAction.HIT, BJAction.DOUBLE_DOWN}
        True
        >>> bj.player_hands[0] = BJHand(BJCard.R_A, BJCard.R_A)
        >>> set(bj.get_legal_actions()) == {BJAction.STAND, BJAction.HIT, BJAction.DOUBLE_DOWN, BJAction.SPLIT}
        True
        >>> # checking LS
        >>> bj = Blackjack()
        >>> bj.deck = BJDeck(1, False)
        >>> bj.new_round()
        >>> Blackjack.SURRENDER_MODE = BJSurrenderMode.LS
        >>> set(bj.get_legal_actions()) == {BJAction.NO_ACTION}
        True
        >>> bj.dealer_open = BJCard.R_A
        >>> set(bj.get_legal_actions()) == {BJAction.NO_ACTION, BJAction.INSURANCE}
        True
        >>> bj.take_action(BJAction.NO_ACTION)[1]
        0
        >>> bj.stage == BJStage.AFTER_CHECKING_NATURAL
        True
        >>> set(bj.get_legal_actions()) == {BJAction.NO_ACTION, BJAction.SURRENDER}
        True
        >>> bj.take_action(BJAction.NO_ACTION)[1]
        0
        >>> bj.stage == BJStage.DEALING_PLAYER
        True
        >>> set(bj.get_legal_actions()) == {BJAction.STAND, BJAction.HIT, BJAction.DOUBLE_DOWN}
        True
        >>> bj.player_hands[0] = BJHand(BJCard.R_A, BJCard.R_A)
        >>> set(bj.get_legal_actions()) == {BJAction.STAND, BJAction.HIT, BJAction.DOUBLE_DOWN, BJAction.SPLIT}
        True
        """
        legal_actions = []
        if self.stage == BJStage.BEFORE_CHECKING_NATURAL:
            legal_actions.append(BJAction.NO_ACTION)
            if Blackjack.ALLOW_INSURANCE and self.dealer_open == BJCard.R_A:
                legal_actions.append(BJAction.INSURANCE)
            if Blackjack.SURRENDER_MODE == BJSurrenderMode.ES:
                legal_actions.append(BJAction.SURRENDER)
        elif self.stage == BJStage.AFTER_CHECKING_NATURAL:
            legal_actions.append(BJAction.NO_ACTION)
            if Blackjack.SURRENDER_MODE == BJSurrenderMode.LS:
                legal_actions.append(BJAction.SURRENDER)
        elif self.stage == BJStage.DEALING_PLAYER:
            legal_actions.append(BJAction.STAND)
            legal_actions.append(BJAction.HIT)
            num_splits = len(self.player_hands) - 1
            if self.player_hands[self.curr_dealing_hand].is_pair() and num_splits < Blackjack.SPLIT_LIMIT:
                legal_actions.append(BJAction.SPLIT)
            if Blackjack.DOUBLE_DOWN_MODE == BJDoubleDownMode.ALLOW_AFTER_SPLIT:
                if (num_splits == 0 and len(self.player_hands[0]) == 2) \
                        or (num_splits > 0 and len(self.player_hands[0]) == 1):
                    legal_actions.append(BJAction.DOUBLE_DOWN)
            elif Blackjack.DOUBLE_DOWN_MODE == BJDoubleDownMode.ALLOW_BEFORE_SPLIT:
                if num_splits == 0 and len(self.player_hands[0]) == 2:
                    legal_actions.append(BJAction.DOUBLE_DOWN)
        return legal_actions

    def get_game_state(self) -> AbstractBJGameState:
        if self.stage == BJStage.BEFORE_CHECKING_NATURAL:
            return BJGameState(self.stage, self.dealer_open, self.curr_dealing_hand, *self.player_hands)
        elif self.stage in [BJStage.AFTER_CHECKING_NATURAL, BJStage.DEALING_PLAYER]:
            return BJGameState(self.stage, self.dealer_open, self.curr_dealing_hand, *self.player_hands)
        else:
            return BJEndOfGameState(self.dealer_hand, *self.player_hands)

    def take_action(self, action: BJAction) -> Tuple[AbstractBJGameState, float]:
        """
        :param action: action taken by the agent
        :return: the resulting state and reward
        >>> bj = Blackjack()
        >>> Blackjack.default_settings()
        >>> bj.SURRENDER_MODE = BJSurrenderMode.LS
        >>> bj.deck  = BJDeck(1, False)
        >>> bj.new_round()
        >>> bj.get_game_state()
        [0, 3, 0, ["S13"]]
        >>> bj.take_action(BJAction.NO_ACTION)
        ([1, 3, 0, ["S13"]], 0)
        >>> bj.take_action(BJAction.NO_ACTION)
        ([2, 3, 0, ["S13"]], 0)
        >>> bj.take_action(BJAction.DOUBLE_DOWN) # player: A25, dealer 3467
        ([3, "20", ["S18"]], -2)
        >>> bj.new_round()
        >>> bj.get_game_state() # player: 89, dealer: TenTen
        [0, 10, 0, ["H17"]]
        >>> bj.take_action(BJAction.NO_ACTION)[0]
        [1, 10, 0, ["H17"]]
        >>> bj.take_action(BJAction.SURRENDER)
        ([3, "20", []], -0.5)
        >>> bj.new_round()
        >>> bj.get_game_state() # player: TenTen, dealer: A2
        [0, 1, 0, ["P20"]]
        >>> bj.take_action(BJAction.INSURANCE)
        ([1, 1, 0, ["P20"]], -0.5)
        >>> bj.take_action(BJAction.NO_ACTION)[0]
        [2, 1, 0, ["P20"]]
        >>> bj.take_action(BJAction.SPLIT)[0] # p T|T, d A2
        [2, 1, 0, ["H10", "H10"]]
        >>> bj.take_action(BJAction.HIT)[0] # p T3|T, d A2
        [2, 1, 0, ["H13", "H10"]]
        >>> bj.take_action(BJAction.HIT)[0] # p T34|T, d A2
        [2, 1, 0, ["H17", "H10"]]
        >>> bj.take_action(BJAction.STAND)[0] #p T34|T, d A2
        [2, 1, 1, ["H17", "H10"]]
        >>> bj.take_action(BJAction.HIT)[0] #p T34|T5, d A2
        [2, 1, 1, ["H17", "H15"]]
        >>> bj.take_action(BJAction.HIT) #p T34|T56, d A27
        ([3, "20", ["H17", "H21"]], 0)
        >>> bj.new_round()
        >>> bj.dealer_open = BJCard.R_10
        >>> bj.dealer_hand = BJHand(BJCard.R_10, BJCard.R_A)
        >>> bj.player_hands = [BJHand(BJCard.R_10, BJCard.R_10)]
        >>> bj.take_action(BJAction.NO_ACTION)
        ([3, "BJ", ["P20"]], -1)
        """
        logging.info('Stage: {}, Player: {}, Current: {}, Attempt: {}'
                     .format(self.stage, [BJHandState.of(h) for h in self.player_hands], self.curr_dealing_hand, action))

        illegal_action_penalty = -100
        if action not in self.get_legal_actions():
            logging.warning('Illegal action')
            return self.get_game_state(), illegal_action_penalty

        if action == BJAction.SURRENDER:
            logging.info('Stage: {}'.format(self.stage))
            return self._handle_surrender(), -0.5

        reward: float = 0

        if action == BJAction.INSURANCE:
            reward += 1 if self.dealer_hand.is_blackjack() else -0.5

        if self.stage == BJStage.BEFORE_CHECKING_NATURAL or self.stage == BJStage.AFTER_CHECKING_NATURAL:
            if self.dealer_hand.is_blackjack():
                self.stage = BJStage.END_OF_ROUND
                reward = self._calculate_hand_reward(self.player_hands[0])
            else:
                if Blackjack.SURRENDER_MODE == BJSurrenderMode.LS:
                    self.stage = self.stage.get_next_stage()
                else:
                    self.stage = BJStage.DEALING_PLAYER
        elif self.stage == BJStage.DEALING_PLAYER:
            reward = self._handle_regular_actions(action)
        logging.info('Stage: {}, Dealer: {}, Player: {}, Current: {}, Reward: {}'
                     .format(self.stage, BJHandState.of(self.dealer_hand),
                             [BJHandState.of(h) for h in self.player_hands], self.curr_dealing_hand, reward))
        return self.get_game_state(), reward

    def is_round_ended(self) -> bool:
        """
        Return True if the current round is ended, False otherwise.
        """
        return self.stage == BJStage.END_OF_ROUND

    def _handle_regular_actions(self, action: BJAction) -> float:
        """
        Handle actions during DEALING_PLAYER stage. It also handles change of stage if all the hands are dealt.
        :param action: One of STAND, HIT, DOUBLE_DOWN, SPLIT
        :return: the reward of the action. It is only non-zero when all the hands are dealt (even if some hand are busted).
        """
        curr_hand = self.player_hands[self.curr_dealing_hand]
        if action == BJAction.STAND:
            self.curr_dealing_hand += 1
        elif action == BJAction.HIT:
            curr_hand.add(self.deck.draw())
            if curr_hand.get_soft_value() >= 21:
                self.curr_dealing_hand += 1
        elif action == BJAction.DOUBLE_DOWN:
            curr_hand.double_down(self.deck.draw())
            self.curr_dealing_hand += 1
        elif action == BJAction.SPLIT:
            self.player_hands.append(curr_hand.split())
        self._update_stage_after_dealing()
        return self._settle_reward()

    def _update_stage_after_dealing(self):
        """
        Update the stage to END_OF_ROUND if all the hands are dealt
        """
        if self.curr_dealing_hand == len(self.player_hands):
            self.stage = BJStage.END_OF_ROUND

    def _settle_reward(self) -> float:
        """
        Settle the reward of this round if the round is ended. Reward is 0 if the round hasn't ended yet.
        """
        if self.stage != BJStage.END_OF_ROUND:
            return 0
        else:
            self._deal_dealer()
            return functools.reduce(lambda reward, hand: reward + self._calculate_hand_reward(hand), self.player_hands,
                                    0)

    def _deal_dealer(self):
        """
        Deal dealer according to Blackjack.dealer_mode
        """
        if Blackjack.DEALER_MODE == BJDealerMode.S17:
            self._deal_dealer_until(17)
        else:
            self._deal_dealer_until(18)

    def _deal_dealer_until(self, when_stand):
        """
        Deal the dealer a new card until dealer's soft value is greater or equal to when_stand.
        """
        value = self.dealer_hand.get_soft_value()
        if value < when_stand:
            self.dealer_hand.add(self.deck.draw())
            self._deal_dealer_until(when_stand)

    def _calculate_hand_reward(self, hand: BJHand) -> float:
        """
        Calculate the reward of the given hand with respect to self.dealer_hand.
        """
        hand_bet = 2 if hand.doubled else 1
        if hand.get_soft_value() > 21:
            return -1 * hand_bet
        elif self.dealer_hand.is_blackjack():
            return 0 if hand.is_blackjack() else -1 * hand_bet
        elif hand.is_blackjack():
            return Blackjack.BLACKJACK_PAYS * hand_bet
        elif self.dealer_hand.get_soft_value() == hand.get_value():
            return 0
        else:
            return hand_bet if hand.get_soft_value() > self.dealer_hand.get_soft_value() else -1 * hand_bet

    def _handle_dealer_natural(self) -> AbstractBJGameState:
        if self.dealer_hand.is_blackjack():
            self.stage = BJStage.END_OF_ROUND
        else:
            if Blackjack.SURRENDER_MODE == BJSurrenderMode.ES:
                self.stage = BJStage.DEALING_PLAYER
            else:
                self.stage = BJStage.AFTER_CHECKING_NATURAL

        return self.get_game_state()

    def _handle_surrender(self) -> AbstractBJGameState:
        self.stage = BJStage.END_OF_ROUND
        self.player_hands.clear()
        return self.get_game_state()
