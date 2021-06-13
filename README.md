# BlackjackLearning

## Game Representation

This project uses an object-oriented approach to represent the Blackjack game. In [`src/blackjack.py`](src/blackjack.py), the following class are defined:
- `BJCard`: an enum to represent a poker card, doesn't distinguish 10, J, Q, or K.
- `BJDeck`: a deck of cards in Blackjack, may contain an arbitrary number of standard 52-card decks, can be shuffled or unshuffled (for testing).
- `BJHand`: the dealer or a player's hand in a round of Blackjack.
- `BJHandState`: a state of a hand, e.g., busted, blackjack, hard N (doesn't contain Ace and the hand has a value of N), soft N (contain Ace and the soft value is N), pair N (a 2-card hand of value N, splittable if allowed).
- `BJStage`: an enum discribes the stages of a single round of Blackjack game.
- `BJAction`: an enum discribes what action can a player take during its turn.
- `AbstractBJGameState`: a game state of a round.
- `Blackjack`: the entire Blackjack game representation

### How to run a game

While the game representation is designed for running models (e.g. it doesn't distinguish 10, J, Q, K), it can still be run mannually.

In terminal, you can start a Python console with `src/blackjack.py` imported
```
python -i src/blackjack.py
```
Then, in the Python console, you can start a blackjack game with a single player
```python
bj = Blackjack()
```
Start a new round
```python
bj.new_round()
```
Get what legal actions are available
```python
bj.get_legal_actions()  # will return a list of BJAction
```
Take on of the action
```python
bj.take_action(BJAction.HIT)  # will return the resulting game state and reward
```
Get the game state
```python
bj.get_game_state()  # will return the current game state
```
Check if the current round is ended
```python
bj.is_round_ended()
```
If it is, start a new round 
```python
bj.new_round()
```

### Features

The Blackjack representation is built to support different varieties of Blackjack:
- Insurance: allowed, not allowed
- Surrender: early, late, not allowed
- Number of split allowed
- Double down: allowed, allowed only before split, not allowed
- How much a hand of Blackjack will pay extra?
- Does dealer hit or stand on 17?

## Probability Analysis

A probability analysis is done to find out the best strategy. Code in [`src/probability_analysis.py`](src/probability_analysis.py). This is a pure probabilistic approach. It doesn't involve in any simulation but assume each card has 1/13 chance of being drawn from the deck. Note that this assumption deviates from the real world: in a casion, dealer won't add the drawn cards back to the deck and reshuffle it.

Steps:
1. Find the transitional probability (e.g. when player has a hand of soft 17, what is the probability distribution after another card is added to the hand)
2. Given the dealer's open card and the current hand, find the expected value if a specific action is performed. This expected value assumes the dealer doesn't have a natural blackjack before everyone can take an action.
3. Given the dealer's open card and the current hand, find the action with highest expected value. That would be the optimal action (a.k.a. the basic strategy).

The basic strategy is saved in [`out/basic_strategy.csv`](out/basic_strategy.csv).

## Monte Carlo Simulation

A monte carlo simulation is also performed to simulate the game using the `Blackjack` model. See [`src/monte_carlo.py`](src/monte_carlo.py) for code. For output, see the files started with `mc` in the [`out`](out/) folder. They show the simulated probability of how a hand is transitioned to another hand. The code will also simulate the game play itself, but the game log is too large that it is not suitable to be uploaded to GitHub.

## Reinformence Learning

A q-learning is also attempted. See [`src/rl.py`](src/rl.py). However, since the numbers of game states is too large, this method is not practical. However, the original intent of this method is the find the best strategy when card counting is also performed. [Card counting](https://en.wikipedia.org/wiki/Card_counting) is a technique in Blackjack to find out the inbalance in the drawing deck by counting cards. However, my machine is unable to handle the large number of simulation needed to perform q-learning for Blackjack with card counting.
