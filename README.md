# Blackjack Model for Training
## Getting Started

The following classes are available to you:

 - BJCard
 - BJDeck
 - BJHand
 - BJHandState
 - BJJSONEncoder
 - BJState
 - BJAction
 - AbstractBJGameState
    - BJEndOfGameState
    - BJGameState
 - BJSurrenderMode
 - BJDoubleDownMode
 - BJDealerMode
 - Blackjack

## Using the Model
### Note on Split
The class `Blackjack` handles all the game logic. To reduce the state space for training, no split mode is recommended. 
To do so, include the following before training:
```python
Blackjack.no_split_training()
```
To find out the optimal strategy, some statistically analysis can be performed (see [Optimizing Split](#Optimizing-Split) 
for more). To reset to default settings, use `Blackjack.default_settings()`.

### Model Basic
`Blackjack` has the following methods:
- `new_round()`: starts a new round.
- `get_legal_action(): List[BJAction]`: gets all the legal actions
- `get_game_state(): AbstractBJGameState`: gets a JSON serializable game state
- `take_action(BJAction): AbstractBJGameState, float`: takes the given action and return the resulting state and action reward. If the given action is illegal, a large penalty is the reward. 
- `is_round_ended(): bool`: tells if the current round is ended

## Optimizing Split 