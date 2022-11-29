import abc
from typing import List, Optional

import numpy as np

from move import Move, Players, Position


class AbstractPlayer(abc.ABC):
    def __init__(self):
        self.player_number = None

    @abc.abstractmethod
    def make_move(self) -> Move:
        pass


class HumanPlayer(AbstractPlayer):
    def __init__(self, player_number: int, player_type: Players, symbol: str):
        self.player_number = player_number
        self.player_type = player_type
        self.symbol = symbol

    def make_move(self) -> Move:
        row = int(input("Input row"))
        column = int(input("input column"))
        move = Move(Position(row=row, column=column),
                    player_number=self.player_number)
        return move



class SimulationPlayer(AbstractPlayer):
    def __init__(self, player_number: int, player_type: Players, positions_to_play: List[Position], symbol: str):
        self.player_number = player_number
        self.player_type = player_type
        self.moves_to_play = positions_to_play

    def make_move(self) -> Move:
        next_move = self.moves_to_play[0]
        self.moves_to_play.remove(next_move)
        move = Move(next_move, player_number=self.player_number)
        return move
