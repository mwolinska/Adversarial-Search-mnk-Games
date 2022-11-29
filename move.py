from dataclasses import dataclass
from enum import Enum


@dataclass
class Position(object):
    row: int
    column: int

    @classmethod
    def from_tuple(cls, position):
        return cls(position[0], position[1])

    @classmethod
    def from_index(cls, index, n_columns: int):
        return cls((index // n_columns), (index % n_columns))

    def __eq__(self, other):
        if isinstance(other, Position):
            return (self.row == other.row) and (self.column == other.column)


@dataclass
class PositionIndex(object):
    position_index: int

    @classmethod
    def from_position(cls, position: Position, n_columns: int):
        return position.row * n_columns + position.column


class Players(str, Enum):
    MAX = "compute_next_move_for_maximising_player player"
    MIN = "compute_next_move_for_minimising_player player"

@dataclass
class Move(object):
    position: Position
    player_number: int

    def __eq__(self, other):
        if isinstance(other, Move):
            return (self.position == other.position) and (self.player_number == other.player_number)


class GameStatus(str, Enum):
    win = "win"
    loss = "loss"
    draw = "draw"
    ongoing = "ongoing"
