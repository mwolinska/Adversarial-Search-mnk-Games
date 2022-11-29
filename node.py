import math
from dataclasses import dataclass
from typing import List, Union

import numpy as np

from game_board import GameBoard
from move import Position, Move, Players



@dataclass
class Leaf:
    board_at_node: GameBoard
    last_move: Move
    score: Union[int, float]

@dataclass
class Node:
    board_at_node: GameBoard
    last_move: Move
    children_nodes: List[Union["Node", Leaf]]
    score: Union[int, float]
    node_type: Players
    alpha: Union[int, float] = -math.inf
    beta: Union[int, float] = math.inf

    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.last_move == other.last_move) and (self.board_at_node.board == other.board_at_node.board)
