import copy
import math
import time
from enum import Enum
from multiprocessing import Pool
from typing import Tuple, Union
from itertools import repeat

import dill as dill
import numpy as np
from tqdm import tqdm

from game_board import GameBoard
from move import Move, Position, Players, PositionIndex
from node import Leaf, Node
from player import AbstractPlayer, SimulationPlayer, HumanPlayer


class GameStatus(str, Enum):
    win = "win"
    loss = "loss"
    draw = "draw"
    ongoing = "ongoing"

    @staticmethod
    def status_to_outcome(status: "GameStatus"):
        if status == GameStatus.win:
            return 1
        elif status == GameStatus.loss:
            return -1
        elif status == GameStatus.draw:
            return 0
        elif GameStatus.ongoing:
            return None


class Game:
    def __init__(
        self,
        m: int, n: int, k: int,
        ab_pruning: bool = True, draw_visual: bool = True,
        simulate: bool = False,
    ):
        self.number_of_rows = m
        self.number_of_columns = n
        self.number_to_align = k
        self.game_board = self.initialize_game()

        # TODO allow AI to start
        if simulate:
            self.player_list = [
                SimulationPlayer(1, Players.MAX, [], "x"),
                SimulationPlayer(2, Players.MIN, [], "o",)
            ]
        else:
            self.player_list = [
                HumanPlayer(1, Players.MAX, "x"),
                SimulationPlayer(2, Players.MIN, [], "o"),
            ]
        self.current_player_index = 0
        self.game_tree = None
        self.prune = ab_pruning
        self.visualise = draw_visual
        self.time_to_build_tree = 0
        self.time_per_decision = []


    def play(self):
        # build strategy
        # loop while game has not ended
        tic = time.time()
        self.simulate_all_games()
        self.score_all_games()
        toc = time.time()
        self.time_to_build_tree = toc-tic
        # print(f"time to build tree {toc-tic}")

        has_game_ended = False
        if self.visualise:
            self.drawboard()
        while not has_game_ended:
            current_player = self.get_next_player()

            if isinstance(current_player, HumanPlayer):
                next_move = self.compute_next_move_for_maximising_player()
                self.report_position(next_move.position)
            elif isinstance(current_player, SimulationPlayer):
                tic = time.time()
                if current_player.player_type == Players.MIN:
                    next_move = self.compute_next_move_for_minimising_player()

                elif current_player.player_type == Players.MAX:
                    next_move = self.compute_next_move_for_maximising_player()
                self.update_next_position_for_current_player(next_move.position)
                toc = time.time()
                self.time_per_decision.append(toc-tic)
                # print(f" time to move {toc-tic}")

            has_game_ended, outcome_string, outcome, last_move = self.play_one_move()
            self.update_game_tree(last_move)

            if self.visualise:
                self.drawboard()
        if self.visualise:
            print(outcome_string)

    def compute_next_move_for_maximising_player(self):
        """Determine maximising move for player."""
        current_max_score = -math.inf
        current_maximising_move = None
        for child_node in self.game_tree.children_nodes:
            if child_node.score > current_max_score:
                current_max_score = child_node.score
                current_maximising_move = child_node.last_move

        return current_maximising_move

    def report_position(self, position: Position):
        print(f"Suggested move row: {position.row}, "
              f"column: {position.column}")

    def update_next_position_for_current_player(self, next_position: Position):
        self.player_list[self.current_player_index % 2].moves_to_play.append(
            next_position)

    def compute_next_move_for_minimising_player(self):
        """Determine minimising move for player."""
        current_min_score = math.inf
        current_minimising_move = None
        for child_node in self.game_tree.children_nodes:
            if child_node.score < current_min_score:
                current_min_score = child_node.score
                current_minimising_move = child_node.last_move
        return current_minimising_move

    def update_game_tree(self, last_move: Move):
        for child_node in self.game_tree.children_nodes:
            if (child_node.last_move == last_move) \
                    and \
                    child_node.board_at_node.board.all() == self.game_board.board.all():
                self.game_tree = child_node
                pass

    def play_one_move(self) -> Tuple[bool, str, GameStatus, Move]:
        current_player = self.get_next_player()

        move = current_player.make_move()
        is_move_valid = self.is_valid(move)
        while not is_move_valid:
            print("Position has already been played or does not exist")
            move = current_player.make_move()
            is_move_valid = self.is_valid(move)

        self.game_board.update_board(move)
        has_game_ended, outcome_string, outcome = self.is_game_over(move)

        self.current_player_index += 1

        return has_game_ended, outcome_string, outcome, move

    def get_next_player(self) -> AbstractPlayer:
        player = self.player_list[self.current_player_index % 2]
        return player

    def initialize_game(self):
        # TODO initialize mnk here or in init
        return GameBoard(m=self.number_of_rows, n=self.number_of_columns)

    def drawboard(self):
        print(self.game_board.board)
        pass

    def is_valid(self, move: Move)-> bool:
        """Check if move is valid."""
        position_index = PositionIndex.from_position(move.position, self.number_of_columns)
        position_available = position_index in self.game_board.available_positions_list

        try:
            test_value = self.game_board.board[move.position.row, move.position.column]
        except IndexError:
            print("This position doesn't exist in the board")
            return False

        return position_available

    def is_terminal(self):
        """Check if state is terminal i.e. if game is over."""
        pass

    def has_player_won(self, last_move: Move) -> bool:
        # possible win directions in order:
        # [left, right], [up, down],
        # [left top diagonal, right bottom diagonal],
        # [left bottom diagonal, right top diagonal]

        possible_win_directions = [
            [Position(0, -1), Position(0, 1)],
            [Position(-1, 0), Position(1, 0)],
            [Position(-1, -1), Position(1, 1)],
            [Position(1, -1), Position(-1, 1)],
        ]

        for dimension in possible_win_directions:
            stones_in_dimension = 0
            for direction in dimension:
                stones_in_dimension += self.game_board.count_neighbours(direction, last_move)

            if stones_in_dimension == (self.number_to_align - 1):
                return True

        return False

    def is_game_draw(self) -> bool:
        """Return True if there are no positions left, False otherwise."""
        return not bool(list(self.game_board.available_positions_list))

    def is_game_over(self, last_move: Move) -> Tuple[bool, str, GameStatus]:
        if self.has_player_won(last_move):
            if last_move.player_number == 1: # TODO fix this if
                outcome = GameStatus.win
            elif last_move.player_number == 2:
                outcome = GameStatus.loss
            outcome_string = "Player " + str(last_move.player_number) + " won the game"
            return True, outcome_string, outcome
        else:
            if self.is_game_draw():
                outcome_string = "This game is a draw"
                return True, outcome_string, GameStatus.draw
            else:
                outcome_string = "The game continues"
                return False, outcome_string, GameStatus.ongoing

    def simulate_all_games(self):
        a_game = copy.deepcopy(self)
        a_game.visualise = False
        a_game.player_list = [
            SimulationPlayer(1, Players.MAX, [], "x"),
            SimulationPlayer(2, Players.MIN, [], "o")
        ]

        current_player = a_game.get_next_player()
        possible_position_indices = a_game.game_board.available_positions_list

        self.game_tree = Node(
            board_at_node=copy.deepcopy(a_game.game_board),
            last_move=None,
            children_nodes=[],
            score=-math.inf,
            node_type=current_player.player_type,
        )

        positions_to_play = [Position.from_index(position_index, self.number_of_columns) for position_index in possible_position_indices]

        with Pool() as pool:
            new_node = pool.starmap(self.recursive_search, zip(repeat(copy.deepcopy(a_game)), positions_to_play))
            self.game_tree.children_nodes = new_node

    def recursive_search(
        self,
        a_game: "Game",
        next_position: Position,
    ) -> Union[Leaf, Node]:
        a_game.update_next_position_for_current_player(next_position)
        current_player = a_game.get_next_player()
        next_move = Move(
            position=next_position,
            player_number=current_player.player_number,
        )
        is_game_over, outcome_string, outcome, _ = a_game.play_one_move()
        next_player = a_game.get_next_player()

        if is_game_over:
            outcome_score = GameStatus.status_to_outcome(outcome)

            return Leaf(
                board_at_node=copy.deepcopy(a_game.game_board),
                last_move=next_move,
                score=outcome_score,
            )
        else:
            child_score = -math.inf if next_player.player_type == Players.MIN else math.inf
            child_node = Node(
                board_at_node=copy.deepcopy(a_game.game_board),
                last_move=next_move,
                children_nodes=[],
                score=child_score,
                node_type=next_player.player_type,
            )
            for available_position in a_game.game_board.available_positions_list:
                next_position_to_play = Position.from_index(available_position,
                                                            self.number_of_columns)
                next_game = copy.deepcopy(a_game)
                new_node = self.recursive_search(next_game, next_position_to_play)
                child_node.children_nodes.append(new_node)
            return child_node

    def score_all_games(self):
        score = self.minimax(self.game_tree)
        self.game_tree.score = score

    def minimax(self, node: Union[Node, Leaf]):
        if isinstance(node, Leaf):
            return node.score
        else:
            # if node.node_type == Players.MAX:
            #     return self.max(node)
            # elif node.node_type == Players.MIN:
            #     return self.min_cw(node)
            if node.node_type == Players.MAX and self.prune:
                return self.max_prune(node)
            elif node.node_type == Players.MAX and not self.prune:
                return self.max(node)
            elif node.node_type == Players.MIN and self.prune:
                return self.min_prune(node)
            elif node.node_type == Players.MIN and not self.prune:
                return self.min_cw(node)

    def max(self, node: Union[Node, Leaf]) -> int:
        """Compute minimax value for maximising player."""
        for child_node in node.children_nodes:
            score = self.minimax(child_node)
            if score > child_node.score:
                child_node.score = score
        all_children_scores = [child_node.score for child_node in
                               node.children_nodes]
        return max(all_children_scores)

    def max_prune(self, node: Union[Node, Leaf]) -> Node:
        score = - math.inf
        for child_node in node.children_nodes:
            child_score = self.minimax(child_node)
            score = max(score, child_score)
            if score >= node.beta:
                break
            node.alpha = max(node.alpha, score)
            child_node.score= score
        return score

    def min_prune(self, node: Union[Node, Leaf]) -> int:
        score = math.inf
        for child_node in node.children_nodes:
            child_score = self.minimax(child_node)
            score = min(child_score, score)
            if score <= node.alpha:
                break
            node.beta = min(node.beta, score)
            child_node.score = score
        return score

    def min_cw(self, node: Union[Node, Leaf]) -> int:
        """Compute minimax value for minimising player."""
        for child_node in node.children_nodes:
            score = self.minimax(child_node)
            if score < child_node.score:
                child_node.score = score
        all_children_scores = [child_node.score for child_node in
                               node.children_nodes]
        return min(all_children_scores)


def run_experiments():
    experiment_tuples = [
        (3, 2, 2),
        (3, 2, 3),
        (3, 3, 2),
        (3, 3, 3),
        (4, 3, 2),
        (4, 3, 3),
        (4, 4, 3),
        (4, 4, 4),
    ]
    experiments_pruned = {}
    experiments_unpruned = {}
    experiment_counter = 0
    for experiment_params in tqdm(experiment_tuples):
        experiment_counter += 1
        m, n, k = experiment_params

        unpruned_game = Game(m, n, k, draw_visual=False, ab_pruning=False, simulate=True)
        unpruned_game.play()

        experiments_unpruned[experiment_params] = {
            "time_to_build_tree": unpruned_game.time_to_build_tree,
            "time_per_decision": unpruned_game.time_per_decision,
        }

        pruned_game = Game(m, n, k, draw_visual=False, ab_pruning=True,
                             simulate=True)
        pruned_game.play()
        experiments_pruned[experiment_params] = {
            "time_to_build_tree": pruned_game.time_to_build_tree,
            "time_per_decision": pruned_game.time_per_decision,
        }

    with open('experiments_unpruned.pik', 'wb') as f:
        dill.dump(experiments_unpruned, f)

    with open('experiments_pruned.pik', 'wb') as file:
        dill.dump(experiments_pruned, file)

def load_data(filename: str):
    with open(filename, 'rb') as f:
        data = dill.load(f)
    return data



if __name__ == '__main__':
    run_experiments()
    # print(load_data("experiments_pruned.pik"))
    # print(load_data("experiments_unpruned.pik"))
    # new_game = Game(4, 4, 3, draw_visual=True, ab_pruning=True, simulate=False)
    # mid_game_board = GameBoard.from_array(np.array(
    #     [
    #         [2, 0, 1, 0],
    #         [1, 0, 2, 0],
    #         [2, 2, 1, 0],
    #         [2, 2, 1, 0],
    #     ]
    # ),
    # )
    # new_game.game_board = mid_game_board
    # new_game.play()
    # # new_game.simulate_all_games()
    # # new_game.score_all_games()
    # print()
    # new_game.simulate_all_games()
