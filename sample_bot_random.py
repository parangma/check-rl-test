#!/usr/bin/env python3
"""
샘플 UCI 봇: 랜덤 플레이어
테스트용으로 합법 수 중 랜덤으로 선택합니다.

이 파일을 참고하여 자신만의 RL 봇을 만드세요.
핵심: get_best_move() 함수를 자신의 모델로 교체하면 됩니다.
"""

import sys
import chess
import random


def get_best_move(board: chess.Board) -> chess.Move:
    """
    =========================================
    이 함수를 자신의 RL 모델로 교체하세요!
    =========================================

    예시 (PyTorch 모델):
        state = board_to_tensor(board)
        with torch.no_grad():
            policy, value = model(state)
        legal_mask = get_legal_mask(board)
        move = select_move(policy, legal_mask)
        return move
    """
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)


def main():
    """UCI 프로토콜 메인 루프"""
    board = chess.Board()

    while True:
        try:
            line = input().strip()
        except EOFError:
            break

        if line == "uci":
            print("id name SampleRandomBot")
            print("id author Tournament")
            print("uciok")
            sys.stdout.flush()

        elif line == "isready":
            print("readyok")
            sys.stdout.flush()

        elif line == "ucinewgame":
            board = chess.Board()

        elif line.startswith("position"):
            parts = line.split()
            if "startpos" in parts:
                board = chess.Board()
                if "moves" in parts:
                    moves_idx = parts.index("moves") + 1
                    for move_uci in parts[moves_idx:]:
                        board.push(chess.Move.from_uci(move_uci))
            elif "fen" in parts:
                fen_idx = parts.index("fen") + 1
                fen = " ".join(parts[fen_idx:fen_idx + 6])
                board = chess.Board(fen)
                if "moves" in parts:
                    moves_idx = parts.index("moves") + 1
                    for move_uci in parts[moves_idx:]:
                        board.push(chess.Move.from_uci(move_uci))

        elif line.startswith("go"):
            move = get_best_move(board)
            print(f"bestmove {move.uci()}")
            sys.stdout.flush()

        elif line == "quit":
            break


if __name__ == "__main__":
    main()
