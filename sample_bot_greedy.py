#!/usr/bin/env python3
"""
샘플 UCI 봇: 탐욕적 캡처 플레이어
기물을 잡을 수 있으면 잡고, 아니면 랜덤으로 둡니다.
"""

import sys
import chess
import random

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

def get_best_move(board: chess.Board) -> chess.Move:
    legal_moves = list(board.legal_moves)
    # 체크메이트 수 우선
    for move in legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()
    # 캡처 수를 가치순으로 정렬
    captures = []
    for move in legal_moves:
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            value = PIECE_VALUES.get(victim.piece_type, 0) if victim else 1
            captures.append((value, move))
    if captures:
        captures.sort(key=lambda x: -x[0])
        return captures[0][1]
    return random.choice(legal_moves)

def main():
    board = chess.Board()
    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if line == "uci":
            print("id name SampleGreedyBot")
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
        elif line.startswith("go"):
            move = get_best_move(board)
            print(f"bestmove {move.uci()}")
            sys.stdout.flush()
        elif line == "quit":
            break

if __name__ == "__main__":
    main()
