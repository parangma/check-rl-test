#!/usr/bin/env python3
"""
샘플 UCI 봇: 센터 제어 플레이어
중앙 칸을 향한 수를 선호합니다.
"""

import sys
import chess
import random

CENTER = {chess.D4, chess.D5, chess.E4, chess.E5}
EXTENDED_CENTER = {chess.C3, chess.C4, chess.C5, chess.C6,
                   chess.D3, chess.D6, chess.E3, chess.E6,
                   chess.F3, chess.F4, chess.F5, chess.F6}

def get_best_move(board: chess.Board) -> chess.Move:
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()
    scored = []
    for move in legal_moves:
        score = 0
        if move.to_square in CENTER: score += 3
        elif move.to_square in EXTENDED_CENTER: score += 1
        if board.is_capture(move): score += 2
        scored.append((score, random.random(), move))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]

def main():
    board = chess.Board()
    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if line == "uci":
            print("id name SampleCenterBot")
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
