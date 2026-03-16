"""
RL Chess Bot Tournament Runner v1.0
====================================
대회 규정 v3.0에 맞춘 자동 대국 진행 시스템

사용법:
    python tournament.py bot_a.py bot_b.py bot_c.py
    python tournament.py --time-per-move 5 --max-moves 300 bot_a.py bot_b.py bot_c.py

봇 요구사항:
    - UCI 프로토콜 호환 실행 파일 (.py 또는 바이너리)
    - stdin/stdout으로 UCI 명령 송수신
    - Python 봇의 경우: #!/usr/bin/env python3 shebang 포함 권장

예시 봇 실행:
    python tournament.py ./bots/alpha/engine.py ./bots/beta/engine.py ./bots/gamma/engine.py
"""

import chess
import chess.pgn
import subprocess
import time
import random
import sys
import os
import signal
import json
from itertools import combinations
from datetime import datetime
from pathlib import Path


# ============================================================
# 설정
# ============================================================
DEFAULT_TIME_PER_MOVE = 5       # 수당 제한 시간 (초)
DEFAULT_MAX_MOVES = 300         # 최대 수 제한 (강제 무승부)
OPENING_FIRST_MOVES = {
    "e4":  chess.Move.from_uci("e2e4"),   # King's Pawn
    "d4":  chess.Move.from_uci("d2d4"),   # Queen's Pawn
    "c4":  chess.Move.from_uci("c2c4"),   # English Opening
    "Nf3": chess.Move.from_uci("g1f3"),   # Réti Opening
}


# ============================================================
# UCI 엔진 래퍼
# ============================================================
class UCIEngine:
    """UCI 프로토콜로 체스 봇과 통신하는 래퍼 클래스"""

    def __init__(self, path: str, name: str = None):
        self.path = path
        self.name = name or Path(path).stem
        self.process = None

    def start(self):
        """엔진 프로세스 시작 및 UCI 초기화"""
        cmd = [sys.executable, "-u", self.path] if self.path.endswith(".py") else [self.path]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok", timeout=10)
        self._send("isready")
        self._wait_for("readyok", timeout=10)

    def stop(self):
        """엔진 프로세스 종료"""
        if self.process and self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()

    def new_game(self):
        """새 게임 초기화"""
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", timeout=10)

    def get_move(self, board: chess.Board, time_limit: float) -> tuple:
        """
        현재 보드 상태에서 최선의 수를 요청.
        Returns: (move, elapsed_time) 또는 시간 초과 시 (None, elapsed_time)
        """
        # position 명령 전송
        moves_str = " ".join(m.uci() for m in board.move_stack)
        if moves_str:
            self._send(f"position startpos moves {moves_str}")
        else:
            self._send("position startpos")

        # go 명령 전송 (movetime = 밀리초)
        movetime_ms = int(time_limit * 1000)
        self._send(f"go movetime {movetime_ms}")

        # bestmove 응답 대기
        start_time = time.time()
        move_uci = None

        while True:
            elapsed = time.time() - start_time
            # 여유 시간 1초 추가 (프로세스 오버헤드 고려)
            if elapsed > time_limit + 1.0:
                break

            line = self._readline(timeout=time_limit + 1.0 - elapsed)
            if line is None:
                break

            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    move_uci = parts[1]
                break

        elapsed = time.time() - start_time

        if move_uci and move_uci != "(none)":
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    return move, elapsed
            except ValueError:
                pass

        return None, elapsed

    def _send(self, command: str):
        """엔진에 명령 전송"""
        if self.process and self.process.poll() is None:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()

    def _readline(self, timeout: float = 10) -> str:
        """엔진으로부터 한 줄 읽기 (타임아웃 포함)"""
        import threading

        result = [None]

        def read():
            try:
                line = self.process.stdout.readline()
                if line:
                    result[0] = line.strip()
            except Exception:
                pass

        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout=timeout)
        return result[0]

    def _wait_for(self, expected: str, timeout: float = 10):
        """특정 응답이 올 때까지 대기"""
        start = time.time()
        while time.time() - start < timeout:
            line = self._readline(timeout=timeout - (time.time() - start))
            if line and expected in line:
                return True
        raise TimeoutError(f"[{self.name}] '{expected}' 응답 대기 시간 초과 ({timeout}초)")


# ============================================================
# 대국 진행
# ============================================================
def play_game(
    white_engine: UCIEngine,
    black_engine: UCIEngine,
    opening_move: chess.Move,
    opening_name: str,
    time_per_move: float,
    max_moves: int,
    game_number: int,
    on_move=None,
    stop_check=None,
) -> dict:
    """
    두 엔진 간 한 판의 대국을 진행합니다.

    Returns:
        dict with keys: result, reason, pgn, white, black, opening,
                        moves_count, white_time_total, black_time_total
    """
    board = chess.Board()

    # 오프닝 첫 수 적용
    board.push(opening_move)

    white_total_time = 0.0
    black_total_time = 0.0
    move_log = []

    # PGN 초기화
    game = chess.pgn.Game()
    game.headers["Event"] = "RL Chess Bot Tournament"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_number)
    game.headers["White"] = white_engine.name
    game.headers["Black"] = black_engine.name
    game.headers["Opening"] = f"{opening_name} (1. {board.move_stack[0]})"

    print(f"    오프닝: 1. {opening_name} ({opening_move.uci()})")
    print(f"    ", end="", flush=True)

    node = game.add_variation(opening_move)

    while not board.is_game_over(claim_draw=True):
        # 중단 체크
        if stop_check and stop_check():
            result = "1/2-1/2"
            reason = "토너먼트 중단 (무승부 처리)"
            print(f"\n    ⏹ {reason}")
            game.headers["Result"] = result
            game.headers["Termination"] = reason
            return _make_result(result, reason, game, board, white_engine, black_engine,
                                opening_name, white_total_time, black_total_time)

        # 최대 수 제한 체크
        if board.fullmove_number > max_moves:
            result = "1/2-1/2"
            reason = f"최대 수 제한 ({max_moves}수 초과, 강제 무승부)"
            print(f"\n    ⏹ {reason}")
            game.headers["Result"] = result
            game.headers["Termination"] = reason
            return _make_result(result, reason, game, board, white_engine, black_engine,
                                opening_name, white_total_time, black_total_time)

        # 현재 턴의 엔진 선택
        is_white_turn = board.turn == chess.WHITE
        current_engine = white_engine if is_white_turn else black_engine
        side = "W" if is_white_turn else "B"

        # 수 요청
        move, elapsed = current_engine.get_move(board, time_per_move)

        if is_white_turn:
            white_total_time += elapsed
        else:
            black_total_time += elapsed

        # 시간 초과 또는 불법 수 처리
        if move is None:
            # 시간 초과 시 랜덤 합법 수 선택
            legal_moves = list(board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
                print(f"⏰", end="", flush=True)
            else:
                break

        # 수 진행
        san = board.san(move)
        board.push(move)
        node = node.add_variation(move)

        # 라이브 콜백
        if on_move:
            on_move(board, san, white_engine.name, black_engine.name,
                    game_number, opening_name)

        # 진행 표시 (간단하게)
        if is_white_turn:
            print(f"{board.fullmove_number - 1}.{san} ", end="", flush=True)
        else:
            print(f"{san} ", end="", flush=True)

        # 50수마다 줄바꿈
        if board.fullmove_number % 25 == 0 and not is_white_turn:
            print(f"\n    ", end="", flush=True)

    # 게임 종료 판정
    result, reason = _determine_result(board)
    print(f"\n    {'🏆' if result != '1/2-1/2' else '🤝'} {reason}")

    game.headers["Result"] = result
    game.headers["Termination"] = reason

    return _make_result(result, reason, game, board, white_engine, black_engine,
                        opening_name, white_total_time, black_total_time)


def _determine_result(board: chess.Board) -> tuple:
    """게임 종료 사유 판정"""
    if board.is_checkmate():
        winner = "백" if board.turn == chess.BLACK else "흑"
        result = "1-0" if board.turn == chess.BLACK else "0-1"
        return result, f"체크메이트 ({winner} 승리)"
    elif board.is_stalemate():
        return "1/2-1/2", "스테일메이트 (무승부)"
    elif board.is_insufficient_material():
        return "1/2-1/2", "불충분 기물 (무승부)"
    elif board.can_claim_threefold_repetition():
        return "1/2-1/2", "3회 반복 (무승부)"
    elif board.can_claim_fifty_moves():
        return "1/2-1/2", "50수 규칙 (무승부)"
    else:
        return "1/2-1/2", "기타 무승부"


def _make_result(result, reason, game, board, white_engine, black_engine,
                 opening, white_time, black_time) -> dict:
    """결과 딕셔너리 생성"""
    return {
        "result": result,
        "reason": reason,
        "pgn": str(game),
        "white": white_engine.name,
        "black": black_engine.name,
        "opening": opening,
        "moves_count": board.fullmove_number,
        "white_time_total": round(white_time, 2),
        "black_time_total": round(black_time, 2),
    }


# ============================================================
# 토너먼트 운영
# ============================================================
class Tournament:
    """라운드 로빈 토너먼트 매니저"""

    def __init__(self, bot_paths: list, time_per_move: float, max_moves: int,
                 games_per_match: int = 2, on_move=None, on_game_end=None):
        self.engines = []
        for path in bot_paths:
            name = Path(path).parent.name or Path(path).stem
            self.engines.append(UCIEngine(path, name))
        self.time_per_move = time_per_move
        self.max_moves = max_moves
        self.games_per_match = games_per_match
        self.on_move = on_move
        self.on_game_end = on_game_end
        self.stopped = False
        self.results = []
        self.scores = {e.name: 0.0 for e in self.engines}
        self.wins = {e.name: 0 for e in self.engines}
        self.draws = {e.name: 0 for e in self.engines}
        self.losses = {e.name: 0 for e in self.engines}

    def run(self):
        """전체 토너먼트 진행"""
        print("=" * 60)
        print("  ♚ RL Chess Bot Tournament ♚")
        print("=" * 60)
        print(f"  참가 봇: {', '.join(e.name for e in self.engines)}")
        print(f"  수당 시간: {self.time_per_move}초")
        print(f"  최대 수: {self.max_moves}")
        total_games = len(list(combinations(self.engines, 2))) * self.games_per_match
        print(f"  매치당 대국: {self.games_per_match}판")
        print(f"  총 대국 수: {total_games}판")
        print("=" * 60)

        # 엔진 초기화
        print("\n[엔진 초기화]")
        for engine in self.engines:
            try:
                engine.start()
                print(f"  ✅ {engine.name} 준비 완료")
            except Exception as e:
                print(f"  ❌ {engine.name} 시작 실패: {e}")
                self._cleanup()
                return

        # 라운드 로빈 진행
        round_num = 0
        matchups = list(combinations(self.engines, 2))

        for i, (engine_a, engine_b) in enumerate(matchups):
            if self.stopped:
                print("\n  ⏹ 토너먼트가 중단되었습니다.")
                break
            round_num += 1
            print(f"\n{'─' * 60}")
            print(f"  라운드 {round_num}: {engine_a.name} vs {engine_b.name}")
            print(f"{'─' * 60}")

            # 랜덤 오프닝 선택 (같은 매치업의 두 판은 같은 오프닝)
            opening_name = random.choice(list(OPENING_FIRST_MOVES.keys()))
            opening_move = OPENING_FIRST_MOVES[opening_name]

            # 매치당 games_per_match 판 진행
            pairings = [(engine_a, engine_b), (engine_b, engine_a)]
            for g in range(self.games_per_match):
                if self.stopped:
                    break
                white, black = pairings[g % 2]
                game_num = i * self.games_per_match + g + 1
                print(f"\n  📋 대국 {game_num}: {white.name}(백) vs {black.name}(흑)")
                engine_a.new_game()
                engine_b.new_game()
                result = play_game(
                    white, black, opening_move, opening_name,
                    self.time_per_move, self.max_moves, game_num,
                    on_move=self.on_move,
                    stop_check=lambda: self.stopped,
                )
                self.results.append(result)
                self._update_scores(result)
                if self.on_game_end:
                    self.on_game_end(result, self.scores, self.wins,
                                    self.draws, self.losses)

            # 라운드 중간 스코어
            if not self.stopped:
                self._print_interim_scores()

        # 최종 결과
        self._print_final_results()
        self._save_results()
        self._cleanup()

    def _update_scores(self, result: dict):
        """승점 업데이트"""
        white = result["white"]
        black = result["black"]
        r = result["result"]

        if r == "1-0":
            self.scores[white] += 1.0
            self.wins[white] += 1
            self.losses[black] += 1
        elif r == "0-1":
            self.scores[black] += 1.0
            self.wins[black] += 1
            self.losses[white] += 1
        else:
            self.scores[white] += 0.5
            self.scores[black] += 0.5
            self.draws[white] += 1
            self.draws[black] += 1

    def _print_interim_scores(self):
        """중간 스코어 출력"""
        print(f"\n  📊 현재 순위:")
        sorted_scores = sorted(self.scores.items(), key=lambda x: (-x[1], -self.wins[x[0]]))
        for rank, (name, score) in enumerate(sorted_scores, 1):
            w, d, l = self.wins[name], self.draws[name], self.losses[name]
            print(f"     {rank}. {name}: {score}점 ({w}승 {d}무 {l}패)")

    def _print_final_results(self):
        """최종 결과 출력"""
        print(f"\n{'=' * 60}")
        print(f"  🏆 최종 결과")
        print(f"{'=' * 60}")

        sorted_scores = sorted(
            self.scores.items(),
            key=lambda x: (-x[1], -self.wins[x[0]])
        )

        medals = ["🥇", "🥈", "🥉"]
        for rank, (name, score) in enumerate(sorted_scores):
            w, d, l = self.wins[name], self.draws[name], self.losses[name]
            medal = medals[rank] if rank < len(medals) else "  "
            print(f"  {medal} {rank + 1}위: {name} — {score}점 ({w}승 {d}무 {l}패)")

        print(f"\n{'─' * 60}")
        print(f"  대국 상세:")
        for i, result in enumerate(self.results, 1):
            r = result["result"]
            symbol = {"1-0": "⬜", "0-1": "⬛", "1/2-1/2": "🤝"}[r]
            print(f"  {symbol} 대국 {i}: {result['white']}(백) vs {result['black']}(흑) "
                  f"= {r} [{result['opening']}] ({result['moves_count']}수)")
        print(f"{'=' * 60}")

    def _save_results(self):
        """PGN 및 JSON 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"tournament_{timestamp}")
        output_dir.mkdir(exist_ok=True)

        # PGN 저장
        pgn_path = output_dir / "all_games.pgn"
        with open(pgn_path, "w") as f:
            for result in self.results:
                f.write(result["pgn"])
                f.write("\n\n")
        print(f"\n  📁 PGN 저장: {pgn_path}")

        # JSON 결과 저장
        summary = {
            "tournament": "RL Chess Bot Tournament",
            "date": datetime.now().isoformat(),
            "settings": {
                "time_per_move": self.time_per_move,
                "max_moves": self.max_moves,
            },
            "standings": [
                {
                    "rank": rank + 1,
                    "name": name,
                    "score": score,
                    "wins": self.wins[name],
                    "draws": self.draws[name],
                    "losses": self.losses[name],
                }
                for rank, (name, score) in enumerate(
                    sorted(self.scores.items(), key=lambda x: (-x[1], -self.wins[x[0]]))
                )
            ],
            "games": [
                {
                    "game": i + 1,
                    "white": r["white"],
                    "black": r["black"],
                    "result": r["result"],
                    "reason": r["reason"],
                    "opening": r["opening"],
                    "moves": r["moves_count"],
                    "white_time": r["white_time_total"],
                    "black_time": r["black_time_total"],
                }
                for i, r in enumerate(self.results)
            ],
        }

        json_path = output_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  📁 결과 저장: {json_path}")

    def _cleanup(self):
        """모든 엔진 종료"""
        for engine in self.engines:
            engine.stop()


# ============================================================
# 메인
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RL Chess Bot Tournament Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python tournament.py bot_a.py bot_b.py bot_c.py
  python tournament.py --time-per-move 5 --max-moves 300 ./bots/*/engine.py
        """,
    )
    parser.add_argument("bots", nargs="+", help="UCI 호환 봇 실행 파일 경로 (2~10개)")
    parser.add_argument("--time-per-move", type=float, default=DEFAULT_TIME_PER_MOVE,
                        help=f"수당 제한 시간 (초, 기본값: {DEFAULT_TIME_PER_MOVE})")
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES,
                        help=f"최대 수 제한 (기본값: {DEFAULT_MAX_MOVES})")
    parser.add_argument("--games-per-match", type=int, default=2,
                        help="매치당 대국 수 (기본값: 2)")

    args = parser.parse_args()

    if len(args.bots) < 2:
        parser.error("최소 2개의 봇이 필요합니다.")

    # 봇 파일 존재 확인
    for bot_path in args.bots:
        if not os.path.isfile(bot_path):
            parser.error(f"봇 파일을 찾을 수 없습니다: {bot_path}")

    tournament = Tournament(args.bots, args.time_per_move, args.max_moves, args.games_per_match)
    tournament.run()


if __name__ == "__main__":
    main()
