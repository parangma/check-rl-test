#!/usr/bin/env python3
"""
라이브 토너먼트 서버
===================
HTTP 서버를 띄우고 토너먼트를 백그라운드에서 실행하면서
매 수마다 live_state.json을 업데이트합니다.

사용법:
    python live_server.py ./bots/bb/engine.py ./bots/junho/junho-chess-engine-untrained
    python live_server.py --time-per-move 5 --games-per-match 1 ./bots/*/engine.py
    # 브라우저에서 http://localhost:8080 접속
"""

import argparse
import json
import os
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# tournament.py를 같은 디렉토리에서 임포트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tournament import Tournament

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_STATE_PATH = os.path.join(SCRIPT_DIR, "live_state.json")

# 토너먼트 인스턴스 참조 (stop용)
current_tournament = None

# 라이브 상태
live_state = {
    "status": "waiting",       # waiting | playing | finished
    "game_number": 0,
    "white": "",
    "black": "",
    "opening": "",
    "moves": [],               # SAN 리스트
    "result": None,            # "1-0", "0-1", "1/2-1/2"
    "reason": None,
    "standings": [],
    "completed_games": [],     # 완료된 게임들의 PGN 문자열
}
state_lock = threading.Lock()


def save_state():
    """live_state.json에 현재 상태 저장"""
    with open(LIVE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(live_state, f, ensure_ascii=False)


def on_move(board, san, white_name, black_name, game_number, opening_name):
    """매 수마다 호출되는 콜백"""
    with state_lock:
        live_state["status"] = "playing"
        live_state["game_number"] = game_number
        live_state["white"] = white_name
        live_state["black"] = black_name
        live_state["opening"] = opening_name
        live_state["moves"].append(san)
        live_state["result"] = None
        live_state["reason"] = None
        save_state()


def on_game_end(result, scores, wins, draws, losses):
    """대국 종료 시 호출되는 콜백"""
    with state_lock:
        live_state["result"] = result["result"]
        live_state["reason"] = result["reason"]
        live_state["completed_games"].append(result["pgn"])

        # 순위 업데이트
        sorted_names = sorted(scores.keys(),
                              key=lambda n: (-scores[n], -wins[n]))
        live_state["standings"] = [
            {"name": n, "score": scores[n],
             "w": wins[n], "d": draws[n], "l": losses[n]}
            for n in sorted_names
        ]
        save_state()


def reset_game_state():
    """새 대국 시작 전 moves 초기화 (Tournament에서 play_game 호출 직전)"""
    with state_lock:
        live_state["moves"] = []
        live_state["result"] = None
        live_state["reason"] = None
        save_state()


# on_move를 래핑하여 새 대국 감지
_last_game_number = [0]

def on_move_wrapper(board, san, white_name, black_name, game_number, opening_name):
    if game_number != _last_game_number[0]:
        _last_game_number[0] = game_number
        with state_lock:
            live_state["moves"] = []
        # 오프닝 첫 수는 board.move_stack[0]에 있음
        # 첫 수(오프닝)의 SAN을 추가
        import chess
        opening_board = chess.Board()
        opening_san = opening_board.san(board.move_stack[0])
        with state_lock:
            live_state["moves"].append(opening_san)
    on_move(board, san, white_name, black_name, game_number, opening_name)


def run_tournament(bot_paths, time_per_move, max_moves, games_per_match):
    """백그라운드 스레드에서 토너먼트 실행"""
    global current_tournament

    # 참가자 이름 추출하여 초기 standings 세팅
    bot_names = [Path(p).parent.name or Path(p).stem for p in bot_paths]
    with state_lock:
        live_state["status"] = "playing"
        live_state["standings"] = [
            {"name": n, "score": 0, "w": 0, "d": 0, "l": 0}
            for n in bot_names
        ]
        live_state["completed_games"] = []
        save_state()

    tournament = Tournament(
        bot_paths, time_per_move, max_moves, games_per_match,
        on_move=on_move_wrapper,
        on_game_end=on_game_end,
    )
    current_tournament = tournament
    tournament.run()

    with state_lock:
        if live_state["status"] != "stopped":
            live_state["status"] = "finished"
        save_state()
    current_tournament = None


class LiveHandler(SimpleHTTPRequestHandler):
    """정적 파일 + /live_state.json API"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SCRIPT_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/live_state.json":
            with state_lock:
                data = json.dumps(live_state, ensure_ascii=False)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data.encode("utf-8"))
        elif self.path == "/" or self.path == "":
            self.path = "/viewer.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/stop":
            global current_tournament
            stopped = False
            if current_tournament and not current_tournament.stopped:
                current_tournament.stopped = True
                with state_lock:
                    live_state["status"] = "stopped"
                    live_state["reason"] = "사용자가 토너먼트를 중단했습니다"
                    save_state()
                stopped = True
                print("  ⏹ 토너먼트 중단 요청 수신")

            resp = json.dumps({"ok": True, "stopped": stopped}, ensure_ascii=False)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(resp.encode("utf-8"))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # 로그 숨김


def main():
    parser = argparse.ArgumentParser(description="Live Tournament Server")
    parser.add_argument("bots", nargs="+", help="UCI 봇 경로")
    parser.add_argument("--time-per-move", type=float, default=5)
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--games-per-match", type=int, default=2)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    for bot_path in args.bots:
        if not os.path.isfile(bot_path):
            print(f"봇 파일을 찾을 수 없습니다: {bot_path}")
            sys.exit(1)

    # 초기 상태 저장
    save_state()

    # 토너먼트를 백그라운드 스레드에서 시작
    t = threading.Thread(
        target=run_tournament,
        args=(args.bots, args.time_per_move, args.max_moves, args.games_per_match),
        daemon=True,
    )
    t.start()

    # HTTP 서버 시작
    server = HTTPServer(("0.0.0.0", args.port), LiveHandler)
    print(f"\n  🌐 라이브 뷰어: http://localhost:{args.port}")
    print(f"  Ctrl+C로 종료\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료")
        server.shutdown()


if __name__ == "__main__":
    main()
