# ♚ RL Chess Bot Tournament Runner

대회 규정 v3.0에 맞춘 자동 대국 진행 시스템입니다.

## 빠른 시작

```bash
# 1. 의존성 설치
pip install chess

# 2. 샘플 봇으로 테스트
python tournament.py sample_bot_random.py sample_bot_greedy.py sample_bot_center.py

# 3. 자신의 봇으로 대회 진행
python tournament.py ./bots/alpha/engine.py ./bots/beta/engine.py ./bots/gamma/engine.py
```

## 옵션

```bash
python tournament.py \
    --time-per-move 5 \    # 수당 제한 시간 (초, 기본: 5)
    --max-moves 300 \      # 최대 수 제한 (기본: 300)
    bot_a.py bot_b.py bot_c.py
```

## 봇 제작 가이드

`sample_bot_random.py`를 템플릿으로 사용하세요. 핵심은 단 하나:

**`get_best_move(board)` 함수를 자신의 RL 모델로 교체하면 됩니다.**

### UCI 프로토콜 최소 요구사항

봇은 stdin/stdout으로 아래 명령을 처리해야 합니다:

| 수신 명령 | 응답 |
|-----------|------|
| `uci` | `id name 봇이름` → `uciok` |
| `isready` | `readyok` |
| `ucinewgame` | (새 게임 초기화) |
| `position startpos moves e2e4 e7e5 ...` | (보드 상태 설정) |
| `go movetime 5000` | `bestmove e2e4` |
| `quit` | (종료) |

### PyTorch 모델 연동 예시

```python
import torch
import chess

model = MyChessNet()
model.load_state_dict(torch.load("model.pt"))
model.eval()

def get_best_move(board: chess.Board) -> chess.Move:
    state = board_to_tensor(board)  # 자신만의 인코딩
    with torch.no_grad():
        policy, value = model(state)

    # 합법 수 마스킹
    legal_moves = list(board.legal_moves)
    best_move = None
    best_score = -float('inf')

    for move in legal_moves:
        idx = move_to_index(move)  # 자신만의 인덱싱
        if policy[idx] > best_score:
            best_score = policy[idx]
            best_move = move

    return best_move
```

## 출력 파일

대회 종료 후 `tournament_YYYYMMDD_HHMMSS/` 폴더에 저장됩니다:

- `all_games.pgn` — 모든 대국 기보 (체스 뷰어에서 재생 가능)
- `results.json` — 순위, 승점, 대국별 상세 결과

## 파일 구조

```
tournament.py           # 대회 진행 스크립트 (이것만 있으면 됨)
sample_bot_random.py    # 샘플: 랜덤 봇
sample_bot_greedy.py    # 샘플: 탐욕적 캡처 봇
sample_bot_center.py    # 샘플: 중앙 제어 봇
```
