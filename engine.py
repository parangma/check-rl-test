#!/usr/bin/env python3
"""
Dalbit Zero - AlphaZero Chess UCI Engine
=========================================
대회 제출용 자체 완결형(self-contained) UCI 봇.
dalbit_zero 프로젝트의 모델을 로드하여 MCTS 탐색으로 수를 선택합니다.

사용법:
    python engine.py                          # 기본 설정 (model.pt, MCTS 사용)
    python engine.py --weights my_model.pt    # 커스텀 가중치
    python engine.py --no-mcts                # Policy-only (MCTS 비활성)

대회 제출:
    이 파일(engine.py)과 모델 가중치(model.pt)를 같은 디렉토리에 놓으세요.
"""

import sys
import os
import time
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

# ============================================================
# 설정
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(SCRIPT_DIR, "model.pt")
BOT_NAME = "DalbitZero"

# MCTS 설정
USE_MCTS = True
MCTS_C_PUCT = 1.4
MCTS_DIRICHLET_ALPHA = 0.03
MCTS_DIRICHLET_EPSILON = 0.0  # 대회에서는 노이즈 없이 순수 탐색
MCTS_BATCH_SIZE = 8
MCTS_VIRTUAL_LOSS = 3.0
MCTS_TIME_MARGIN = 0.3  # 시간 제한에서 여유분 (초)


# ============================================================
# 모델 아키텍처 (dalbit_zero/model/network.py 동일)
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, num_channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(num_channels, num_channels // reduction)
        self.fc2 = nn.Linear(num_channels // reduction, num_channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = x.mean(dim=[2, 3])
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, num_channels, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.se = SEBlock(num_channels) if use_se else None

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.se is not None:
            x = self.se(x)
        x += residual
        return F.relu(x)


class AlphaZeroNet(nn.Module):
    def __init__(self, num_res_blocks, num_channels, action_size=4672,
                 in_channels=18, use_se=False, policy_channels=2,
                 value_channels=1, value_hidden=64):
        super().__init__()
        self.action_size = action_size
        board_size = 8

        self.initial_conv = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.initial_bn = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_channels, use_se=use_se) for _ in range(num_res_blocks)]
        )

        # Policy Head
        self.policy_conv = nn.Conv2d(num_channels, policy_channels, 1)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels * board_size * board_size, action_size)

        # Value Head
        self.value_conv = nn.Conv2d(num_channels, value_channels, 1)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * board_size * board_size, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, x):
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, encoded_state):
        """단일 추론: (C,H,W) numpy → (policy_probs, value)"""
        self.eval()
        with torch.no_grad():
            t = torch.FloatTensor(encoded_state).unsqueeze(0).to(self._device)
            p, v = self.forward(t)
            return F.softmax(p, dim=1).cpu().numpy()[0], v.cpu().item()

    def predict_batch(self, encoded_states):
        """배치 추론: (N,C,H,W) numpy → (policies, values)"""
        self.eval()
        with torch.no_grad():
            t = torch.FloatTensor(encoded_states).to(self._device)
            p, v = self.forward(t)
            return F.softmax(p, dim=1).cpu().numpy(), v.cpu().numpy().flatten()


def load_model(weights_path):
    """체크포인트에서 모델 아키텍처를 자동 감지하여 로드"""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    # 아키텍처 자동 감지
    num_channels = sd["initial_conv.weight"].shape[0]
    in_channels = sd["initial_conv.weight"].shape[1]
    use_se = any("se." in k for k in sd.keys())
    num_res_blocks = len(set(
        int(k.split(".")[1]) for k in sd.keys() if k.startswith("res_blocks.")
    ))
    policy_channels = sd["policy_conv.weight"].shape[0]
    value_channels = sd["value_conv.weight"].shape[0]
    value_hidden = sd["value_fc1.weight"].shape[0]
    action_size = sd["policy_fc.weight"].shape[0]

    model = AlphaZeroNet(
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        action_size=action_size,
        in_channels=in_channels,
        use_se=use_se,
        policy_channels=policy_channels,
        value_channels=value_channels,
        value_hidden=value_hidden,
    )
    model.load_state_dict(sd)
    model.eval()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model._device = device
    model.to(device)

    info = (f"{num_res_blocks}blocks, {num_channels}ch, "
            f"SE={'on' if use_se else 'off'}, device={device}")
    return model, info


# ============================================================
# 체스 인코딩 (dalbit_zero/game/chess_game.py 동일)
# ============================================================
PIECE_TO_INT = {
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
    chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
}

QUEEN_DIRECTIONS = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
]
KNIGHT_MOVES = [
    (-2, -1), (-2, 1), (-1, 2), (1, 2),
    (2, 1), (2, -1), (1, -2), (-1, -2),
]
UNDERPROMO_DIRS = [(-1, -1), (-1, 0), (-1, 1)]
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

ACTION_SIZE = 4672  # 64 * 73


def _square_to_rowcol(sq):
    return 7 - chess.square_rank(sq), chess.square_file(sq)


def _rowcol_to_square(row, col):
    return chess.square(col, 7 - row)


def encode_action(move, player):
    """chess.Move → canonical action index"""
    from_sq = move.from_square
    to_sq = move.to_square

    if player == -1:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    from_row, from_col = _square_to_rowcol(from_sq)
    to_row, to_col = _square_to_rowcol(to_sq)
    dr = to_row - from_row
    dc = to_col - from_col

    promotion = move.promotion

    if promotion and promotion != chess.QUEEN:
        norm_dr = -1 if dr < 0 else (1 if dr > 0 else 0)
        norm_dc = -1 if dc < 0 else (1 if dc > 0 else 0)
        dir_idx = UNDERPROMO_DIRS.index((norm_dr, norm_dc))
        piece_idx = UNDERPROMO_PIECES.index(promotion)
        move_type = 64 + dir_idx * 3 + piece_idx
    elif (abs(dr), abs(dc)) in {(2, 1), (1, 2)}:
        move_type = 56 + KNIGHT_MOVES.index((dr, dc))
    else:
        norm_dr = 0 if dr == 0 else (dr // abs(dr))
        norm_dc = 0 if dc == 0 else (dc // abs(dc))
        dir_idx = QUEEN_DIRECTIONS.index((norm_dr, norm_dc))
        distance = max(abs(dr), abs(dc))
        move_type = dir_idx * 7 + (distance - 1)

    return (from_row * 8 + from_col) * 73 + move_type


def decode_action(action, board):
    """canonical action index → chess.Move"""
    player = 1 if board.turn == chess.WHITE else -1

    from_idx = action // 73
    move_type = action % 73
    from_row = from_idx // 8
    from_col = from_idx % 8

    promotion = None

    if move_type < 56:
        dir_idx = move_type // 7
        distance = move_type % 7 + 1
        dr, dc = QUEEN_DIRECTIONS[dir_idx]
        to_row = from_row + dr * distance
        to_col = from_col + dc * distance
    elif move_type < 64:
        knight_idx = move_type - 56
        dr, dc = KNIGHT_MOVES[knight_idx]
        to_row = from_row + dr
        to_col = from_col + dc
    else:
        up_idx = move_type - 64
        dir_idx = up_idx // 3
        piece_idx = up_idx % 3
        dr, dc = UNDERPROMO_DIRS[dir_idx]
        to_row = from_row + dr
        to_col = from_col + dc
        promotion = UNDERPROMO_PIECES[piece_idx]

    if player == -1:
        from_row = 7 - from_row
        to_row = 7 - to_row

    from_sq = _rowcol_to_square(from_row, from_col)
    to_sq = _rowcol_to_square(to_row, to_col)

    if promotion is None:
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            target_rank = 7 if piece.color == chess.WHITE else 0
            if chess.square_rank(to_sq) == target_rank:
                promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


def board_to_state(board):
    """chess.Board → (9, 8) numpy array"""
    state = np.zeros((9, 8), dtype=np.float32)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            row, col = _square_to_rowcol(sq)
            sign = 1 if piece.color == chess.WHITE else -1
            state[row, col] = sign * PIECE_TO_INT[piece.piece_type]

    state[8, 0] = float(board.has_kingside_castling_rights(chess.WHITE))
    state[8, 1] = float(board.has_queenside_castling_rights(chess.WHITE))
    state[8, 2] = float(board.has_kingside_castling_rights(chess.BLACK))
    state[8, 3] = float(board.has_queenside_castling_rights(chess.BLACK))
    state[8, 4] = float(chess.square_file(board.ep_square)) if board.ep_square else -1.0
    state[8, 5] = float(board.halfmove_clock)
    state[8, 6] = float(board.fullmove_number)
    state[8, 7] = 1.0 if board.turn == chess.WHITE else -1.0

    return state


def get_canonical_state(state, player):
    """현재 플레이어 관점으로 보드 정규화"""
    if player == 1:
        return state.copy()

    canonical = np.zeros_like(state)
    canonical[:8] = -np.flipud(state[:8])
    canonical[8, 0] = state[8, 2]
    canonical[8, 1] = state[8, 3]
    canonical[8, 2] = state[8, 0]
    canonical[8, 3] = state[8, 1]
    canonical[8, 4] = state[8, 4]
    canonical[8, 5] = state[8, 5]
    canonical[8, 6] = state[8, 6]
    canonical[8, 7] = 1.0

    return canonical


def encode_state(state):
    """(9, 8) state → (18, 8, 8) 신경망 입력"""
    encoded = np.zeros((18, 8, 8), dtype=np.float32)
    board = state[:8]

    for i, pv in enumerate([1, 2, 3, 4, 5, 6]):
        encoded[i] = (board == pv).astype(np.float32)
    for i, pv in enumerate([1, 2, 3, 4, 5, 6]):
        encoded[6 + i] = (board == -pv).astype(np.float32)
    for i in range(4):
        encoded[12 + i] = state[8, i]

    ep_col = int(state[8, 4])
    if ep_col >= 0:
        encoded[16, 2, ep_col] = 1.0

    encoded[17] = 1.0
    return encoded


def get_valid_moves_mask(board):
    """현재 보드에서 합법 수 마스크 (canonical 관점)"""
    valid = np.zeros(ACTION_SIZE, dtype=np.uint8)
    player = 1 if board.turn == chess.WHITE else -1
    for move in board.legal_moves:
        action = encode_action(move, player)
        valid[action] = 1
    return valid


def get_next_state(state, action, player):
    """state + action → next_state"""
    board = state_to_board(state)
    move = decode_action(action, board)
    board.push(move)
    return board_to_state(board)


def state_to_board(state):
    """(9, 8) numpy → chess.Board"""
    INT_TO_PIECE = {1: chess.PAWN, 2: chess.KNIGHT, 3: chess.BISHOP,
                    4: chess.ROOK, 5: chess.QUEEN, 6: chess.KING}
    board = chess.Board(fen=None)
    board.clear()

    for row in range(8):
        for col in range(8):
            val = int(state[row, col])
            if val != 0:
                color = chess.WHITE if val > 0 else chess.BLACK
                piece_type = INT_TO_PIECE[abs(val)]
                sq = _rowcol_to_square(row, col)
                board.set_piece_at(sq, chess.Piece(piece_type, color))

    board.castling_rights = chess.BB_EMPTY
    if state[8, 0]: board.castling_rights |= chess.BB_H1
    if state[8, 1]: board.castling_rights |= chess.BB_A1
    if state[8, 2]: board.castling_rights |= chess.BB_H8
    if state[8, 3]: board.castling_rights |= chess.BB_A8

    ep_col = int(state[8, 4])
    if ep_col >= 0:
        if state[8, 7] > 0:
            board.ep_square = chess.square(ep_col, 5)
        else:
            board.ep_square = chess.square(ep_col, 2)
    else:
        board.ep_square = None

    board.halfmove_clock = int(state[8, 5])
    board.fullmove_number = max(1, int(state[8, 6]))
    board.turn = chess.WHITE if state[8, 7] > 0 else chess.BLACK

    return board


def is_terminal(state):
    """(terminal, winner) — winner: 1=white, -1=black, 0=draw"""
    board = state_to_board(state)
    if board.is_checkmate():
        return True, (-1 if board.turn == chess.WHITE else 1)
    if board.is_stalemate() or board.is_insufficient_material() or board.halfmove_clock >= 100:
        return True, 0
    return False, 0


# ============================================================
# MCTS (dalbit_zero/mcts/mcts.py 기반 — BatchedMCTS)
# ============================================================
class MCTSNode:
    __slots__ = ('state', 'player', 'parent', 'action', 'prior',
                 'children', 'visit_count', 'value_sum')

    def __init__(self, state, player, parent=None, action=None, prior=0.0):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def is_expanded(self):
        return len(self.children) > 0

    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class BatchedMCTS:
    def __init__(self, model, batch_size=MCTS_BATCH_SIZE,
                 c_puct=MCTS_C_PUCT, virtual_loss=MCTS_VIRTUAL_LOSS):
        self.model = model
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss

    def search(self, state, player, time_limit=None, num_sims=200):
        root = MCTSNode(state, player)

        # 루트 확장
        value = self._expand_node(root)
        self._backup(root, value)

        if MCTS_DIRICHLET_EPSILON > 0:
            self._add_dirichlet_noise(root)

        if time_limit is not None and time_limit > 0:
            deadline = time.monotonic() + time_limit
            while time.monotonic() < deadline:
                sims_done = self._run_batch(root)
                if sims_done == 0:
                    break
        else:
            done = 0
            target = num_sims - 1
            while done < target:
                batch = min(self.batch_size, target - done)
                sims_done = self._run_batch(root, max_leaves=batch)
                done += sims_done
                if sims_done == 0:
                    break

        return root

    def _run_batch(self, root, max_leaves=None):
        k = max_leaves or self.batch_size
        leaves = []
        paths = []
        terminal_results = []

        for _ in range(k):
            node = root
            path = [node]

            while node.is_expanded():
                node = self._select_child(node)
                path.append(node)

            ended, winner = is_terminal(node.state)
            if ended:
                value = 0.0 if winner == 0 else (1.0 if winner == node.player else -1.0)
                terminal_results.append((node, path, value))
                self._apply_virtual_loss(path)
            else:
                leaves.append(node)
                paths.append(path)
                self._apply_virtual_loss(path)

        for node, path, value in terminal_results:
            self._remove_virtual_loss(path)
            self._backup(node, value)

        if not leaves:
            return len(terminal_results)

        encoded_batch = []
        for leaf in leaves:
            canonical = get_canonical_state(leaf.state, leaf.player)
            encoded = encode_state(canonical)
            encoded_batch.append(encoded)

        policies, values = self.model.predict_batch(np.array(encoded_batch))

        for i, (leaf, path) in enumerate(zip(leaves, paths)):
            self._remove_virtual_loss(path)
            self._expand_with_policy(leaf, policies[i])
            self._backup(leaf, float(values[i]))

        return len(leaves) + len(terminal_results)

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        sqrt_parent = math.sqrt(node.visit_count)
        for child in node.children.values():
            q = -child.q_value()
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _apply_virtual_loss(self, path):
        for node in path:
            node.visit_count += 1
            node.value_sum -= self.virtual_loss

    def _remove_virtual_loss(self, path):
        for node in path:
            node.visit_count -= 1
            node.value_sum += self.virtual_loss

    def _expand_node(self, node):
        canonical = get_canonical_state(node.state, node.player)
        encoded = encode_state(canonical)
        policy, value = self.model.predict(encoded)

        valid = get_valid_moves_mask(state_to_board(node.state))
        policy = policy * valid
        ps = policy.sum()
        policy = policy / ps if ps > 0 else valid / valid.sum()

        for action in range(ACTION_SIZE):
            if valid[action]:
                next_state = get_next_state(node.state, action, node.player)
                child = MCTSNode(next_state, -node.player, parent=node,
                                 action=action, prior=policy[action])
                node.children[action] = child
        return value

    def _expand_with_policy(self, node, policy):
        canonical = get_canonical_state(node.state, node.player)
        valid = get_valid_moves_mask(state_to_board(node.state))

        policy = policy * valid
        ps = policy.sum()
        policy = policy / ps if ps > 0 else valid / valid.sum()

        for action in range(ACTION_SIZE):
            if valid[action]:
                next_state = get_next_state(node.state, action, node.player)
                child = MCTSNode(next_state, -node.player, parent=node,
                                 action=action, prior=policy[action])
                node.children[action] = child

    def _backup(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def _add_dirichlet_noise(self, node):
        if not node.children:
            return
        actions = list(node.children.keys())
        noise = np.random.dirichlet([MCTS_DIRICHLET_ALPHA] * len(actions))
        eps = MCTS_DIRICHLET_EPSILON
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1 - eps) * child.prior + eps * noise[i]


# ============================================================
# 수 선택 엔진
# ============================================================
class DalbitEngine:
    def __init__(self, model, use_mcts=True):
        self.model = model
        self.use_mcts = use_mcts
        if use_mcts:
            self.mcts = BatchedMCTS(model)

    def get_best_move(self, board, time_limit=5.0):
        """현재 보드에서 최선의 수를 반환"""
        state = board_to_state(board)
        player = 1 if board.turn == chess.WHITE else -1

        if self.use_mcts:
            # 시간 기반 MCTS 탐색
            search_time = max(0.1, time_limit - MCTS_TIME_MARGIN)
            root = self.mcts.search(state, player, time_limit=search_time)

            # 방문 횟수 기반 최선 수 선택 (temperature=0)
            best_action = None
            best_visits = -1
            for action, child in root.children.items():
                if child.visit_count > best_visits:
                    best_visits = child.visit_count
                    best_action = action

            if best_action is not None:
                return decode_action(best_action, board)
        else:
            # Policy-only: 신경망 출력만 사용
            canonical = get_canonical_state(state, player)
            encoded = encode_state(canonical)
            policy, _ = self.model.predict(encoded)

            valid = get_valid_moves_mask(board)
            policy = policy * valid
            best_action = np.argmax(policy)

            if valid[best_action]:
                return decode_action(best_action, board)

        # 폴백: 합법 수 중 랜덤
        import random
        return random.choice(list(board.legal_moves))


# ============================================================
# UCI 프로토콜 핸들러
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Dalbit Zero UCI Engine")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="모델 가중치 경로")
    parser.add_argument("--no-mcts", action="store_true", help="MCTS 비활성 (Policy-only)")
    args = parser.parse_args()

    engine = None  # lazy load
    board = chess.Board()
    move_time = 5.0

    while True:
        try:
            line = input().strip()
        except EOFError:
            break

        if not line:
            continue

        if line == "uci":
            print(f"id name {BOT_NAME}")
            print(f"id author DalbitZero Team")
            print("uciok")
            sys.stdout.flush()

        elif line == "isready":
            # 모델을 isready 시점에 로드 (uci 핸드셰이크 이후)
            if engine is None:
                model, info = load_model(args.weights)
                engine = DalbitEngine(model, use_mcts=not args.no_mcts)
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
                fen_parts = parts[fen_idx:fen_idx + 6]
                fen = " ".join(fen_parts)
                board = chess.Board(fen)
                if "moves" in parts:
                    moves_idx = parts.index("moves") + 1
                    for move_uci in parts[moves_idx:]:
                        board.push(chess.Move.from_uci(move_uci))

        elif line.startswith("go"):
            parts = line.split()
            if "movetime" in parts:
                mt_idx = parts.index("movetime") + 1
                move_time = int(parts[mt_idx]) / 1000.0
            else:
                move_time = 5.0

            move = engine.get_best_move(board, time_limit=move_time)
            print(f"bestmove {move.uci()}")
            sys.stdout.flush()

        elif line == "quit":
            break


if __name__ == "__main__":
    main()
