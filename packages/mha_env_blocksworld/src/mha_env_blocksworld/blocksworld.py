from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path
from typing import Any, cast, Literal
from enum import Enum
from dataclasses import dataclass
import warnings

import imageio
import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
    )
    import pygame
import gymnasium as gym

@dataclass(frozen=True)
class BWSnapshot:
    flat_state: np.ndarray
    arm_pos: int
    arm_holds: int
    last_reward: float
    total_reward: float
    action: int = -1
    legal: bool = True


@dataclass
class BWTrace:
    flat_state: np.ndarray
    arm_pos: np.ndarray
    arm_holds: np.ndarray
    last_reward: np.ndarray
    total_reward: np.ndarray
    action: np.ndarray
    legal: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "BWTrace":
        data = np.load(path)
        return cls(
            flat_state=data["flat_state"],
            arm_pos=data["arm_pos"],
            arm_holds=data["arm_holds"],
            last_reward=data["last_reward"],
            total_reward=data["total_reward"],
            action=data["action"],
            legal=data["legal"]
        )

    def snapshot_at(self, i: int) -> BWSnapshot:
        return BWSnapshot(
            flat_state=self.flat_state[i],
            arm_pos=int(self.arm_pos[i]),
            arm_holds=int(self.arm_holds[i]),
            last_reward=float(self.last_reward[i]),
            total_reward=float(self.total_reward[i]),
            action=int(self.action[i]),
            legal=bool(self.legal[i])
        )


class BlocksWorldEnv(gym.Env):
    """
    A simple blocks-world environment without intrinsic goals & corresponding rewards

    Actions:
        0: pick up
        1: put down
        2: move left
        3: move right

    Rewards:
        pick up: -0.05
        put down: -0.05
        move left: -0.01
        move right: -0.01
        illegal move: -0.5
    """
    class Actions(Enum):
        PICK_UP = 0
        PUT_DOWN = 1
        MOVE_LEFT = 2
        MOVE_RIGHT = 3

    ACTION_NAMES: dict[int, str] = {
        Actions.PICK_UP.value: 'Pick Up',
        Actions.PUT_DOWN.value: 'Put Down',
        Actions.MOVE_LEFT.value: 'Move Left',
        Actions.MOVE_RIGHT.value: 'Move Right',
        -1: '<RESET>'
    }

    R_PICK_UP = -0.05
    R_PUT_DOWN = -0.05
    R_MOVE_LEFT = -0.01
    R_MOVE_RIGHT = -0.01
    R_ILLEGAL = -0.5

    BLOCK_COLOR = (70, 70, 70)
    BLOCK_EDGE_COLOR = (0, 0, 0)
    FONT_COLOR = (240, 240, 245)
    BACKGROUND_COLOR = (240, 240, 245)
    ARM_COLOR = (20, 20, 80)
    INFO_COLOR = (20, 80, 20)
    INFO_ILLEGAL_COLOR = (80, 20, 20)

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    F_HAND_EMPTY = 'HandEmpty'
    F_HOLDING = 'Holding'
    F_ON = 'On'
    F_CLEAR = 'Clear'
    F_ABOVE = 'Above'
    # F_NOT_ABOVE = 'NotAbove'
    F_AT_LOC = 'AtLoc'
    F_LEFT_OF = 'LeftOf'
    # F_RIGHT_OF = 'RightOf'

    def __init__(
            self,
            table_len: int,
            num_blocks: int,
            render_mode=None,
            width=640,
            height=480,
            symbolic: bool = False
    ) -> None:
        assert render_mode in (None, *self.metadata["render_modes"])
        self.render_mode = render_mode
        self.width, self.height = width, height
        self.block_w = int(self.width / (1.2 * table_len))
        self.space_w = max(int(0.1 * self.block_w), 2)
        self.block_h = int((self.height - self.width / (6 * table_len)) / (num_blocks + 3))
        self.arm_thickness = int(0.2 * self.block_w)
        self.font = None

        self.observation_space = gym.spaces.MultiBinary((num_blocks + 2, table_len, num_blocks))
        self.action_space = gym.spaces.Discrete(4)
        self._num_blocks = num_blocks
        self._table_len = table_len
        self._state = np.zeros(self.observation_space.shape)
        self._arm_pos: int = -1
        self._arm_holds: int = -1
        self._flat_state = -np.ones((num_blocks, table_len), dtype=int)
        self._clear_pos = np.zeros((table_len,), dtype=int)

        self._symbolic = symbolic
        self._objects: dict[str, Any] = dict()
        self._fluents: dict[str, Any] = dict()
        self._sym_state: set[Any] = set()
        self._fnode_type: type[Any] | None = None
        self._objType = None
        self._blockType = None
        self._locType = None
        self._n_b_digits = len(str(self._num_blocks - 1))
        self._n_t_digits = len(str(self._table_len - 1))
        if self._symbolic:
            self._init_symbolic()

        self._last_reward = 0
        self._total_reward = 0

        self.expose_snapshot = False
        self._last_action: int = -1
        self._last_legal: bool = True

        self._rng = np.random.default_rng()

        self.window = None  # the display Surface for "human"
        self.canvas = None  # an offscreen Surface to draw on
        self.clock = None
        self.font = None

    def _init_symbolic(self) -> None:
        from unified_planning.model import FNode, Fluent, Object
        from unified_planning.shortcuts import BoolType, UserType

        self._fnode_type = FNode
        self._objType = UserType('obj')
        self._blockType = UserType('block', self._objType)
        self._locType = UserType('loc', self._objType)
        for i in range(self._num_blocks):
            nam = f'B{i:0{self._n_b_digits}d}'
            self._objects[nam] = Object(nam, self._blockType)
        for i in range(self._table_len):
            nam = f't{i:0{self._n_t_digits}d}'
            self._objects[nam] = Object(nam, self._locType)
        self._fluents[self.F_HAND_EMPTY] = Fluent(self.F_HAND_EMPTY, BoolType())
        self._fluents[self.F_HOLDING] = Fluent(self.F_HOLDING, BoolType(), x=self._blockType)
        self._fluents[self.F_ON] = Fluent(self.F_ON, BoolType(), x=self._blockType, y=self._objType)
        self._fluents[self.F_CLEAR] = Fluent(self.F_CLEAR, BoolType(), x=self._objType)
        self._fluents[self.F_ABOVE] = Fluent(self.F_ABOVE, BoolType(), x=self._locType)
        self._fluents[self.F_AT_LOC] = Fluent(self.F_AT_LOC, BoolType(), x=self._objType, y=self._locType)
        self._fluents[self.F_LEFT_OF] = Fluent(self.F_LEFT_OF, BoolType(), x=self._locType, y=self._locType)

    def _resolve_sym_args(self, args: Iterable[str | FNode | Object]) -> list[Object]:
        resolved_args: list[Object] = list()
        for arg in args:
            if isinstance(arg, str):
                resolved_args.append(self._objects[arg])
            elif self._fnode_type is not None and isinstance(arg, self._fnode_type):
                resolved_args.append(arg.object())
            else:
                resolved_args.append(arg)
        return resolved_args

    def _sym_add(self, fluent: str, *args, state: set[FNode] | None = None) -> None:
        if state is None:
            state = self._sym_state
        state.add(self._fluents[fluent](*self._resolve_sym_args(args)))

    def _sym_rm(self, fluent: str, *args, state: set[FNode] | None = None) -> None:
        if state is None:
            state = self._sym_state
        state.remove(self._fluents[fluent](*self._resolve_sym_args(args)))

    def _random_state(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, set[FNode]]:
        state = np.zeros_like(self._state)
        flat_state = -np.ones_like(self._flat_state)
        clear_pos = np.ones((self._table_len,), dtype=int) * self._num_blocks - 1
        sym_state: set[FNode] = set()
        col_tops: list[Object] = list()
        if self._symbolic:
            col_tops = [self._objects[f't{i:0{self._n_t_digits}d}'] for i in range(self._table_len)]
            for i in range(self._table_len - 1):
                t = self._objects[f't{i:0{self._n_t_digits}d}']
                tn = self._objects[f't{(i + 1):0{self._n_t_digits}d}']
                self._sym_add(self.F_LEFT_OF, t, tn, state=sym_state)
                # sym_state.add(self._fluents[self.F_LEFT_OF](t, tn))
                self._sym_add(self.F_AT_LOC, t, t, state=sym_state)
                # sym_state.add(self._fluents[self.F_AT_LOC](t, t))
            t = self._objects[f't{(self._table_len - 1):0{self._n_t_digits}d}']
            self._sym_add(self.F_AT_LOC, t, t, state=sym_state)
            # sym_state.add(self._fluents[self.F_AT_LOC](t, t))
            self._sym_add(self.F_HAND_EMPTY, state=sym_state)
            # sym_state.add(self._fluents[self.F_HAND_EMPTY]())
        for b in range(self._num_blocks):
            col = rng.integers(0, self._table_len)
            state[clear_pos[col] + 2, col, b] = 1
            flat_state[clear_pos[col], col] = b
            clear_pos[col] -= 1
            if self._symbolic:
                block = self._objects[f'B{b:0{self._n_b_digits}d}']
                loc = self._objects[f't{col:0{self._n_t_digits}d}']
                self._sym_add(self.F_ON, block, col_tops[col], state=sym_state)
                col_tops[col] = block
                self._sym_add(self.F_AT_LOC, block, loc, state=sym_state)
        if self._symbolic:
            for i in range(self._table_len):
                self._sym_add(self.F_CLEAR, col_tops[i], state=sym_state)

        arm_pos = rng.integers(0, self._table_len).item()
        if self._symbolic:
            t = self._objects[f't{arm_pos:0{self._n_t_digits}d}']
            self._sym_add(self.F_ABOVE, t, state=sym_state)
        for i in range(self._num_blocks):
            state[0, arm_pos, i] = 1
        return state, flat_state, clear_pos, arm_pos, sym_state

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ) -> tuple[gym.core.ObsType, dict[str, Any]]:
        super().reset()
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state, self._flat_state, self._clear_pos, self._arm_pos, self._sym_state = self._random_state(self._rng)
        self._arm_holds = -1

        self._last_reward = 0
        self._total_reward = 0
        self._last_action = -1
        self._last_legal = True

        if self.render_mode == "human":
            self.render()

        info = {'snapshot': self._snapshot} if self.expose_snapshot else dict()

        if self._symbolic:
            state = [str(node) for node in self._sym_state]
        else:
            state = self._state.copy()

        return state, info

    def step(
            self,
            action: gym.core.ActType
    ) -> tuple[gym.core.ObsType | list[str], gym.core.SupportsFloat, bool, bool, dict[str, Any]]:
        if self._arm_pos < 0:
            raise RuntimeError("BlocksWorldEnv.step() called before reset()")
        reward = self.R_ILLEGAL
        self._last_action = action
        match action:
            case 0:  # pick up
                if self._arm_holds != -1 or self._clear_pos[self._arm_pos] == self._num_blocks - 1:  # arm is holding something or its column is empty
                    self._last_legal = False
                else:
                    self._last_legal = True
                    self._clear_pos[self._arm_pos] += 1
                    self._arm_holds = self._flat_state[self._clear_pos[self._arm_pos], self._arm_pos]
                    self._state[1, self._arm_pos, self._arm_holds] = 1

                    self._state[self._clear_pos[self._arm_pos] + 2, self._arm_pos, self._arm_holds] = 0
                    self._flat_state[self._clear_pos[self._arm_pos], self._arm_pos] = -1
                    reward = self.R_PICK_UP
                    if self._symbolic:
                        x = self._objects[f'B{self._arm_holds:0{self._n_b_digits}d}']
                        y: Object | None = None
                        z = self._objects[f't{self._arm_pos:0{self._n_t_digits}d}']
                        for obj in self._objects.values():
                            if self._fluents[self.F_ON](x, obj) in self._sym_state:
                                y = obj
                                break
                        assert y is not None
                        self._sym_rm(self.F_ON, x, y)
                        self._sym_rm(self.F_CLEAR, x)
                        self._sym_rm(self.F_HAND_EMPTY)
                        self._sym_rm(self.F_AT_LOC, x, z)
                        self._sym_add(self.F_HOLDING, x)
                        self._sym_add(self.F_CLEAR, y)
            case 1:  # put down
                if self._arm_holds == -1:
                    self._last_legal = False
                else:
                    self._last_legal = True
                    idx_holding = self._arm_holds
                    self._flat_state[self._clear_pos[self._arm_pos], self._arm_pos] = self._arm_holds
                    self._state[self._clear_pos[self._arm_pos] + 2, self._arm_pos, self._arm_holds] = 1
                    self._state[1, self._arm_pos, self._arm_holds] = 0
                    self._clear_pos[self._arm_pos] -= 1
                    self._arm_holds = -1
                    reward = self.R_PUT_DOWN
                    if self._symbolic:
                        x = self._objects[f'B{idx_holding:0{self._n_b_digits}d}']
                        y: Object | None = None
                        z = self._objects[f't{self._arm_pos:0{self._n_t_digits}d}']
                        for obj in self._objects.values():
                            if self._fluents[self.F_AT_LOC](obj, z) in self._sym_state and self._fluents[self.F_CLEAR](obj) in self._sym_state:
                                y = obj
                                break
                        assert y is not None
                        self._sym_rm(self.F_HOLDING, x)
                        self._sym_rm(self.F_CLEAR, y)
                        self._sym_add(self.F_ON, x, y)
                        self._sym_add(self.F_CLEAR, x)
                        self._sym_add(self.F_HAND_EMPTY)
                        self._sym_add(self.F_AT_LOC, x, z)
            case 2:  # move left
                if self._arm_pos == 0:  # cannot move left anymore
                    self._last_legal = False
                else:
                    self._last_legal = True
                    self._state[0, [self._arm_pos - 1, self._arm_pos], :] = self._state[
                        0, [self._arm_pos, self._arm_pos - 1], :]
                    self._state[1, [self._arm_pos - 1, self._arm_pos], :] = self._state[
                        1, [self._arm_pos, self._arm_pos - 1], :]
                    self._arm_pos -= 1
                    reward = self.R_MOVE_LEFT
                    if self._symbolic:
                        x = self._objects[f't{(self._arm_pos + 1):0{self._n_t_digits}d}']
                        y = self._objects[f't{self._arm_pos:0{self._n_t_digits}d}']
                        self._sym_rm(self.F_ABOVE, x)
                        self._sym_add(self.F_ABOVE, y)
            case 3:  # move right
                if self._arm_pos == self._table_len - 1:  # cannot move right anymore
                    self._last_legal = False
                else:
                    self._last_legal = True
                    self._state[0, [self._arm_pos, self._arm_pos + 1], :] = self._state[
                        0, [self._arm_pos + 1, self._arm_pos], :]
                    self._state[1, [self._arm_pos, self._arm_pos + 1], :] = self._state[
                        1, [self._arm_pos + 1, self._arm_pos], :]
                    self._arm_pos += 1
                    reward = self.R_MOVE_RIGHT
                    if self._symbolic:
                        x = self._objects[f't{(self._arm_pos - 1):0{self._n_t_digits}d}']
                        y = self._objects[f't{self._arm_pos:0{self._n_t_digits}d}']
                        self._sym_rm(self.F_ABOVE, x)
                        self._sym_add(self.F_ABOVE, y)
            case _:
                self._last_legal = False

        self._last_reward = reward
        self._total_reward += reward

        if self.render_mode == "human":
            self.render()

        info = {'snapshot': self._snapshot} if self.expose_snapshot else dict()

        if self._symbolic:
            state = [str(node) for node in self._sym_state]
            for i in range(len(state)):
                if '(' not in state[i]:
                    state[i] = f'{state[i]}()'
        else:
            state = self._state.copy()

        return state, reward, False, False, info

    def render(self) -> np.ndarray | None:
        if self.canvas is None:
            # Create a canvas to draw each frame (works for both human & rgb_array)
            pygame.display.init()  # safe even if already inited
            self.canvas = pygame.Surface((self.width, self.height))

        # Clear background
        self.canvas.fill(self.BACKGROUND_COLOR)

        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 24)  # default font

        # Draw blocks
        x = self.space_w
        highest = (self._num_blocks + 2) * self.block_h
        for col in range(self._table_len):
            y = self.height - self.block_h
            for row in range(self._num_blocks - 1, -1, -1):
                if self._flat_state[row, col] == -1:
                    continue
                highest = min(highest, y)
                pygame.draw.rect(self.canvas, self.BLOCK_EDGE_COLOR, pygame.Rect(x, y, self.block_w, self.block_h))
                block_rect = pygame.Rect(x + 4, y + 4, self.block_w - 8, self.block_h - 8)
                pygame.draw.rect(self.canvas, self.BLOCK_COLOR, block_rect)
                label = self.font.render(str(self._flat_state[row, col]), True, self.FONT_COLOR)
                label_rect = label.get_rect(center=block_rect.center)
                self.canvas.blit(label, label_rect)

                y -= self.block_h
            x += self.block_w + 2 * self.space_w

        # Draw arm
        x = self._arm_pos * (self.block_w + 2 * self.space_w) + self.space_w
        y = highest - 2 * self.block_h
        pygame.draw.rect(self.canvas, self.ARM_COLOR,
                         pygame.Rect(x - self.arm_thickness, y, self.arm_thickness, self.block_h))
        pygame.draw.rect(self.canvas, self.ARM_COLOR, pygame.Rect(x - self.arm_thickness, y - self.arm_thickness,
                                                                  self.block_w + 2 * self.arm_thickness,
                                                                  self.arm_thickness))
        pygame.draw.rect(self.canvas, self.ARM_COLOR,
                         pygame.Rect(x + self.block_w, y, self.arm_thickness, self.block_h))

        # Draw held block
        if self._arm_holds != -1:
            pygame.draw.rect(self.canvas, self.BLOCK_EDGE_COLOR, pygame.Rect(x, y, self.block_w, self.block_h))
            block_rect = pygame.Rect(x + 4, y + 4, self.block_w - 8, self.block_h - 8)
            pygame.draw.rect(self.canvas, self.BLOCK_COLOR, block_rect)
            label = self.font.render(str(self._arm_holds), True, self.FONT_COLOR)
            label_rect = label.get_rect(center=block_rect.center)
            self.canvas.blit(label, label_rect)

        # Draw reward info
        reward_info_rect = pygame.Rect(0, 0, self.width, self.block_h)
        pygame.draw.rect(self.canvas, self.INFO_COLOR if self._last_legal else self.INFO_ILLEGAL_COLOR, reward_info_rect)
        label = self.font.render(f'Action taken: {self.ACTION_NAMES[self._last_action]}{'' if self._last_legal else ' (ILLEGAL)'}, '
                                 f'reward: {self._last_reward}, total reward: {self._total_reward:.3f}', True,
                                 self.FONT_COLOR)
        label_rect = label.get_rect(center=reward_info_rect.center)
        self.canvas.blit(label, label_rect)

        if self.render_mode == "human":
            self._ensure_pygame()

            # Handle OS events so the window stays responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

            # Blit canvas to the display window and show it
            self.window.blit(self.canvas, (0, 0))
            pygame.display.flip()

            pygame.event.pump()

            self.clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            # Convert the offscreen surface to an RGB numpy array
            return self._surface_to_array(self.canvas)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        if self.canvas is not None:
            self.canvas = None
        if self.clock is not None:
            self.clock = None
        if self.font is not None:
            self.font = None

    def _ensure_pygame(self) -> None:
        if self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Blocks World")
        if self.clock is None:
            self.clock = pygame.time.Clock()

    @staticmethod
    def _surface_to_array(surface: pygame.Surface):
        # returns (H, W, 3) uint8 in RGB order
        array = pygame.surfarray.array3d(surface)  # (W, H, 3)
        return np.transpose(array, (1, 0, 2))

    @property
    def _snapshot(self) -> BWSnapshot:
        return BWSnapshot(
            flat_state=self._flat_state.copy(),
            arm_pos=int(self._arm_pos),
            arm_holds=int(self._arm_holds),
            last_reward=float(self._last_reward),
            total_reward=float(self._total_reward),
            action=int(self._last_action),
            legal=bool(self._last_legal)
        )

    @property
    def table_len(self) -> int:
        return self._table_len

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def _action_name(self, action: int) -> str:
        if action in self.ACTION_NAMES:
            return self.ACTION_NAMES[action]
        else:
            return f'UNKNOWN ({action})'

    def restore_snapshot(self, snapshot: BWSnapshot) -> None:
        self._flat_state = snapshot.flat_state.copy()
        self._arm_pos = snapshot.arm_pos
        self._arm_holds = snapshot.arm_holds
        self._last_reward = snapshot.last_reward
        self._total_reward = snapshot.total_reward
        self._last_action = snapshot.action
        self._last_legal = snapshot.legal
        self._rebuild_observation_from_flat_state()

    def _rebuild_observation_from_flat_state(self) -> None:
        self._state = np.zeros_like(self._state)
        self._clear_pos = np.full(
            (self._table_len,),
            self._num_blocks - 1,
            dtype=int,
        )

        if self._symbolic:
            self._sym_state = set()
            col_tops: list[Object] = [
                self._objects[f"t{i:0{self._n_t_digits}d}"]
                for i in range(self._table_len)
            ]

            for i in range(self._table_len - 1):
                t = self._objects[f"t{i:0{self._n_t_digits}d}"]
                tn = self._objects[f"t{i + 1:0{self._n_t_digits}d}"]
                self._sym_add(self.F_LEFT_OF, t, tn)
                self._sym_add(self.F_AT_LOC, t, t)

            last_t = self._objects[f"t{self._table_len - 1:0{self._n_t_digits}d}"]
            self._sym_add(self.F_AT_LOC, last_t, last_t)
        else:
            col_tops = []

        seen_blocks: set[int] = set()

        for col in range(self._table_len):
            clear_pos = self._num_blocks - 1
            seen_empty_above_stack = False

            # Bottom to top.
            for row in range(self._num_blocks - 1, -1, -1):
                block_idx = int(self._flat_state[row, col])

                if block_idx == -1:
                    seen_empty_above_stack = True
                    continue

                if seen_empty_above_stack:
                    raise ValueError(
                        f"Invalid floating block {block_idx} at row={row}, col={col}"
                    )

                if block_idx < 0 or block_idx >= self._num_blocks:
                    raise ValueError(
                        f"Invalid block id {block_idx} at row={row}, col={col}"
                    )

                if block_idx in seen_blocks:
                    raise ValueError(f"Block {block_idx} appears more than once")

                seen_blocks.add(block_idx)
                self._state[row + 2, col, block_idx] = 1
                clear_pos = row - 1

                if self._symbolic:
                    block = self._objects[f"B{block_idx:0{self._n_b_digits}d}"]
                    loc = self._objects[f"t{col:0{self._n_t_digits}d}"]

                    self._sym_add(self.F_ON, block, col_tops[col])
                    self._sym_add(self.F_AT_LOC, block, loc)
                    col_tops[col] = block

            self._clear_pos[col] = clear_pos

        if self._arm_pos < 0 or self._arm_pos >= self._table_len:
            raise ValueError(f"Invalid arm position: {self._arm_pos}")

        self._state[0, self._arm_pos, :] = 1

        if self._symbolic:
            for top in col_tops:
                self._sym_add(self.F_CLEAR, top)

            above_loc = self._objects[f"t{self._arm_pos:0{self._n_t_digits}d}"]
            self._sym_add(self.F_ABOVE, above_loc)

        if self._arm_holds == -1:
            if self._symbolic:
                self._sym_add(self.F_HAND_EMPTY)
        else:
            if self._arm_holds < 0 or self._arm_holds >= self._num_blocks:
                raise ValueError(f"Invalid held block id: {self._arm_holds}")

            if self._arm_holds in seen_blocks:
                raise ValueError(
                    f"Held block {self._arm_holds} also appears in _flat_state"
                )

            self._state[1, self._arm_pos, self._arm_holds] = 1
            seen_blocks.add(self._arm_holds)

            if self._symbolic:
                held = self._objects[f"B{self._arm_holds:0{self._n_b_digits}d}"]
                self._sym_add(self.F_HOLDING, held)

        if len(seen_blocks) != self._num_blocks:
            missing = set(range(self._num_blocks)) - seen_blocks
            raise ValueError(f"Missing blocks from reconstructed state: {missing}")


def generate_trace_gif(
        path: Path | str,
        env: BlocksWorldEnv,
        fps: float = 2.,
        save_format: Literal['mp4', 'gif', 'both'] = 'mp4'
) -> None:
    path = Path(path).resolve()
    trace = BWTrace.load(path)
    frames: list[np.ndarray] = []

    for i in range(len(trace.arm_pos)):
        env.restore_snapshot(trace.snapshot_at(i))
        frames.append(cast(np.ndarray, env.render()))

    if save_format == 'mp4' or save_format == 'both':
        imageio.mimsave(path.with_suffix('.mp4'), frames, fps=fps)
    if save_format == 'gif' or save_format == 'both':
        imageio.mimsave(path.with_suffix('.gif'), frames, fps=fps)
