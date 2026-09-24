import datetime
import json
import pathlib

import imageio
import numpy as np


class Recorder:

  def __init__(
      self, env, directory, save_stats=True, save_video=True,
      save_episode=True, video_size=(512, 512), video_fps=None):
    if directory and save_stats:
      env = StatsRecorder(env, directory)
    if directory and save_video:
      env = VideoRecorder(env, directory, video_size, video_fps)
    if directory and save_episode:
      env = EpisodeRecorder(env, directory)
    self._env = env

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)


class StatsRecorder:

  def __init__(self, env, directory):
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self._file = (self._directory / 'stats.jsonl').open('a')
    self._length = None
    self._reward = None
    self._unlocked = None
    self._stats = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    obs = self._env.reset()
    self._length = 0
    self._reward = 0
    self._unlocked = None
    self._stats = None
    return obs

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    self._length += 1
    self._reward += info['reward']
    if done:
      self._stats = {'length': self._length, 'reward': round(self._reward, 1)}
      for key, value in info['achievements'].items():
        self._stats[f'achievement_{key}'] = value
      self._save()
    return obs, reward, done, info

  def _save(self):
    self._file.write(json.dumps(self._stats) + '\n')
    self._file.flush()


class VideoRecorder:

  def __init__(self, env, directory, size=(512, 512), fps=None):
    if not hasattr(env, 'episode_name'):
      env = EpisodeName(env)
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self._size = size
    self._fps = fps
    self._frames = None
    self._saved = False
    self._saved_path = None
    self._closed = False

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    obs = self._env.reset()
    self._frames = [self._env.render(self._size)]
    self._saved = False
    self._saved_path = None
    return obs

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    self._frames.append(self._env.render(self._size))
    if done:
      self._save()
    return obs, reward, done, info

  def _save(self):
    if self._saved:
      return self._saved_path
    if not self._frames:
      return None
    name = self._env.episode_name
    if name.startswith('None-'):
      name = f'partial-len{len(self._frames) - 1}'
    filename = self._directory / (name + '.mp4')
    kwargs = {}
    if self._fps is not None:
      kwargs['fps'] = self._fps
    imageio.mimsave(str(filename), self._frames, **kwargs)
    if not filename.is_file():
      raise RuntimeError(f'Video encoder did not create {filename}.')
    self._saved = True
    self._saved_path = filename.resolve()
    return self._saved_path

  def close(self):
    """Finalize a complete or partial episode exactly once and return its path."""
    path = self._save()
    if not self._closed:
      close = getattr(self._env, 'close', None)
      if close:
        close()
      self._closed = True
    return path


class EpisodeRecorder:

  def __init__(self, env, directory):
    if not hasattr(env, 'episode_name'):
      env = EpisodeName(env)
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self._episode = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    obs = self._env.reset()
    self._episode = [{'image': obs}]
    return obs

  def step(self, action):
    # Transitions are defined from the environment perspective, meaning that a
    # transition contains the action and the resulting reward and next
    # observation produced by the environment in response to said action.
    obs, reward, done, info = self._env.step(action)
    transition = {
        'action': action, 'image': obs, 'reward': reward, 'done': done,
    }
    for key, value in info.items():
      if key in ('inventory', 'achievements'):
        continue
      transition[key] = value
    for key, value in info['achievements'].items():
      transition[f'achievement_{key}'] = value
    for key, value in info['inventory'].items():
      transition[f'ainventory_{key}'] = value
    self._episode.append(transition)
    if done:
      self._save()
    return obs, reward, done, info

  def _save(self):
    filename = str(self._directory / (self._env.episode_name + '.npz'))
    # Fill in zeros for keys missing at the first time step.
    for key, value in self._episode[1].items():
      if key not in self._episode[0]:
        self._episode[0][key] = np.zeros_like(value)
    episode = {
        k: np.array([step[k] for step in self._episode])
        for k in self._episode[0]}
    np.savez_compressed(filename, **episode)


class EpisodeName:

  def __init__(self, env):
    self._env = env
    self._timestamp = None
    self._unlocked = None
    self._length = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    obs = self._env.reset()
    self._timestamp = None
    self._unlocked = None
    self._length = 0
    return obs

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    self._length += 1
    if done:
      self._timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
      self._unlocked = sum(int(v >= 1) for v in info['achievements'].values())
    return obs, reward, done, info

  @property
  def episode_name(self):
    return f'{self._timestamp}-ach{self._unlocked}-len{self._length}'
