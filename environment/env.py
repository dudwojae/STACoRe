# ==========================================================================
# This code is utilized from https://github.com/aravindsrinivas/curl_rainbow
# ==========================================================================
import cv2
import random
import atari_py

import torch

from collections import deque


class Atari_Env():
    def __init__(self, args):
        self.args = args
        self.atari = atari_py.ALEInterface()
        self.atari.setInt('random_seed', args.seed)
        self.atari.setInt('frame_skip', 0)
        self.atari.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.atari.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.atari.setBool('color_averaging', False)
        self.atari.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options

        actions = self.atari.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.atari.getScreenGrayscale(),
                           (self.args.resize, self.args.resize), interpolation=cv2.INTER_LINEAR)

        return torch.tensor(state, dtype=torch.float32, device=self.args.cuda).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(
                self.args.resize, self.args.resize, device=self.args.cuda))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.atari.act(0)  # Use a no-op after loss of life

        else:
            # Reset internals
            self._reset_buffer()
            self.atari.reset_game()

            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.atari.act(0)  # Assumes raw action 0 is always no-op

                if self.atari.game_over():
                    self.atari.reset_game()

        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.atari.lives()

        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, self.args.resize, self.args.resize, device=self.args.cuda)
        reward, done = 0, False

        for t in range(4):
            reward += self.atari.act(self.actions.get(action))

            if t == 2:
                frame_buffer[0] = self._get_state()

            elif t == 3:
                frame_buffer[1] = self._get_state()

            done = self.atari.game_over()

            if done:
                break

        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)

        # Detect loss if life as terminal in training mode
        if self.training:
            lives = self.atari.lives()

            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True

            self.lives = lives

        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):

        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.atari.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(20)

    def close(self):
        cv2.destroyAllWindows()
