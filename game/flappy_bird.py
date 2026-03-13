from __future__ import annotations

from typing import List, Tuple, Dict

import pygame
import numpy as np
from pathlib import Path

from .bird import Bird
from .pipe import Pipe


# Window and game configuration
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800
FPS = 60

GROUND_HEIGHT = 50
GROUND_Y = WINDOW_HEIGHT - GROUND_HEIGHT

PIPE_SPAWN_DELAY_FRAMES = 90

BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 255)  # blue
GROUND_COLOR: Tuple[int, int, int] = (139, 69, 19) # brown
TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255) # white
GAME_OVER_COLOR: Tuple[int, int, int] = (255, 0, 0) # red


class FlappyBird:

    def __init__(self) -> None:
        # Pygame rendering objects (only created when needed)
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.title_font: pygame.font.Font | None = None
        self.text_font: pygame.font.Font | None = None

        # Sprite and sound assets (loaded lazily on first render)
        self.background_img: pygame.Surface | None = None
        self.base_img: pygame.Surface | None = None
        self.pipe_img: pygame.Surface | None = None
        self.bird_frames: List[pygame.Surface] | None = None
        self.message_img: pygame.Surface | None = None
        self.gameover_img: pygame.Surface | None = None
        self.digit_images: Dict[str, pygame.Surface] = {}

        self.sound_wing: pygame.mixer.Sound | None = None
        self.sound_hit: pygame.mixer.Sound | None = None
        self.sound_die: pygame.mixer.Sound | None = None
        self.sound_point: pygame.mixer.Sound | None = None
        self.sound_swoosh: pygame.mixer.Sound | None = None

        # Ground and base scrolling parameters
        self.ground_height: int = GROUND_HEIGHT
        self.ground_y: int = GROUND_Y
        self.base_x: float = 0.0
        self.base_scroll_speed: float = 3.0

        self.reset()

    def reset(self) -> None:
        self.bird = Bird(x=WINDOW_WIDTH / 2, y=WINDOW_HEIGHT / 2)
        # Start with a single pipe ahead of the bird so observations are always valid
        self.pipes: List[Pipe] = [Pipe(WINDOW_WIDTH)]
        self.spawn_timer = 0
        self.score = 0
        self.game_over = False

        # If assets are already loaded, immediately re-attach sprites so we
        # do not fall back to primitive shapes after a reset.
        if self.bird_frames:
            self.bird.set_sprites(self.bird_frames)
        if self.pipe_img and self.pipes:
            for pipe in self.pipes:
                pipe.set_sprite(self.pipe_img)

        return self._get_observation()
    
    def take_action(self, action=None):
        """
        Step the Flappy Bird game once for RL.

        action:
          - 0: do nothing
          - 1: flap
        """
        # If the episode is already over, just return final observation
        if self.game_over:
            return self._get_observation(), 0.0, True, False, {"score": self.score}

        # Interpret action (supports scalar or numpy array from SB3)
        if action is not None:
            act = int(np.asarray(action).item())
            if act == 1:
                self.bird.flap()

        # Run one game update step
        prev_score = self.score
        self.update()

        terminated = self.game_over

        # Reward shaping:
        # staying alive +0.1
        # passing a pipe +10
        # death -100
        reward = 0.1
        if self.score > prev_score:
            reward = 10.0
        if terminated:
            reward = -100.0

        observation = self._get_observation()
        info = {"score": self.score}

        # No time-limit truncation logic yet, so truncated=False
        return observation, reward, terminated, False, info
    
    def _get_observation(self):
        # Find the next pipe that the bird has not passed yet
        next_pipe = None
        for pipe in self.pipes:
            if not pipe.passed:
                next_pipe = pipe
                break

        # If there are no pipes yet, create a "virtual" one straight ahead
        if next_pipe is None:
            pipe_x = WINDOW_WIDTH
            gap_center_y = WINDOW_HEIGHT / 2
            gap_size = 200
        else:
            pipe_x = next_pipe.x
            gap_center_y = next_pipe.gap_y
            gap_size = next_pipe.gap_size

        # Bird state
        bird_x = self.bird.x
        bird_y = self.bird.y
        bird_velocity = self.bird.velocity

        # Pipe gap geometry derived from center + gap size
        top_pipe_bottom = gap_center_y - gap_size / 2
        bottom_pipe_top = gap_center_y + gap_size / 2

        # --- Observations (normalized) ---

        # Bird vertical position relative to screen height
        bird_height_norm = bird_y / WINDOW_HEIGHT

        # Bird vertical velocity normalized by max velocity
        bird_velocity_norm = bird_velocity / self.bird.max_velocity

        # Horizontal distance from bird to next pipe
        # Positive = pipe is ahead, negative = bird passed it
        horizontal_distance_to_pipe = (pipe_x - bird_x) / WINDOW_WIDTH

        # Vertical distance between bird and center of pipe gap
        # Helps the agent align with the gap
        vertical_distance_to_gap = (gap_center_y - bird_y) / WINDOW_HEIGHT

        # Distance between bird and bottom of the top pipe
        # Helps avoid hitting the top pipe
        distance_to_top_pipe = (top_pipe_bottom - bird_y) / WINDOW_HEIGHT

        # Distance between bird and top of the bottom pipe
        # Helps avoid hitting the bottom pipe
        distance_to_bottom_pipe = (bird_y - bottom_pipe_top) / WINDOW_HEIGHT

        # Clip values to keep them in stable range for RL algorithms
        horizontal_distance_to_pipe = np.clip(horizontal_distance_to_pipe, -1, 1)
        vertical_distance_to_gap = np.clip(vertical_distance_to_gap, -1, 1)
        bird_velocity_norm = np.clip(bird_velocity_norm, -1, 1)

        # Return observation vector
        return np.array([
            bird_height_norm,
            bird_velocity_norm,
            horizontal_distance_to_pipe,
            vertical_distance_to_gap,
            distance_to_top_pipe,
            distance_to_bottom_pipe
        ], dtype=np.float32)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not self.game_over:
                self.bird.flap()
                try:
                    self.sound_wing.play()
                except Exception:
                    pass
            elif event.key == pygame.K_RETURN and self.game_over:
                try:
                    self.sound_swoosh.play()
                except Exception:
                    pass
                self.reset()

    def update(self) -> None:
        if self.game_over:
            return

        self.bird.update()

        # collision check for ground
        if self.bird.y >= self.ground_y:
            self.bird.y = self.ground_y
            self.bird.velocity = 0
            self.game_over = True
            try:
                self.sound_die.play()
            except Exception:
                pass

        # generate pipes
        self.spawn_timer += 1
        if self.spawn_timer > PIPE_SPAWN_DELAY_FRAMES:
            new_pipe = Pipe(WINDOW_WIDTH)
            if self.pipe_img is not None:
                new_pipe.set_sprite(self.pipe_img)
            self.pipes.append(new_pipe)
            self.spawn_timer = 0

        # update pipes, handle collisions and scoring
        for pipe in self.pipes:
            pipe.update()

                # collision check for pipes
            if self.bird.x + self.bird.size > pipe.x and self.bird.x < pipe.x + pipe.width:
                top = pipe.gap_y - pipe.gap_size // 2
                bottom = pipe.gap_y + pipe.gap_size // 2
                if self.bird.y < top or self.bird.y > bottom:
                    self.game_over = True
                    try:
                        self.sound_hit.play()
                    except Exception:
                        pass

            # increase score if pipe is passed
            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                self.score += 1
                pipe.passed = True
                try:
                    self.sound_point.play()
                except Exception:
                    pass

        # remove pipes that have moved off-screen
        self.pipes = [pipe for pipe in self.pipes if pipe.x + pipe.width > 0]

        # Scroll the ground/base if we have a sprite
        if not self.game_over and self.base_img is not None:
            self.base_x -= self.base_scroll_speed
            base_width = self.base_img.get_width()
            if self.base_x <= -base_width:
                self.base_x += base_width

    def _load_assets(self) -> None:
        """Load images and sounds from the asset pack once."""
        if self.background_img is not None:
            return

        try:
            base_path = Path(__file__).resolve().parent
            assets_root = base_path / "assets"
            game_objects = assets_root / "Game Objects"
            ui_dir = assets_root / "UI"
            numbers_dir = ui_dir / "Numbers"
            sounds_dir = assets_root / "Sound Efects"

            # Images
            self.background_img = pygame.image.load(
                str(game_objects / "background-day.png")
            ).convert()
            self.base_img = pygame.image.load(
                str(game_objects / "base.png")
            ).convert_alpha()
            self.pipe_img = pygame.image.load(
                str(game_objects / "pipe-green.png")
            ).convert_alpha()

            self.bird_frames = [
                pygame.image.load(str(game_objects / "yellowbird-downflap.png")).convert_alpha(),
                pygame.image.load(str(game_objects / "yellowbird-midflap.png")).convert_alpha(),
                pygame.image.load(str(game_objects / "yellowbird-upflap.png")).convert_alpha(),
            ]

            self.message_img = pygame.image.load(
                str(ui_dir / "message.png")
            ).convert_alpha()
            self.gameover_img = pygame.image.load(
                str(ui_dir / "gameover.png")
            ).convert_alpha()

            # Score digits 0–9
            self.digit_images = {}
            for d in range(10):
                img = pygame.image.load(str(numbers_dir / f"{d}.png")).convert_alpha()
                self.digit_images[str(d)] = img

            # Sounds
            self.sound_wing = pygame.mixer.Sound(str(sounds_dir / "wing.ogg"))
            self.sound_hit = pygame.mixer.Sound(str(sounds_dir / "hit.ogg"))
            self.sound_die = pygame.mixer.Sound(str(sounds_dir / "die.ogg"))
            self.sound_point = pygame.mixer.Sound(str(sounds_dir / "point.ogg"))
            self.sound_swoosh = pygame.mixer.Sound(str(sounds_dir / "swoosh.ogg"))

            # Scale sprites based on the background height to match the pack's proportions.
            bg_h = self.background_img.get_height()
            scale = WINDOW_HEIGHT / bg_h if bg_h > 0 else 1.0
            if scale != 1.0:
                # Use nearest-neighbour scaling to keep pixel art sharp.
                self.base_img = pygame.transform.scale(
                    self.base_img,
                    (
                        int(self.base_img.get_width() * scale),
                        int(self.base_img.get_height() * scale),
                    ),
                )
                self.pipe_img = pygame.transform.scale(
                    self.pipe_img,
                    (
                        int(self.pipe_img.get_width() * scale),
                        int(self.pipe_img.get_height() * scale),
                    ),
                )
                bird_extra_scale = 1.15
                scaled_frames: List[pygame.Surface] = []
                for frame in self.bird_frames:
                    scaled_frames.append(
                        pygame.transform.scale(
                            frame,
                            (
                                int(frame.get_width() * scale * bird_extra_scale),
                                int(frame.get_height() * scale * bird_extra_scale),
                            ),
                        )
                    )
                self.bird_frames = scaled_frames

            # Ground collision follows the base sprite.
            self.ground_height = self.base_img.get_height()
            self.ground_y = WINDOW_HEIGHT - self.ground_height

            # Attach sprites to existing game objects.
            self.bird.set_sprites(self.bird_frames)
            for pipe in self.pipes:
                pipe.set_sprite(self.pipe_img)

        except Exception as exc:
            # Fall back to simple shapes if loading fails.
            print(f"Asset loading failed: {exc}")
            self.background_img = None
            self.base_img = None
            self.pipe_img = None
            self.bird_frames = None
            self.message_img = None
            self.gameover_img = None
            self.digit_images = {}
            self.sound_wing = None
            self.sound_hit = None
            self.sound_die = None
            self.sound_point = None
            self.sound_swoosh = None

    def _draw_background(self) -> None:
        if self.background_img is not None:
            # Tile or stretch the background to window size
            bg = pygame.transform.scale(self.background_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
            self.screen.blit(bg, (0, 0))
        else:
            self.screen.fill(BACKGROUND_COLOR)

    def _draw_base(self) -> None:
        if self.base_img is None:
            pygame.draw.rect(
                self.screen,
                GROUND_COLOR,
                (0, self.ground_y, WINDOW_WIDTH, self.ground_height),
            )
            return

        base_width = self.base_img.get_width()
        x = self.base_x
        while x < WINDOW_WIDTH:
            self.screen.blit(self.base_img, (int(x), WINDOW_HEIGHT - self.base_img.get_height()))
            x += base_width

    def _draw_score(self) -> None:
        if not self.digit_images:
            # Fallback to font-based score
            if self.text_font is None:
                return
            score_text = self.text_font.render(str(self.score), True, TEXT_COLOR)
            text_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 100))
            self.screen.blit(score_text, text_rect)
            return

        score_str = str(self.score)
        digit_surfaces = [self.digit_images[d] for d in score_str]
        total_width = sum(surf.get_width() for surf in digit_surfaces)
        x = (WINDOW_WIDTH - total_width) // 2
        y = 100
        for surf in digit_surfaces:
            self.screen.blit(surf, (x, y))
            x += surf.get_width()

    def draw(self) -> None:
        if self.screen is None or self.text_font is None or self.title_font is None:
            return

        # draw background
        self._draw_background()

        # draw pipes first so the base/ground can appear in front of the
        # bottom pipe segments, matching the original game's look.
        for pipe in self.pipes:
            pipe.draw(self.screen, WINDOW_HEIGHT)

        # draw ground/base in front of the bottom pipes
        self._draw_base()

        # draw bird last so it appears above both pipes and ground
        self.bird.draw(self.screen)

        if not self.game_over:
            # display score using sprite digits or font
            self._draw_score()
        else:
            if self.gameover_img is not None:
                rect = self.gameover_img.get_rect(center=(WINDOW_WIDTH // 2, 300))
                self.screen.blit(self.gameover_img, rect)

            # Final score below game over banner
            self._draw_score()

            # Simple restart hint using font
            restart_text = self.text_font.render(
                "Press ENTER to restart", True, TEXT_COLOR
            )
            self.screen.blit(
                restart_text,
                restart_text.get_rect(center=(WINDOW_WIDTH // 2, 450)),
            )

    def render(self, mode: str = "human") -> None:
        """Render the current game frame for Gymnasium."""
        if mode == "human":
            # Lazily create the window and rendering resources
            if self.screen is None:
                self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
                pygame.display.set_caption("Flappy Bird")
                self.clock = pygame.time.Clock()
                self.title_font = pygame.font.SysFont("Arial", 60)
                self.text_font = pygame.font.SysFont("Arial", 40)

                # Load all images/sounds now that a display exists
                self._load_assets()

            self.draw()
            pygame.display.flip()

            if self.clock is not None:
                self.clock.tick(FPS)

    def close(self) -> None:
        """Close the game and pygame."""
        pygame.quit()


def main() -> None:
    pygame.init()

    game = FlappyBird()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                game.handle_event(event)

        game.update()
        game.render("human")

    pygame.quit()


if __name__ == "__main__":
    main()
