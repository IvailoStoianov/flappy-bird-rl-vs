import pygame
from typing import Tuple, List


class Bird:

    def __init__(self, x: float = 250, y: float = 400) -> None:
        self.x: float = x
        self.y: float = y
        self.size: int = 30
        self.color: Tuple[int, int, int] = (255, 255, 0)  # yellow

        self.velocity: float = 0.0
        self.max_velocity: float = 10.0
        self.gravity: float = 0.5
        self.flap_strength: float = -10.0

        self.sprite_frames: List[pygame.Surface] | None = None
        self.animation_counter: int = 0

    def update(self) -> None:
        self.velocity += self.gravity
        self.y += self.velocity

        if self.sprite_frames:
            self.animation_counter = (self.animation_counter + 1) % (
                len(self.sprite_frames) * 5
            )

    def flap(self) -> None:
        self.velocity = self.flap_strength

    def set_sprites(self, frames: List[pygame.Surface]) -> None:
        """Attach sprite frames for rendering."""
        self.sprite_frames = frames
        self.animation_counter = 0

    def draw(self, screen: pygame.Surface) -> None:
        if self.sprite_frames:
            frame_index = self.animation_counter // 5
            frame = self.sprite_frames[frame_index]

            angle = max(-25, min(90, -self.velocity * 3))
            rotated = pygame.transform.rotate(frame, angle)
            rect = rotated.get_rect(center=(int(self.x), int(self.y)))
            screen.blit(rotated, rect)
        else:
            bird_pos = (int(self.x), int(self.y))
            pygame.draw.circle(screen, self.color, bird_pos, self.size)