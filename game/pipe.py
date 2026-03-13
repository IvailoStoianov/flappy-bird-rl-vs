import random
from typing import Tuple

import pygame


class Pipe:

    def __init__(self, x: float) -> None:
        self.x: float = x
        self.width: int = 80
        self.gap_size: int = 200
        self.gap_y: int = random.randint(150, 450)

        self.speed: float = 3.0
        self.color: Tuple[int, int, int] = (0, 255, 0)

        self.passed: bool = False
        self.sprite: pygame.Surface | None = None

    def update(self) -> None:
        self.x -= self.speed

    def set_sprite(self, sprite: pygame.Surface) -> None:
        """Attach a sprite and synchronize collision width."""
        self.sprite = sprite
        self.width = sprite.get_width()

    def draw(self, screen: pygame.Surface, screen_height: int) -> None:
        top_height = self.gap_y - self.gap_size // 2
        bottom_y = self.gap_y + self.gap_size // 2

        if self.sprite is None:
            pygame.draw.rect(screen, self.color, (int(self.x), 0, self.width, top_height))
            pygame.draw.rect(
                screen,
                self.color,
                (int(self.x), bottom_y, self.width, screen_height - bottom_y),
            )
            return

        # Draw bottom pipe using the sprite at the gap's bottom edge.
        bottom_rect = self.sprite.get_rect()
        bottom_rect.midtop = (int(self.x + self.width / 2), bottom_y)
        screen.blit(self.sprite, bottom_rect)

        # Draw top pipe as a vertically flipped sprite.
        flipped = pygame.transform.flip(self.sprite, False, True)
        top_rect = flipped.get_rect()
        top_rect.midbottom = (int(self.x + self.width / 2), top_height)
        screen.blit(flipped, top_rect)