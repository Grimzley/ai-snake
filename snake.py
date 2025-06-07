'''

snake.py

Michael Grimsley
6/7/2025

Snake AI:
    Recreated the Snake game using the pygame library

'''

import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Screen Settings
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 30
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (100, 100, 100)
BLUE = (0, 0, 255)

# Fonts
font_name = "Georgia"
font = pygame.font.SysFont(font_name, 30)
small_font = pygame.font.SysFont(font_name, 20)

# Clock
clock = pygame.time.Clock()

class Snake:
    '''
    Class to represent the Snake controlled by the player

    Attributes
    ----------
    body: list
        List of coordinates for the Snake's body
    direction: tuple
        Tuple indicating the Snake's direction of movement
    grow_pending: boolean
        Flag for checking if the Snake's body should grow when moving

    Methods
    -------
    move():
        Move the Snake in its current direction
    grow():
        Increase the Snake's body size
    change_direction():
        Change the direction the Snake is moving
    collides_with_self():
        Check if the Snake collides with itself
    collides_with_wall():
        Check if the Snake collides with a wall
    draw():
        Draw the Snake on the screen
    '''
    def __init__(self):
        '''
        Constructor to initialize the Snake attributes
        '''
        self.body = [(5, 5)]
        self.direction = (0, 0)
        self.grow_pending = False

    def move(self):
        '''
        Move the Snake in its current direction

        A new head is added to the beginning of the Snake's body based on its current direction
        The tail of the Snake's body is then removed unless the Snake is growing
        '''
        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        self.body = [new_head] + self.body

        if self.grow_pending:
            self.grow_pending = False
        else:
            self.body.pop()

    def grow(self):
        '''
        Increase the Snake's body size

        The grow_pending flag is set to True which allows the Snake's body to grow in the move function
        '''
        self.grow_pending = True

    def change_direction(self, new_dir):
        '''
        Change the direction the Snake is moving

        The Snake's direction will not change if it attempts to go in the opposite direction
        '''
        opposite = (-self.direction[0], -self.direction[1])
        if new_dir != opposite:
            self.direction = new_dir

    def collides_with_self(self):
        '''
        Check if the Snake collides with itself

        Returns
        -------
        boolean:
            True if the Snake's head is in its body
        '''
        return self.body[0] in self.body[1:]
    
    def collides_with_wall(self):
        '''
        Check if the Snake collides with a wall

        Returns
        -------
        boolean:
            True if the Snake goes out of bounds
        '''
        x, y = self.body[0]
        return x < 0 or x >= WIDTH // GRID_SIZE or y < 0 or y >= HEIGHT // GRID_SIZE

    def draw(self, surface):
        '''
        Draw the Snake on the screen
        '''
        for segment in self.body:
            rect = pygame.Rect(segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(surface, GREEN, rect)

class Food:
    '''
    Class to represent the Food eaten by the Snake

    Attributes
    ----------
    position: tuple
        Coordinates of the Food's location

    Methods
    -------
    random_position():
        Get a random position on the screen to place the Food
    draw():
        Draw the Food on the screen
    '''
    def __init__(self, snake):
        '''
        Constructor to initialize the Food attributes

        Parameters
        ----------
        snake: Snake
            Snake object to place the Food around
        '''
        self.position = self.random_position(snake)

    def random_position(self, snake):
        '''
        Get a random position on the screen to place the Food

        The Snake's body is accessed to prevent Food from being placed inside it

        Parameters
        ----------
        snake: Snake
            Snake object to place the Food around

        Returns
        -------
        tuple:
            Coordinates to place the Food
        '''
        positions = [(x, y) for x in range(WIDTH // GRID_SIZE) for y in range(HEIGHT // GRID_SIZE)
                     if (x, y) not in snake.body]
        return random.choice(positions)

    def draw(self, surface):
        '''
        Draw the Food on the screen
        '''
        rect = pygame.Rect(self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(surface, RED, rect)

##############################
#        UI Functions        #
##############################

def draw_text(text, font, color, surface, x, y):
    '''
    Draw text on the screen centered at (x, y)

    Parameters
    ----------
    text: String
        Text content to be written
    font: pygame font
        Font for the text to be written in
    color: tuple
        RGB values for color
    surface: pygame display
        Screen for the text to be displayed on
    x: int
        x-coordinate of the text
    y: int
        y-coordinate of the text
    '''
    textobj = font.render(text, True, color)
    rect = textobj.get_rect(center=(x, y))
    surface.blit(textobj, rect)

def draw_slider(surface, speed):
    '''
    Draw speed slider on the screen

    Parameters
    ----------
    surface: pygame display
        Screen for the slider to be displayed on
    speed: int
        Number indicating how fast the Snake will move
    '''
    pygame.draw.rect(surface, GRAY, (200, 450, 200, 10))
    handle_x = 200 + ((speed - 5) * 20)
    pygame.draw.rect(surface, BLUE, (handle_x - 10, 440, 20, 30))
    draw_text(f"Speed: {speed}", small_font, WHITE, surface, WIDTH // 2, 500)

def menu_screen():
    '''
    Draw the Menu screen

    Includes a Click to Play button and a speed slider
    '''
    speed = 10
    slider_rect = pygame.Rect(200, 440, 200, 30)
    dragging = False

    while True:
        screen.fill(BLACK)
        draw_text("Snake Game", font, GREEN, screen, WIDTH // 2, 150)
        draw_text("Click to Play", small_font, WHITE, screen, WIDTH // 2, 250)
        draw_slider(screen, speed)

        mx, my = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()[0]

        if click and 220 < mx < 380 and 240 < my < 270:
            return speed

        if click and slider_rect.collidepoint(mx, my):
            dragging = True
        elif not click:
            dragging = False

        if dragging:
            rel_x = max(0, min(mx - 200, 200))
            speed = int(5 + rel_x / 20)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()
        clock.tick(30)

def game_over_screen(score):
    '''
    Draw the Game Over screen

    Includes a Play Again button

    Parameters
    ----------
    score: int
        Final score of the game
    '''
    while True:
        screen.fill(BLACK)
        draw_text("Game Over", font, RED, screen, WIDTH // 2, 200)
        draw_text(f"Score: {score}", small_font, WHITE, screen, WIDTH // 2, 270)
        draw_text("Click to Play Again", small_font, WHITE, screen, WIDTH // 2, 320)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if pygame.mouse.get_pressed()[0]:
            return

        pygame.display.flip()
        clock.tick(30)

##############################
#          Game Loop         #
##############################

def game_loop(speed):
    snake = Snake()
    food = Food(snake)
    score = 0
    move_counter = 0

    directions = {
        pygame.K_UP: (0, -1),
        pygame.K_DOWN: (0, 1),
        pygame.K_LEFT: (-1, 0),
        pygame.K_RIGHT: (1, 0),
    }

    while True:
        screen.fill(BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key in directions:
                snake.change_direction(directions[event.key])

        move_counter += 1
        if move_counter >= (15 - speed):
            move_counter = 0
            snake.move()

            if snake.body[0] == food.position:
                snake.grow()
                food = Food(snake)
                score += 1

            if snake.collides_with_self() or snake.collides_with_wall():
                game_over_screen(score)
                return

        snake.draw(screen)
        food.draw(screen)
        draw_text(f"Score: {score}", small_font, WHITE, screen, 60, 20)

        pygame.display.flip()
        clock.tick(30)

def main():
    while True:
        speed = menu_screen()
        game_loop(speed)

if __name__ == "__main__":
    main()
