'''

snake.py

Michael Grimsley
6/8/2025

Snake AI:
    Recreated the Snake game using the pygame library
    Developed a Deep Q-Learning agent to play Snake autonomously
    Implemented a neural network with Pytorch
    Tuned hyperparameter and included reward shaping to accelerate training
    
Wishlist:
    Fix bug where player can move in the opposite direction and lose by changing directions too fast

'''

import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time

# Initialize Pygame
pygame.init()

# Game Settings
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 30
DEFAULT_SPEED = 10
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

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
    reset():
        Reset the Snake's attributes for training
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
    render():
        Render the Snake on the screen
    '''
    def __init__(self):
        '''
        Constructor to initialize the Snake attributes
        
        The reset function is used for training purposes
        '''
        self.reset()
    
    def reset(self):
        self.body = [(4 - i, 4) for i in range(3)]
        self.direction = (1, 0)
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

    def render(self, surface):
        '''
        Render the Snake on the screen
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
    render():
        Render the Food on the screen
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

    def render(self, surface):
        '''
        Render the Food on the screen
        '''
        rect = pygame.Rect(self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(surface, RED, rect)

class Brain:
    '''
    Class to represent the Snake agent's brain
    
    Attributes
    ----------
    snake: Snake
        Snake object for the agent to control
    food: Food
        Food object for the Snake to find
    
    Methods
    -------
    reset():
        Reset the Brain's attributes for training
    get_state():
        Get the state of the environment for the neural network to analyze
    step():
        Step function for the agent to move and receive feedback
    '''
    def __init__(self, snake, food):
        self.snake = snake
        self.food = food
        self.score = 0
    
    def reset(self):
        self.score = 0
        self.snake.reset()
        self.food.position = self.food.random_position(self.snake)
    
    def get_state(self):
        '''
        Get the state of the environment for the neural network to analyze
        
        Current inputs (5):
            x- and y-coordinates of the Snake's head
            distance from head to food
            length of the snake
        
        Returns
        -------
        np.array:
            state vector
        '''
        head = self.snake.body[0]
        food_dx = self.food.position[0] - head[0]
        food_dy = self.food.position[1] - head[1]
        score = self.score
        
        return np.array([
            head[0], head[1], food_dx, food_dy, score
        ], dtype=np.float32)
        
    def step(self, action):
        '''
        Step function for the agent to move and receive feedback
        
        Implements reward shaping
        
        Parameters
        ----------
        action: tuple
            Tuple indicating the direction to move
            
        Returns
        -------
        np.array:
            Environment state for that step
        float:
            Reward value for that step
        boolean:
            Flag for checking when the training iteration is complete
        '''
        self.snake.change_direction(action)
        self.snake.move()
        
        reward = -0.01 # initiale penalty for each step taken
        done = self.snake.collides_with_self() or self.snake.collides_with_wall()
        
        if done:
            reward -= 10 # penalty for losing
        else:
            if self.snake.body[0] == self.food.position:
                self.food.position = self.food.random_position(self.snake)
                self.snake.grow()
                reward += 10 # reward for reaching the food
                self.score += 1
        
        return self.get_state(), reward, done
        
##############################
#        UI Functions        #
##############################

def render_text(text, font, color, surface, x, y):
    '''
    Render text on the screen centered at (x, y)

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

def menu_screen():
    '''
    Render the Menu screen
    
    Returns number indicating game mode
    - 0: Menu
    - 1: Play
    - 2: Train
    - 3: Watch
    
    Returns
    -------
    int:
        Mode for game state loop
    '''
    curr_speed = DEFAULT_SPEED
    while True:
        screen.fill(BLACK)
        render_text("Snake Game", font, GREEN, screen, WIDTH // 2, 150)
        render_text("Play", small_font, WHITE, screen, WIDTH // 4, 350)
        render_text("Train", small_font, WHITE, screen, WIDTH // 2, 350)
        render_text("Watch", small_font, WHITE, screen, WIDTH // 4 + WIDTH // 2, 350)

        mx, my = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()[0]

        if click and 100 < mx < 200 and 300 < my < 400: # Play button
            return 1
        elif click and 250 < mx < 350 and 300 < my < 400: # Train button
            return 2
        elif click and 400 < mx < 500 and 300 < my < 400: # Watch button
            return 3

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()
        clock.tick(curr_speed)

def game_over_screen(score):
    '''
    Render the Game Over screen

    Includes a Play Again button

    Parameters
    ----------
    score: int
        Final score of the game
    '''
    curr_speed = DEFAULT_SPEED
    while True:
        screen.fill(BLACK)
        render_text("Game Over", font, RED, screen, WIDTH // 2, 200)
        render_text(f"Score: {score}", small_font, WHITE, screen, WIDTH // 2, 270)
        render_text("Click to Play Again", small_font, WHITE, screen, WIDTH // 2, 320)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if pygame.mouse.get_pressed()[0]:
            return

        pygame.display.flip()
        clock.tick(curr_speed)

##############################
#             DQN            #
##############################

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

##############################
#          Game Loops        #
##############################

def player_loop():
    '''
    Official player game loop
    
    Press the arrow keys to move
    Multiple the movement speed by pressing numbers 1-0
    '''
    snake = Snake()
    food = Food(snake)
    score = 0
    curr_speed = DEFAULT_SPEED

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
            elif event.type == pygame.KEYDOWN and 48 < event.key < 58:
                curr_speed = (event.key - 48) * 10

        snake.move()

        if snake.body[0] == food.position:
            snake.grow()
            food = Food(snake)
            score += 1

        if snake.collides_with_self() or snake.collides_with_wall():
            return score

        render_text(f"Score: {score}", small_font, WHITE, screen, 60, 20)
        
        snake.render(screen)
        food.render(screen)
        pygame.display.flip()
        clock.tick(curr_speed)

def train_loop(file):
    snake = Snake()
    food = Food(snake)
    brain = Brain(snake, food)
    curr_speed = DEFAULT_SPEED
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(input_size=5, hidden_size=128, output_size=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()
    memory = deque(maxlen=5000)
    batch_size = 64
    gamma = 0.98
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    
    directions = (
        (0, -1),    # Up
        (0, 1),     # Down
        (-1, 0),    # Left
        (1, 0),     # Right
    )
    
    for iteration in range(1000):
        start_time = time.time()
        state = brain.get_state()
        brain.reset()
        total_reward = 0
        done = False
        t = 0
        while not done:
            screen.fill(BLACK)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and 48 < event.key < 58:
                    curr_speed = (event.key - 48) * 10
            
            render_text(f"Score: {brain.score}", small_font, WHITE, screen, 60, 20)
            
            snake.render(screen)
            food.render(screen)
            pygame.display.flip()
            clock.tick(curr_speed)
            
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32).to(device)
                    action = torch.argmax(model(s)).item()
                    
            next_state, reward, done = brain.step(directions[action])
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            if not done:
                if t % 5 == 0 and len(memory) >= batch_size:
                    batch = random.sample(memory, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                    actions = torch.tensor(actions).to(device)
                    rewards = torch.tensor(rewards).to(device)
                    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
                    dones = torch.tensor(dones, dtype=torch.bool).to(device)
                    
                    q_vals = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                    max_next_q_vals = model(next_states).max(1)[0]
                    target_q_vals = rewards + gamma * max_next_q_vals * (~dones)
                    
                    loss = criterion(q_vals, target_q_vals.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            t += 1
            
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        elapsed_time = time.time() - start_time
        print(f"Iteration: {iteration} - Score: {brain.score} - Epsilon: {epsilon:.2f} - Time: {elapsed_time:.2f}s")
    
    torch.save(model.state_dict(), file)
    print(f"Model saved as {file}")

def watch_loop(file):
    snake = Snake()
    food = Food(snake)
    brain = Brain(snake, food)
    curr_speed = DEFAULT_SPEED
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(input_size=5, hidden_size=128, output_size=4).to(device)
    model.load_state_dict(torch.load(file, map_location=device))
    model.eval()
    
    directions = (
        (0, -1),    # Up
        (0, 1),     # Down
        (-1, 0),    # Left
        (1, 0),     # Right
    )
    
    while True:
        state = brain.get_state()
        brain.reset()
        done = False
        while not done:
            screen.fill(BLACK)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and 48 < event.key < 58:
                    curr_speed = (event.key - 48) * 10
            
            render_text(f"Score: {brain.score}", small_font, WHITE, screen, 60, 20)
            
            snake.render(screen)
            food.render(screen)
            pygame.display.flip()
            clock.tick(curr_speed)
            
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).to(device)
                q_values = model(s).cpu().numpy()
                action = np.argmax(q_values)
                
            state, _, done = brain.step(directions[action])
        
def main():
    mode = 0
    file = "'snake.pth'"
    while True:
        if mode == 0:
            mode = menu_screen()
        elif mode == 1:
            score = player_loop()
            game_over_screen(score)
        elif mode == 2:
            train_loop(file)
        elif mode == 3:
            watch_loop(file)

if __name__ == "__main__":
    main()
