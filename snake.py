'''

snake.py

Michael Grimsley
6/11/2025

Snake AI:
    Recreated the Snake game using the Pygame library
    Developed a Deep Q-Learning agent to play Snake autonomously
    Implemented a neural network with Pytorch
    Tuned hyperparameters and included reward shaping to accelerate training
    Saves the model for watching and further training
    
Wishlist:
    Fix bug where player can move in the opposite direction and lose by changing directions too fast
    Add input error handling
    Improve agent's spatial awareness

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
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# AI Settings
ITERATIONS = 1000

DIRECTIONS = (
    (0, -1),    # Up
    (0, 1),     # Down
    (-1, 0),    # Left
    (1, 0),     # Right
)

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
        Reset the Snake's attributes
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
        '''
        Reset the Snake's attributes
        '''
        self.body = [(5 - i, 5) for i in range(3)]
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
        
        Parameters
        ----------
        new_dir: tuple
            Tuple indicating the direction to change to if possible
            
        Returns
        -------
        boolean:
            True if the direction was successfully changed
        '''
        opposite = (-self.direction[0], -self.direction[1])
        success = new_dir != opposite
        if success:
            self.direction = new_dir
        return success

    def collides_with_self(self, point):
        '''
        Check if the point collides with itself
        
        Parameters
        ----------
        point: Tuple
            Point to analyze
        
        Returns
        -------
        boolean:
            True if the Snake's head is in its body
        '''
        return point in self.body[1:]
    
    def collides_with_wall(self, point):
        '''
        Check if the point collides with a wall
        
        Parameters
        ----------
        point: Tuple
            Point to analyze
        
        Returns
        -------
        boolean:
            True if the Snake goes out of bounds
        '''
        x, y = point
        return not (0 <= x < WIDTH // GRID_SIZE and 0 <= y < HEIGHT // GRID_SIZE)

    def render(self, surface):
        '''
        Render the Snake on the screen
        
        Parameters
        ----------
        surface: pygame display
            Screen for the Snake to be rendered on
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
        
        Parameters
        ----------
        surface: pygame display
            Screen for the Food to be rendered on
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
    score: int
        Counter for game score
    steps_since_last_food:
        Counter for steps taken since last Food was reached
    last_positions: deque
        Queue containing the most recent head positions
    
    Methods
    -------
    reset():
        Reset the Brain's attributes for training
    count_safe_cells():
        Count the number of safe cells around a given point
    count_spaces_until_collision():
        Count the number of safe spaces in a given directions
    get_state():
        Get the state of the environment for the neural network to analyze
    step():
        Step function for the agent to move and receive feedback
    '''
    def __init__(self, snake, food):
        '''
        Constructor to initialize the Brain attributes
        
        The reset function is used for training purposes
        
        Parameters
        ----------
        snake: Snake
            Snake object to control
        food: Food
            Food object to reach
        '''
        self.snake = snake
        self.food = food
        self.reset()
    
    def reset(self):
        '''
        Reset the Brain's attributes
        '''
        self.score = 0
        self.snake.reset()
        self.food.position = self.food.random_position(self.snake)
        
        self.steps_since_last_food = 0
        self.last_positions = deque(maxlen=50)
        
    def count_safe_cells(self, point):
        '''
        Count the number of safe cells around a given point
        
        Breadth-first search traversal 
        
        Parameters
        ----------
        point: Tuple
            The point to analyze
            
        Returns
        -------
        int:
            The number of safe cells around the point
        '''
        visited = set()
        queue = [point]
        count = 0
        for _ in range(30):
            if not queue:
                break
            x, y = queue.pop()
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx = x + dx
                ny = y + dy
                np = (nx, ny)
                if 0 <= nx < WIDTH // GRID_SIZE and 0 <= ny < HEIGHT // GRID_SIZE:
                    if np not in self.snake.body and np not in visited:
                        visited.add(np)
                        queue.append(np)
                        count += 1
        return count
    
    def count_spaces_until_collision(self, point, direction):
        '''
        Count the number of safe spaces in a given directions
        
        Parameters
        ----------
        point: Tuple
            Position to analyze
        direction: Tuple
            Check to check
            
        Returns
        -------
        int:
            Number of safe spaces from point in direction
        '''
        count = 0
        
        p = np.add(point, direction)
        x, y = p
 
        while 0 <= x < WIDTH // GRID_SIZE and 0 <= y < HEIGHT // GRID_SIZE:
            if (x, y) not in self.snake.body:
                count += 1
                p = np.add(p, direction)
                x, y = p
            else:
                break
        
        return count
    
    def get_state(self):
        '''
        Get the state of the environment for the neural network to analyze
        
        Current inputs:
            Danger in each direction (4)
            Current move directions (4)
            Valid directions (4)
            Distance from Food (2)
            Food directions (4)
            Distance from tail (2)
            Tail directions (4)
            Number of safe cells in each direction (4)
            Number of safe spaces in each direction (4)
            Length of the snake (score) (1)
        
        Returns
        -------
        np.array:
            state vector for the neural network
        '''
        head = self.snake.body[0]
        tail = self.snake.body[-1]
        
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        
        # Danger directions
        danger_up = int(self.snake.collides_with_self(point_u) or self.snake.collides_with_wall(point_u))
        danger_down = int(self.snake.collides_with_self(point_d) or self.snake.collides_with_wall(point_d))
        danger_left = int(self.snake.collides_with_self(point_l) or self.snake.collides_with_wall(point_l))
        danger_right = int(self.snake.collides_with_self(point_r) or self.snake.collides_with_wall(point_r))
        
        # One-hot move directions
        dir_up = int(self.snake.direction == (0, -1))
        dir_down = int(self.snake.direction == (0, 1))
        dir_left = int(self.snake.direction == (-1, 0))
        dir_right = int(self.snake.direction == (1, 0))
        
        # Valid directions
        valid_up = int(not dir_up)
        valid_down = int(not dir_down)
        valid_left = int(not dir_left)
        valid_right = int(not dir_right)
        
        # Distance from Food
        food_dx = self.food.position[0] - head[0]
        food_dy = self.food.position[1] - head[1]
        
        # One-hot Food directions
        food_up = int(food_dy < 0)
        food_down = int(food_dy > 0)
        food_left = int(food_dx < 0)
        food_right = int(food_dx > 0)
        
        # Distance from Tail
        tail_dx = tail[0] - head[0]
        tail_dy = tail[1] - head[1]
        
        # One-hot Tail directions
        tail_up = int(tail_dy < 0)
        tail_down = int(tail_dy > 0)
        tail_left = int(tail_dx < 0)
        tail_right = int(tail_dx > 0)
        
        # Number of safe cells ahead
        open_space_up = self.count_safe_cells(np.add(head, (0, -1)))
        open_space_down = self.count_safe_cells(np.add(head, (0, 1)))
        open_space_left = self.count_safe_cells(np.add(head, (-1, 0)))
        open_space_right = self.count_safe_cells(np.add(head, (1, 0)))
        
        # Number of safe spaces in each direction
        safe_up = self.count_spaces_until_collision(head, (0, -1))
        safe_down = self.count_spaces_until_collision(head, (0, 1))
        safe_left = self.count_spaces_until_collision(head, (-1, 0))
        safe_right = self.count_spaces_until_collision(head, (1, 0))
        
        # Score
        score = self.score
        
        return np.array([
            danger_up, danger_down, danger_left, danger_right,
            dir_up, dir_down, dir_left, dir_right,
            #valid_up, valid_down, valid_left, valid_right,
            #food_dx, food_dy,
            food_up, food_down, food_left, food_right,
            #tail_dx, tail_dy,
            #tail_up, tail_down, tail_left, tail_right,
            #open_space_up, open_space_down, open_space_left, open_space_right,
            #safe_up, safe_down, safe_left, safe_right,
            score
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
        curr_head = self.snake.body[0]
        success = self.snake.change_direction(action)
        self.snake.move()
        new_head = self.snake.body[0]
        food = self.food.position
        
        # Reward Shaping
        reward = -0.01 # penalty for each step taken
        
        if not success: # penalty for attempting to move in the oppopsite direction
            reward -= 1
        
        # Manhattan distance
        #old_dist = abs(food[0] - curr_head[0]) + abs(food[1] - curr_head[1])
        #new_dist = abs(food[0] - new_head[0]) + abs(food[1] - new_head[1])
        # Incentive to move closer to the Food
        #if new_dist < old_dist:
        #    reward += 0.02
        #elif new_dist > old_dist:
        #    reward -= 0.01
        
        done = self.snake.collides_with_self(new_head) or self.snake.collides_with_wall(new_head)
        if done:
            reward -= 10 # penalty for losing
        else:
            if new_head == food:
                self.food.position = self.food.random_position(self.snake)
                self.snake.grow()
                reward += 10 # reward for reaching the Food
                self.score += 1
                self.steps_since_last_food = 0
            else:
                self.steps_since_last_food += 1
        
        self.last_positions.append(new_head)
        if len(set(self.last_positions)) < len(self.snake.body) * 1.5: # penalty for repeating moves
            reward -= 0.5
        
        # Incentive to leave open space to prevent trapping
        #open_space = self.count_safe_cells(np.add(new_head, self.snake.direction))
        #reward += min(open_space * 0.002, 0.2)
        
        if self.steps_since_last_food > 300: # penalty for taking too long to reach Food
            reward -= 1
            done = True # repetition fail safe
        
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
        SCREEN.fill(BLACK)
        render_text("Snake Game", font, GREEN, SCREEN, WIDTH // 2, 150)
        render_text("Play", small_font, WHITE, SCREEN, WIDTH // 4, 350)
        render_text("Train", small_font, WHITE, SCREEN, WIDTH // 2, 350)
        render_text("Watch", small_font, WHITE, SCREEN, WIDTH // 4 + WIDTH // 2, 350)

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
        SCREEN.fill(BLACK)
        render_text("Game Over", font, RED, SCREEN, WIDTH // 2, 200)
        render_text(f"Score: {score}", small_font, WHITE, SCREEN, WIDTH // 2, 270)
        render_text("Click to Play Again", small_font, WHITE, SCREEN, WIDTH // 2, 320)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if pygame.mouse.get_pressed()[0]:
            return

        pygame.display.flip()
        clock.tick(curr_speed)

def render_env(snake, food, score, speed):
    '''
    Render the game environment
    
    Parameters
    ----------
    snake: Snake
        Snake object to be rendered
    food: Food
        Food object to be rendered
    score: int
        Current game score
    speed: int
        Snake movement speed
    '''
    SCREEN.fill(BLACK)
    
    render_text(f"Score: {score}", small_font, WHITE, SCREEN, 60, 20)
        
    snake.render(SCREEN)
    food.render(SCREEN)
    pygame.display.flip()
    clock.tick(speed)

##############################
#             DQN            #
##############################

class DQN(nn.Module):
    '''
    DQN class to build the model
    '''
    def __init__(self, input_size, hidden_size=128, output_size=4):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

def save_checkpoint(model, optimizer, epsilon, filename="checkpoint.pth"):
    '''
    Save the model for evaluation and future retraining
    '''
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epsilon': epsilon
    }
    torch.save(checkpoint, f"{filename}")
    print(f"Saved checkpoint to {filename}")

##############################
#          Game Loops        #
##############################

def player_loop():
    '''
    Official player game loop
    
    Press the arrow keys to move
    Multiple the movement speed by pressing numbers 1-0
    '''
    print("Playing Snake ...")
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
        render_env(snake, food, score, curr_speed)
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

        if snake.collides_with_self(snake.body[0]) or snake.collides_with_wall(snake.body[0]):
            return score

def train_loop():
    print("Enter model pth file to train (Leave blank to train new model): ")
    file = input()
    print("Enter model input size: ")
    nn_size = int(input())
    
    snake = Snake()
    food = Food(snake)
    brain = Brain(snake, food)
    curr_speed = DEFAULT_SPEED
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(input_size=nn_size).to(device)
    # Learning rate controls how much the model should change with each training step
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    # Probability the agent will take a random action instead of the best-known action
    epsilon = 1.0
    if not file:
        print("Generating new model ...")
    else:
        print("Getting previous model ...")
        checkpoint = torch.load(file)
        
        pretrained_dict = checkpoint['model_state']
        model_dict = model.state_dict()
        matched_state = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(matched_state)
        model.load_state_dict(model_dict)
        print("Successfully loaded weights")

        # Comment out if changing NN inputs or hyperparameters
        #optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        epsilon = checkpoint['epsilon']
        
        model.train()
    
    criterion = nn.MSELoss()
    memory = deque(maxlen=5000)
    batch_size = 64 # How much experience is used per training update
    gamma = 0.98 # Determines how much the agent values future vs immediate rewards
    epsilon_min = 0.05
    epsilon_decay = 0.995
    
    for iteration in range(ITERATIONS):
        start_time = time.time()
        state = brain.get_state()
        brain.reset()
        total_reward = 0
        done = False
        t = 0
        while not done:
            render_env(snake, food, brain.score, curr_speed)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and 48 < event.key < 58:
                    curr_speed = (event.key - 48) * 10
            
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32).to(device)
                    action = torch.argmax(model(s)).item()
                    
            next_state, reward, done = brain.step(DIRECTIONS[action])
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
        print(f"Iteration: {iteration} - Score/Reward: {brain.score}/{total_reward:.2f} - Epsilon: {epsilon:.2f} - Time: {elapsed_time:.2f}s")
    
    save_checkpoint(model, optimizer, epsilon)  

def watch_loop():
    print("Enter model pth file to watch: ")
    file = input()
    print("Enter model input size: ")
    nn_size = int(input())
    
    snake = Snake()
    food = Food(snake)
    brain = Brain(snake, food)
    curr_speed = DEFAULT_SPEED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(file)
    model = DQN(input_size=nn_size)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    while True:
        state = brain.get_state()
        brain.reset()
        done = False
        while not done:
            render_env(snake, food, brain.score, curr_speed)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and 48 < event.key < 58:
                    curr_speed = (event.key - 48) * 10
            
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).to(device)
                q_values = model(s).cpu().numpy()
                action = np.argmax(q_values)
                
            state, _, done = brain.step(DIRECTIONS[action])
        
def main():
    mode = 0
    while True:
        if mode == 0:
            mode = menu_screen()
        elif mode == 1:
            score = player_loop()
            game_over_screen(score)
        elif mode == 2:
            train_loop()
        elif mode == 3:
            watch_loop()

if __name__ == "__main__":
    main()
