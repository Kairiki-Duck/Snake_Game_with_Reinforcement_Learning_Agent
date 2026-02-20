import pygame
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle

class FancySnakeGame:
    def __init__(self, width=600, height=400, block_size=20, render=True):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.render_enable = render
        self.episode = 0

        pygame.init()

        if self.render_enable:
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Eating Agent - Enhanced")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.font_bold = pygame.font.SysFont(None, 48, bold=True)

        self.reset()
    
    def reset(self):
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.dx = self.block_size
        self.dy = 0
        self.score = 0
        
        # 游戏机制变量
        self.food_count = 0
        self.food = self.generate_food()
        self.super_food = None
        self.shrink_food = None
        
        # 状态帧数计数器 (用于替代毫秒计时器，保证无头训练时逻辑一致)
        self.frenzy_mode = False
        self.frenzy_duration_frames = 24
        self.frenzy_frames_left = 0
        self.super_food_frames_left = 0
        self.current_super_duration_frames = 160 # 约 8 秒
        self.effect_frames_left = 0
        
        self.combo_count = 0
        self.combo_frames_left = 0
        
        # 视觉特效容器
        self.particles = []
        self.floating_texts = []
        self.ghost_segments = []

        self.done = False
        self.episode += 1
        return self.get_state()
    
    def generate_food(self):
        all_positions = [(x, y) for x in range(0, self.width, self.block_size) for y in range(0, self.height, self.block_size)]
        available_positions = [pos for pos in all_positions if pos not in self.snake]
        if not available_positions:
            return (0, 0) # 兜底
        return random.choice(available_positions)
    
    def create_particles(self, x, y, color):
        for _ in range(10):
            self.particles.append([x+10, y+10, random.uniform(-2,2), random.uniform(-2,2), 255, color])

    def update_fx(self):
        for p in self.particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 15
            if p[4] <= 0:
                self.particles.remove(p)
        for ft in self.floating_texts[:]:
            ft[1] -= 1
            ft[3] -= 10
            if ft[3] <= 0:
                self.floating_texts.remove(ft)

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True
        
        # action: 0-up, 1-down, 2-left, 3-right
        if action == 0 and self.dy == 0:
            self.dx, self.dy = 0, -self.block_size
        elif action == 1 and self.dy == 0:
            self.dx, self.dy = 0, self.block_size
        elif action == 2 and self.dx == 0:
            self.dx, self.dy = -self.block_size, 0
        elif action == 3 and self.dx == 0:
            self.dx, self.dy = self.block_size, 0
        
        if self.render_enable:
            self.ghost_segments.append([self.snake[0][0], self.snake[0][1], 150])

        head_x, head_y = self.snake[0]
        new_head = (head_x + self.dx, head_y + self.dy)
        new_head = (new_head[0] % self.width, new_head[1] % self.height)

        reward = -0.05

        # 狂暴模式下碰到自己不会死！
        if new_head in self.snake and not self.frenzy_mode:
            self.done = True
            reward = -100
            return self.get_state(), reward, True
        
        self.snake.insert(0, new_head)

        collision_pos = None
        eat_type = None

        if self.food and new_head == self.food:
            collision_pos, eat_type = self.food, "normal"
        elif self.super_food and new_head == self.super_food:
            collision_pos, eat_type = self.super_food, "super"
        elif self.shrink_food and new_head == self.shrink_food:
            collision_pos, eat_type = self.shrink_food, "shrink"

        # 处理连击倒计时
        if self.combo_frames_left <= 0:
            self.combo_count = 0
        else:
            self.combo_frames_left -= 1

        if collision_pos:
            self.combo_count += 1
            self.combo_frames_left = 30 # 约 2 秒帧数
            base_gain = 1
            p_color = (255, 0, 0)

            if eat_type == "normal":
                if self.frenzy_mode:
                    base_gain = 10
                    self.frenzy_frames_left = 24
                    if self.render_enable:
                        self.floating_texts.append([collision_pos[0], collision_pos[1]-20, "TIME EXTENDED", 255])
                    p_color = (0, 255, 255)
                self.food_count += 1
                self.food = None
                
                if self.food_count % 20 == 0:
                    self.shrink_food = self.generate_food()
                    self.super_food_frames_left = 160
                    self.current_super_duration_frames = 160
                elif self.food_count % 5 == 0:
                    self.super_food = self.generate_food()
                    # 速度越快，超级食物存在帧数越短，保持现实时间感觉一致
                    speed = min(10 + self.food_count // 24, 15)
                    speed_multiplier = speed / 10.0
                    self.super_food_frames_left = int(160 / speed_multiplier)
                    self.current_super_duration_frames = self.super_food_frames_left
                else:
                    self.food = self.generate_food()
            
            elif eat_type == "super":
                base_gain = 20
                self.frenzy_mode = True
                self.frenzy_frames_left = 24
                p_color = (255, 215, 0)
                self.super_food = None
                self.super_food_frames_left = 0
                self.food = self.generate_food()
                self.effect_frames_left = 24

            elif eat_type == "shrink":
                base_gain = 20
                self.frenzy_mode = True
                self.frenzy_frames_left = 24
                shrink_amount = 20
                if len(self.snake) > shrink_amount + 2:
                    self.snake = self.snake[:-shrink_amount]
                else:
                    self.snake = self.snake[:3]
                p_color = (255, 105, 180)
                self.shrink_food = None
                self.super_food_frames_left = 0
                self.food = self.generate_food()
                self.effect_frames_left = 24

            earned_score = base_gain * self.combo_count
            self.score += earned_score
            reward = earned_score # 用实际得分作为奖励，鼓励打出连击和吃特殊食物

            if self.render_enable:
                self.create_particles(collision_pos[0], collision_pos[1], p_color)
                self.floating_texts.append([collision_pos[0], collision_pos[1], f"+{earned_score}", 255])
        else:
            self.snake.pop()

        # 处理各种状态的帧数倒计时
        if self.super_food_frames_left > 0:
            self.super_food_frames_left -= 1
            if self.super_food_frames_left <= 0:
                self.super_food = None
                self.shrink_food = None
                self.food = self.generate_food()

        if self.frenzy_mode:
            if self.frenzy_frames_left > 0:
                self.frenzy_frames_left -= 1
            else:
                self.frenzy_mode = False

        if self.effect_frames_left > 0:
            self.effect_frames_left -= 1

        return self.get_state(), reward, False
    
    def get_state(self):
        head_x, head_y = self.snake[0]

        point_l = (head_x - self.block_size, head_y)
        point_r = (head_x + self.block_size, head_y)
        point_u = (head_x, head_y - self.block_size)
        point_d = (head_x, head_y + self.block_size)

        dir_l = self.dx < 0
        dir_r = self.dx > 0
        dir_u = self.dy < 0
        dir_d = self.dy > 0

        def danger(point):
            if self.frenzy_mode:
                return False # 狂暴模式下没有任何危险
            # 考虑穿墙逻辑，使AI视野更精准
            wrapped_point = (point[0] % self.width, point[1] % self.height)
            return wrapped_point in self.snake
        
        # 动态锁定当前激活的食物（只取其中一个），保证AI输入向量始终为11维
        target_food = self.food or self.super_food or self.shrink_food
        
        state = [
            (dir_r and danger(point_r)) or 
            (dir_l and danger(point_l)) or 
            (dir_u and danger(point_u)) or 
            (dir_d and danger(point_d)),

            (dir_u and danger(point_r)) or
            (dir_d and danger(point_l)) or
            (dir_l and danger(point_u)) or
            (dir_r and danger(point_d)),

            (dir_d and danger(point_r)) or
            (dir_u and danger(point_l)) or
            (dir_r and danger(point_u)) or
            (dir_l and danger(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            target_food[0] < head_x,
            target_food[0] > head_x,
            target_food[1] < head_y,
            target_food[1] > head_y
        ]
        return list(map(int, state))
    
    def render(self):
        if not self.render_enable:
            return
        
        game_surface = pygame.Surface((self.width, self.height))
        bg_color = (10, 30, 40) if self.frenzy_mode else (0, 0, 0)
        
        # 屏幕闪烁特效
        if self.effect_frames_left > 0:
            if self.effect_frames_left > 16 or (8 < self.effect_frames_left < 12):
                game_surface.fill((255, 215, 0))
            else:
                game_surface.fill(bg_color)
        else:
            game_surface.fill(bg_color)

        # 网格
        for x in range(0, self.width, self.block_size):
            pygame.draw.line(game_surface, (20, 20, 20), (x, 0), (x, self.height))
        for y in range(0, self.height, self.block_size):
            pygame.draw.line(game_surface, (20, 20, 20), (0, y), (self.width, y))

        # 残影
        for g in self.ghost_segments[:]:
            s = pygame.Surface((self.block_size, self.block_size))
            s.set_alpha(g[2])
            s.fill((0, 255, 255) if self.frenzy_mode else (0, 150, 255))
            game_surface.blit(s, (g[0], g[1]))
            g[2] -= 20
            if g[2] <= 0:
                self.ghost_segments.remove(g)

        pulse = (math.sin(pygame.time.get_ticks() / 150) + 1) / 2
        
        # 渲染食物
        if self.food:
            glow_size = int(pulse * 6)
            pygame.draw.rect(game_surface, (100, 0, 0), (self.food[0]-glow_size//2, self.food[1]-glow_size//2, self.block_size+glow_size, self.block_size+glow_size))
            pygame.draw.rect(game_surface, (255, 0, 0), (self.food[0], self.food[1], self.block_size, self.block_size))
        
        if self.super_food:
            if pygame.time.get_ticks() // 400 % 2 == 0:
                pygame.draw.rect(game_surface, (255, 215, 0), (self.super_food[0], self.super_food[1], self.block_size, self.block_size))
                pygame.draw.rect(game_surface, (255, 255, 255), (self.super_food[0], self.super_food[1], self.block_size, self.block_size), 2)
        
        if self.shrink_food:
            size_offset = int(pulse * 4)
            pygame.draw.rect(game_surface, (255, 105, 180), (self.shrink_food[0]-size_offset//2, self.shrink_food[1]-size_offset//2, self.block_size+size_offset, self.block_size+size_offset))
            pygame.draw.rect(game_surface, (255, 255, 255), (self.shrink_food[0], self.shrink_food[1], self.block_size, self.block_size), 1)

        # 渲染蛇
        for i, segment in enumerate(self.snake):
            if self.frenzy_mode:
                color = (0, 200 + 55 * math.sin(pygame.time.get_ticks() / 50), 255)
            else:
                color_val = max(100, 255 - i * 12)
                color = (0, color_val, 0) if i > 0 else (50, 255, 150)
            
            pygame.draw.rect(game_surface, color, (segment[0], segment[1], self.block_size, self.block_size))
            
            if i == 0:
                eye_color = (255, 255, 255) if not self.frenzy_mode else (255, 0, 0)
                pygame.draw.circle(game_surface, eye_color, (segment[0] + 14, segment[1] + 6), 3)
            
            pygame.draw.rect(game_surface, (0, 50, 0), (segment[0], segment[1], self.block_size, self.block_size), 1)

        # UI 文本
        text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        game_surface.blit(text, (10, 10))

        text_ep = self.font.render(f"Episode: {self.episode-1}", True, (255, 255, 255))
        game_surface.blit(text_ep, (10, 40))

        if self.combo_count > 1:
            c_text = self.font.render(f"COMBO X{self.combo_count}", True, (255, 255, 0))
            game_surface.blit(c_text, (self.width - 150, 10))

        # 狂暴模式 UI
        if self.frenzy_mode:
            frenzy_remaining = max(0, self.frenzy_frames_left / 30)
            alpha = 150 + 105 * math.sin(pygame.time.get_ticks() / 100)
            frenzy_text = self.font_bold.render("FRENZY MODE", True, (0, 255, 255))
            frenzy_text.set_alpha(int(alpha))

            text_rect = frenzy_text.get_rect(center=(self.width // 2, 45))
            game_surface.blit(frenzy_text, text_rect)

            f_bar_width = 200
            f_bar_height = 8
            f_bar_x = self.width // 2 - f_bar_width // 2
            f_bar_y = 65

            pygame.draw.rect(game_surface, (0, 50, 50), (f_bar_x, f_bar_y, f_bar_width, f_bar_height))

            current_f_width = int(f_bar_width * frenzy_remaining)
            if current_f_width > 0:
                pygame.draw.rect(game_surface, (0, 255, 255), (f_bar_x, f_bar_y, current_f_width, f_bar_height))
                pygame.draw.line(game_surface, (200, 255, 255), (f_bar_x, f_bar_y), (f_bar_x + current_f_width, f_bar_y), 1)
            pygame.draw.rect(game_surface, (255, 255, 255), (f_bar_x, f_bar_y, f_bar_width, f_bar_height), 1)

        # 粒子和悬浮字
        self.update_fx()
        for p in self.particles:
            s = pygame.Surface((4, 4))
            s.set_alpha(p[4])
            s.fill(p[5])
            game_surface.blit(s, (int(p[0]), int(p[1])))
        for ft in self.floating_texts:
            t = self.font.render(ft[2], True, (255, 255, 255))
            t.set_alpha(ft[3])
            game_surface.blit(t, (int(ft[0]), int(ft[1])))

        shake_offset = (0, 0)

        # 超级食物进度条
        if (self.super_food or self.shrink_food) and self.super_food_frames_left > 0:
            remaining_ratio = max(0, self.super_food_frames_left / self.current_super_duration_frames)
            
            progress_bar_width = 220
            progress_bar_height = 14
            progress_bar_pos = (self.width // 2 - progress_bar_width // 2, 10)

            progress_length = int(progress_bar_width * remaining_ratio)
            
            r = int(255 * (1 - remaining_ratio))
            g = int(255 * remaining_ratio)
            b = 0

            bar_pulse = abs(pygame.time.get_ticks() % 1000 - 500) / 500
            brightness = 0.7 + 0.3 * bar_pulse

            bar_color = (int(r * brightness), int(g * brightness), int(b * brightness))

            pygame.draw.rect(game_surface, (60, 60, 60), (*progress_bar_pos, progress_bar_width, progress_bar_height))

            glow_surface = pygame.Surface((progress_length, progress_bar_height), pygame.SRCALPHA)
            glow_surface.fill((*bar_color, 120))
            game_surface.blit(glow_surface, progress_bar_pos)

            pygame.draw.rect(game_surface, bar_color, (*progress_bar_pos, progress_length, progress_bar_height))
            pygame.draw.rect(game_surface, (255, 255, 255), (*progress_bar_pos, progress_bar_width, progress_bar_height), 2)
            
            seconds_left = int(self.super_food_frames_left / 20) + 1 # 假设基准为 20fps
            timer_text = self.font.render(f"{seconds_left}", True, (255, 255, 255))
            game_surface.blit(timer_text, (progress_bar_pos[0] + progress_bar_width + 10, progress_bar_pos[1] - 6))
            
            if remaining_ratio < 0.25:
                shake_offset = (random.randint(-3, 3), random.randint(-3, 3))

        self.screen.fill((0, 0, 0))
        self.screen.blit(game_surface, shake_offset)
        pygame.display.flip()

        base_speed = min(30 + self.food_count // 8, 45)
        speed = base_speed * 1.5 if self.frenzy_mode else base_speed
        self.clock.tick(speed)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.fc(x)
    
class Agent:
    def __init__(self):
        self.model = Net()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=50000)

        self.gamma = 0.9

        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.998

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        state = torch.tensor(state, dtype=torch.float32)
        q = self.model(state)

        return torch.argmax(q).item()
    
    def train(self, batch_size=1024):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)

        for s, a, r, ns, d in batch:
            s = torch.tensor(s, dtype=torch.float32)
            ns = torch.tensor(ns, dtype=torch.float32)

            target = r

            if not d:
                target += self.gamma * torch.max(self.model(ns)).item()
            
            pred = self.model(s)

            target_vec = pred.clone().detach()
            target_vec[a] = target
            
            loss = self.loss_fn(pred, target_vec)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

if __name__ == "__main__":
    game = FancySnakeGame(render=False)
    agent = Agent()

    try:
        checkpoint = torch.load("./snake_agent.pth")
        agent.model.load_state_dict(checkpoint["model_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent.epsilon = checkpoint["epsilon"]
        game.episode = checkpoint["episode"]
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("No saved model found, starting fresh training.")

    try:
        with open("./snake_memory.pkl", "rb") as f:
            agent.memory = pickle.load(f)
        print("Memory loaded successfully!")
    except FileNotFoundError:
        print("No saved memory found, starting fresh memory.")

    episodes = 5000

    for ep in range(episodes):
        state = game.reset()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    torch.save({
                        "model_state": agent.model.state_dict(),
                        "optimizer_state": agent.optimizer.state_dict(),
                        "epsilon": agent.epsilon,
                        "episode": game.episode
                    }, "./snake_agent.pth")
                    with open("./snake_memory.pkl", "wb") as f:
                        pickle.dump(agent.memory, f)
                    pygame.quit()
                    exit()
            
            action = agent.act(state)

            next_state, reward, done = game.step(action)

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            game.render()

            if done:
                break
        
        agent.train()

        print(f"Episode: {ep+1} Score: {game.score} Epsilon: {agent.epsilon:.3f}")

        if(ep+1)%100==0:
            torch.save({
                "model_state": agent.model.state_dict(),
                "optimizer_state": agent.optimizer.state_dict(),
                "epsilon": agent.epsilon,
                "episode": game.episode
            }, "./snake_agent.pth")
            with open("./snake_memory.pkl", "wb") as f:
                pickle.dump(agent.memory, f)
            print("Checkpoint saved.")
    
    torch.save({
        "model_state": agent.model.state_dict(),
        "optimizer_state": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "episode": game.episode
    }, "./snake_agent.pth")
    with open("./snake_memory.pkl", "wb") as f:
        pickle.dump(agent.memory, f)

    pygame.quit()