import pygame
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle

class SnakeGame:
    def __init__(self,width=600,height=400,block_size=20,render=True):
        self.width=width
        self.height=height
        self.block_size=block_size
        self.render_enable=render
        self.episode=0

        pygame.init()

        if self.render_enable:
            self.screen=pygame.display.set_mode((width,height))
            pygame.display.set_caption("Eating Snake")

        self.clock=pygame.time.Clock()
        self.font=pygame.font.SysFont(None,36)

        self.reset()
    
    def reset(self):
        self.snake=[(100,100),(80,100),(60,100)]
        self.dx=self.block_size
        self.dy=0

        self.score=0
        self.food=self.generate_food()

        self.done=False
        self.episode+=1
        return self.get_state()
    
    def generate_food(self):
        all_positions=[(x,y) for x in range(0,self.width,self.block_size) for y in range(0,self.height,self.block_size)]
        available_positions=[pos for pos in all_positions if pos not in self.snake]
        return random.choice(available_positions)
    
    def step(self,action):
        if self.done:
            return self.get_state(),0,True
        
        #action:0-up,1-down,2-left,3-right

        if action==0 and self.dy==0:
            self.dx,self.dy=0,-self.block_size
        elif action==1 and self.dy==0:
            self.dx,self.dy=0,self.block_size
        elif action==2 and self.dx==0:
            self.dx,self.dy=-self.block_size,0
        elif action==3 and self.dx==0:
            self.dx,self.dy=self.block_size,0
        
        head_x,head_y=self.snake[0]
        new_head=(head_x+self.dx,head_y+self.dy)
        new_head=(new_head[0]%self.width,new_head[1]%self.height)

        reward=-0.05

        if new_head in self.snake:
            self.done=True
            reward=-100
            return self.get_state(),reward,True
        
        self.snake.insert(0,new_head)

        if new_head==self.food:
            self.score+=1
            reward=10
            self.food=self.generate_food()
        else:
            self.snake.pop()
        return self.get_state(),reward,False
    
    def get_state(self):
        head_x,head_y=self.snake[0]

        point_l=(head_x-self.block_size,head_y)
        point_r=(head_x+self.block_size,head_y)
        point_u=(head_x,head_y-self.block_size)
        point_d=(head_x,head_y+self.block_size)

        dir_l=self.dx<0
        dir_r=self.dx>0
        dir_u=self.dy<0
        dir_d=self.dy>0

        def danger(point):
            return point in self.snake
        
        state=[
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

            self.food[0]<head_x,
            self.food[0]>head_x,
            self.food[1]<head_y,
            self.food[1]>head_y
        ]
        return list(map(int,state))
    
    def render(self):
        if not self.render_enable:
            return
        
        self.screen.fill((0,0,0))

        for x in range(0,self.width,self.block_size):
            pygame.draw.line(self.screen,(40,40,40),(x,0),(x,self.height))
        for y in range(0,self.height,self.block_size):
            pygame.draw.line(self.screen,(40,40,40),(0,y),(self.width,y))

        pulse=(math.sin(pygame.time.get_ticks()/150)+1)/2
        glow_size=int(pulse*6)
        pygame.draw.rect(self.screen,(100,0,0),(self.food[0]-glow_size//2,self.food[1]-glow_size//2,self.block_size+glow_size,self.block_size+glow_size))
        pygame.draw.rect(self.screen,(255,0,0),(self.food[0],self.food[1],self.block_size,self.block_size))
        
        for i,segment in enumerate(self.snake):
            color_val=max(100,255-i*12)
            color=(0,color_val,0) if i>0 else (50,255,150)
            pygame.draw.rect(
                self.screen,
                color,
                (segment[0],segment[1],self.block_size,self.block_size)
            )
            pygame.draw.rect(self.screen,(0,50,0),(segment[0],segment[1],self.block_size,self.block_size),1)

        pygame.draw.rect(
            self.screen,
            (255,0,0),
            (self.food[0],self.food[1],self.block_size,self.block_size)
        )

        text=self.font.render(f"Score:{self.score}",True,(255,255,255))
        self.screen.blit(text,(10,10))

        text_ep=self.font.render(f"Episode:{self.episode-1}",True,(255,255,255))
        self.screen.blit(text_ep,(10,40))

        pygame.display.flip()
        self.clock.tick(20)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Sequential(
            nn.Linear(11,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,4)
        )

    def forward(self,x):
        return self.fc(x)
    
class Agent:
    def __init__(self):
        self.model=Net()
        self.optimizer=optim.Adam(self.model.parameters(),lr=0.001)
        self.loss_fn=nn.MSELoss()

        self.memory=deque(maxlen=50000)

        self.gamma=0.9

        self.epsilon=1.0
        self.eps_min=0.05
        self.eps_decay=0.997

    def remember(self,s,a,r,ns,d):
        self.memory.append((s,a,r,ns,d))

    def act(self,state):
        if random.random()<self.epsilon:
            return random.randint(0,3)
        
        state=torch.tensor(state,dtype=torch.float32)
        q=self.model(state)

        return torch.argmax(q).item()
    
    def train(self,batch_size=1000):
        if len(self.memory)<batch_size:
            return
        
        batch=random.sample(self.memory,batch_size)

        for s,a,r,ns,d in batch:
            s=torch.tensor(s,dtype=torch.float32)
            ns=torch.tensor(ns,dtype=torch.float32)

            target=r

            if not d:
                target+=self.gamma*torch.max(self.model(ns)).item()
            
            pred=self.model(s)

            target_vec=pred.clone().detach()
            target_vec[a]=target
            
            loss=self.loss_fn(pred,target_vec)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon>self.eps_min:
            self.epsilon*=self.eps_decay

if __name__=="__main__":

    game=SnakeGame(render=True)
    agent=Agent()

    try:
        checkpoint=torch.load("./classic_snake_agent.pth")
        agent.model.load_state_dict(checkpoint["model_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent.epsilon=checkpoint["epsilon"]
        game.episode=checkpoint["episode"]
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("No saved model found, starting fresh training.")

    try:
        with open("./classic_snake_memory.pkl","rb") as f:
            agent.memory=pickle.load(f)
        print("Memory loaded successfully!")
    except FileNotFoundError:
        print("No saved memory found,starting fresh memory.")

    episodes=2000

    for ep in range(episodes):
        state=game.reset()

        while True:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    torch.save({
                        "model_state":agent.model.state_dict(),
                        "optimizer_state":agent.optimizer.state_dict(),
                        "epsilon":agent.epsilon,
                        "episode":game.episode
                    },"./classic_snake_agent.pth")
                    with open("./classic_snake_memory.pkl","wb") as f:
                        pickle.dump(agent.memory,f)
                    pygame.quit()
                    exit()
            
            action=agent.act(state)

            next_state,reward,done=game.step(action)

            agent.remember(state,action,reward,next_state,done)

            state=next_state

            game.render()

            if done:
                break
        
        agent.train()

        print(f"Episode:{ep+1} Score:{game.score} Epsilon:{agent.epsilon:.3f}")

    torch.save({
        "model_state":agent.model.state_dict(),
        "optimizer_state":agent.optimizer.state_dict(),
        "epsilon":agent.epsilon,
        "episode":game.episode
    },"./classic_snake_agent.pth")
    with open("./classic_snake_memory.pkl","wb") as f:
        pickle.dump(agent.memory,f)

    pygame.quit()