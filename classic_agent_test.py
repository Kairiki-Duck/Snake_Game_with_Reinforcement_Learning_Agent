import pygame
import torch
import pickle
from classic_agent_training import SnakeGame,Agent

def test_snake(agent,episodes=10,render=True):
    agent.epsilon=0.0
    game=SnakeGame(render=render)
    for ep in range(episodes):
        state=game.reset()
        total_reward=0

        while True:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    pygame.quit()
                    return
            
            state_tensor=torch.tensor(state,dtype=torch.float32)
            q_values=agent.model(state_tensor)
            action=torch.argmax(q_values).item()

            next_state,reward,done=game.step(action)
            total_reward+=reward
            state=next_state

            score=game.score

            if render:
                game.render()
            
            if done:
                print(f"Episode {ep+1}/{episodes} , Score: {score}")
                break
    
    pygame.quit()

if __name__=="__main__":
    agent=Agent()

    checkpoint=torch.load("./classic_snake_agent.pth")
    agent.model.load_state_dict(checkpoint["model_state"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("Model loaded successfully!")

    test_snake(agent,episodes=10,render=True)
