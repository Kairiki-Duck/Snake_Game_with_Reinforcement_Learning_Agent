import pygame
import sys
import random
import math

pygame.init()

width=600
height=400
block_size=20
screen=pygame.display.set_mode((width,height))
pygame.display.set_caption("Eating Snake")

clock=pygame.time.Clock()
font=pygame.font.SysFont(None,36)
font_bold=pygame.font.SysFont(None,48,bold=True)


snake=[(100,100),(80,100),(60,100)]
dx=block_size
dy=0
food_count=0
frenzy_mode=False

score=0

super_food_timer=None
super_food_duration=8000
current_super_duration=8000

effect_timer=None
effect_duration=1200

particles=[]
floating_texts=[]

frenzy_mode=False
frenzy_timer=0
frenzy_duration=3000
ghost_segments=[]
combo_count=0
combo_timer=0


def create_particles(x,y,color):
    for _ in range(10):
        particles.append([x+10,y+10,random.uniform(-2,2),random.uniform(-2,2),255,color])

def update_fx():
    for p in particles[:]:
        p[0]+=p[2]
        p[1]+=p[3]
        p[4]-=15
        if p[4]<=0:
            particles.remove(p)
    for ft in floating_texts[:]:
        ft[1]-=1
        ft[3]-=10
        if ft[3]<=0:
            floating_texts.remove(ft)

def generate_food():
    all_positions=[(x,y) for x in range(0,width,block_size) for y in range(0,height,block_size)]
    available_positions=[pos for pos in all_positions if pos not in snake]
    return random.choice(available_positions)

food=generate_food()
super_food=None
shrink_food=None

def draw_snake(surface):
    for i,(gx,gy,g_alpha) in enumerate(ghost_segments):
        s=pygame.Surface((block_size,block_size))
        s.set_alpha(g_alpha)
        s.fill((0,255,255) if frenzy_mode else (0,150,255))
        surface.blit(s,(gx,gy))

    for i,segment in enumerate(snake):
        if frenzy_mode:
            color=(0,200+55*math.sin(pygame.time.get_ticks()/50),255)
        else:
            color_val=max(100,255-i*12)
            color=(0,color_val,0) if i>0 else (50,255,150)
        
        pygame.draw.rect(surface,color,(segment[0],segment[1],block_size,block_size))
        if i==0:
            eye_color=(255,255,255) if not frenzy_mode else (255,0,0)
            pygame.draw.circle(surface,eye_color,(segment[0]+14,segment[1]+6),3)
        
        pygame.draw.rect(surface,(0,50,0),(segment[0],segment[1],block_size,block_size),1)

def draw_food(surface):
    pulse=(math.sin(pygame.time.get_ticks()/150)+1)/2
    if food:
        glow_size=int(pulse*6)
        pygame.draw.rect(surface,(100,0,0),(food[0]-glow_size//2,food[1]-glow_size//2,block_size+glow_size,block_size+glow_size))
        pygame.draw.rect(surface,(255,0,0),(food[0],food[1],block_size,block_size))
    if super_food:
        if pygame.time.get_ticks()//400%2==0:
            pygame.draw.rect(surface,(255,215,0),(super_food[0],super_food[1],block_size,block_size))
            pygame.draw.rect(surface,(255,255,255),(super_food[0],super_food[1],block_size,block_size),2)
    if shrink_food:
        size_offset=int(pulse*4)
        pygame.draw.rect(surface,(255,105,180),(shrink_food[0]-size_offset//2,shrink_food[1]-size_offset//2,block_size+size_offset,block_size+size_offset))
        pygame.draw.rect(surface,(255,255,255),(shrink_food[0],shrink_food[1],block_size,block_size),1)

def show_score(surface):
    text=font.render(f"Score:{score}",True,(255,255,255))
    surface.blit(text,(10,10))

while True:

    base_speed = min(10 + food_count // 24,15)

    speed=base_speed*1.5 if frenzy_mode else base_speed

    if pygame.time.get_ticks()>combo_timer:
        combo_count=0

    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_UP and dy==0:
                dx,dy=0,-block_size
            elif event.key==pygame.K_DOWN and dy==0:
                dx,dy=0,block_size
            elif event.key==pygame.K_LEFT and dx==0:
                dx,dy=-block_size,0
            elif event.key==pygame.K_RIGHT and dx==0:
                dx,dy=block_size,0
    
    ghost_segments.append([snake[0][0],snake[0][1],150])
    for g in ghost_segments:
        g[2]-=20
        if g[2]<=0:
            ghost_segments.remove(g)
    
    head_x,head_y=snake[0]
    new_head=(head_x+dx,head_y+dy)

    new_head=(new_head[0]%width,new_head[1]%height)

    if new_head in snake and not frenzy_mode:
        pygame.quit()
        sys.exit()

    snake.insert(0,new_head)

    collision_pos=None
    eat_type=None
    
    if food and new_head==food:
        collision_pos,eat_type=food,"normal"
    elif super_food and new_head==super_food:
        collision_pos,eat_type=super_food,"super"
    elif shrink_food and new_head==shrink_food:
        collision_pos,eat_type=shrink_food,"shrink"
    
    if collision_pos:
        combo_count+=1
        combo_timer=pygame.time.get_ticks()+2000
        base_gain=1
        p_color=(255,0,0)

        if eat_type=="normal":
            if frenzy_mode:
                base_gain=10
                frenzy_timer=pygame.time.get_ticks()+frenzy_duration
                floating_texts.append([collision_pos[0],collision_pos[1]-20,"TIME EXTENDED",255])
                p_color=(0,255,255)
            food_count+=1
            food=None
            if food_count%20==0:
                shrink_food=generate_food()
                super_food_timer=pygame.time.get_ticks()
                current_super_duration=8000
            elif food_count%5==0:
                super_food=generate_food()
                super_food_timer=pygame.time.get_ticks()
                current_super_duration=8000*10/speed
            else:
                food=generate_food()
        
        elif eat_type=="super":
            base_gain=20
            frenzy_mode=True
            frenzy_timer=pygame.time.get_ticks()+frenzy_duration
            p_color=(255,215,0)
            super_food=None
            super_food_timer=None
            food=generate_food()
            effect_timer=pygame.time.get_ticks()
        
        elif eat_type=="shrink":
            base_gain=20
            frenzy_mode=True
            frenzy_timer=pygame.time.get_ticks()+frenzy_duration
            shrink_amount=20
            if len(snake)>shrink_amount+2:
                snake=snake[:-shrink_amount]
            else:
                snake=snake[:3]
            p_color=(255,105,180)
            shrink_food=None
            super_food_timer=None
            food=generate_food()
            effect_timer=pygame.time.get_ticks()
        
        score+=base_gain*combo_count
        create_particles(collision_pos[0],collision_pos[1],p_color)
        floating_texts.append([collision_pos[0],collision_pos[1],f"+{base_gain*combo_count}",255])
    else:
        snake.pop()

    if super_food_timer:
        elapsed=pygame.time.get_ticks() - super_food_timer
        if elapsed>current_super_duration:
            super_food=None
            shrink_food=None
            super_food_timer=None
            food=generate_food()

    if frenzy_mode and pygame.time.get_ticks()>frenzy_timer:
        frenzy_mode=False

    game_surface=pygame.Surface((width,height))

    bg_color=(10,30,40) if frenzy_mode else (0,0,0)
    
    if effect_timer:
        if pygame.time.get_ticks()-effect_timer<400:
            game_surface.fill((255,215,0))
        elif pygame.time.get_ticks()-effect_timer<800:
            game_surface.fill(bg_color)
        elif pygame.time.get_ticks()-effect_timer<effect_duration:
            game_surface.fill((255,215,0))
        else:
            effect_timer=None
            game_surface.fill(bg_color)
    else:
        game_surface.fill(bg_color)
    
    for x in range(0,width,block_size):
        pygame.draw.line(game_surface,(20,20,20),(x,0),(x,height))
    for y in range(0,height,block_size):
        pygame.draw.line(game_surface,(20,20,20),(0,y),(width,y))

    draw_snake(game_surface)
    draw_food(game_surface)
    show_score(game_surface)

    if frenzy_mode:
        current_time=pygame.time.get_ticks()
        frenzy_remaining=max(0,(frenzy_timer-current_time)/frenzy_duration)

        alpha=150+105*math.sin(current_time/100)
        frenzy_text=font_bold.render("FRENZY MODE",True,(0,255,255))
        frenzy_text.set_alpha(alpha)

        text_rect=frenzy_text.get_rect(center=(width//2,45))
        game_surface.blit(frenzy_text,text_rect)

        f_bar_width=200
        f_bar_height=8
        f_bar_x=width//2-f_bar_width//2
        f_bar_y=65

        pygame.draw.rect(game_surface,(0,50,50),(f_bar_x,f_bar_y,f_bar_width,f_bar_height))

        current_f_width=int(f_bar_width*frenzy_remaining)
        if current_f_width>0:
            pygame.draw.rect(game_surface,(0,255,255),(f_bar_x,f_bar_y,current_f_width,f_bar_height))
            pygame.draw.line(game_surface,(200,255,255),(f_bar_x,f_bar_y),(f_bar_x+current_f_width,f_bar_y),1)
        pygame.draw.rect(game_surface,(255,255,255),(f_bar_x,f_bar_y,f_bar_width,f_bar_height),1)

    if combo_count>1:
        c_text=font.render(f"COMBO X{combo_count}",True,(255,255,0))
        game_surface.blit(c_text,(width-150,10))

    update_fx()
    for p in particles:
        s=pygame.Surface((4,4))
        s.set_alpha(p[4])
        s.fill(p[5])
        game_surface.blit(s,(p[0],p[1]))
    for ft in floating_texts:
        t=font.render(ft[2],True,(255,255,255))
        t.set_alpha(ft[3])
        game_surface.blit(t,(ft[0],ft[1]))

    shake_offset=(0,0)

    if (super_food or shrink_food) and super_food_timer:
        current_time=pygame.time.get_ticks()
        elapsed=current_time-super_food_timer
        remaining_ratio=max(0,(current_super_duration-(current_time-super_food_timer))/current_super_duration)
        
        progress_bar_width=220
        progress_bar_height=14
        progress_bar_pos=(width//2-progress_bar_width//2,10)

        progress_length=int(progress_bar_width*remaining_ratio)
        
        ratio=remaining_ratio
        r=int(255*(1-ratio))
        g=int(255*ratio)
        b=0

        pulse=abs(pygame.time.get_ticks()%1000-500)/500
        brightness=0.7+0.3*pulse

        bar_color=(
            int(r*brightness),
            int(g*brightness),
            int(b*brightness)
        )

        pygame.draw.rect(game_surface,(60,60,60),
                         (*progress_bar_pos,progress_bar_width,progress_bar_height))

        glow_surface=pygame.Surface((progress_length,progress_bar_height),pygame.SRCALPHA)
        glow_surface.fill((*bar_color,120))
        game_surface.blit(glow_surface,progress_bar_pos)

        pygame.draw.rect(game_surface,bar_color,
                         (*progress_bar_pos,progress_length,progress_bar_height))
        
        pygame.draw.rect(game_surface,(255,255,255),
                         (*progress_bar_pos,progress_bar_width,progress_bar_height),2)
        
        seconds_left = int((current_super_duration - elapsed) / 1000) + 1
        timer_text = font.render(f"{seconds_left}", True, (255,255,255))
        game_surface.blit(timer_text,
                          (progress_bar_pos[0] + progress_bar_width + 10,
                           progress_bar_pos[1] - 6))
        
        if remaining_ratio < 0.25:
            shake_offset = (random.randint(-3,3), random.randint(-3,3))

        if elapsed > current_super_duration:
            super_food=None
            shrink_food=None
            super_food_timer=None
            food=generate_food()

    
    screen.blit(game_surface, shake_offset)

    pygame.display.flip()

    clock.tick(speed)