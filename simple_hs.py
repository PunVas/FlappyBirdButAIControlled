import sys, time, random, pygame
from collections import deque
import cv2 as cv
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

# Pygame setup
pygame.init()

try:
    display_info = pygame.display.get_desktop_display_mode()
    FPS = display_info.refresh_rate if display_info.refresh_rate > 0 else 60
except:
    FPS = 60  # Fallback if detection fails

clock = pygame.time.Clock()

# OpenCV webcam
VID_CAP = cv.VideoCapture(0)
if not VID_CAP.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

# Get webcam resolution
window_size = (
    int(VID_CAP.get(cv.CAP_PROP_FRAME_WIDTH)),
    int(VID_CAP.get(cv.CAP_PROP_FRAME_HEIGHT))
)
screen = pygame.display.set_mode(window_size, pygame.FULLSCREEN)
pygame.display.set_caption("Flappy Bird Face Control")

# Load and scale bird
bird_img = pygame.image.load("bird_sprite.png")
bird_img = pygame.transform.scale(bird_img, (bird_img.get_width() // 6, bird_img.get_height() // 6))
bird_frame = bird_img.get_rect()
bird_frame.center = (window_size[0] // 6, window_size[1] // 2)

# Load and prepare pipes
pipe_img = pygame.image.load("pipe_sprite_single.png")
pipe_template = pipe_img.get_rect()
pipe_frames = deque()
space_between_pipes = int(bird_img.get_height() * 2.5)

# High score functionality
HIGH_SCORE_FILE = "highscore.txt"

def load_high_score():
    """Loads the high score from a file."""
    if os.path.exists(HIGH_SCORE_FILE):
        with open(HIGH_SCORE_FILE, "r") as file:
            try:
                return int(file.read())
            except (ValueError, IOError):
                return 0
    return 0

def save_high_score(new_score):
    """Saves a new high score to a file."""
    with open(HIGH_SCORE_FILE, "w") as file:
        file.write(str(new_score))

high_score = load_high_score()

# Game variables
def reset_game():
    global stage, pipeSpawnTimer, time_between_pipe_spawn, score, didUpdateScore, game_clock, game_is_running
    pipe_frames.clear()
    bird_frame.center = (window_size[0] // 6, window_size[1] // 2)
    stage = 1
    pipeSpawnTimer = 0
    time_between_pipe_spawn = 40
    score = 0
    didUpdateScore = False
    game_clock = time.time()
    game_is_running = True

reset_game()

pipe_velocity = lambda: dist_between_pipes / time_between_pipe_spawn
dist_between_pipes = 500
level = 0

# Text rendering helper
def render_text(txt, size, pos):
    font = pygame.font.SysFont("Helvetica", size, bold=True)
    text = font.render(txt, True, (99, 245, 255))
    rect = text.get_rect(center=pos)
    screen.blit(text, rect)

# Run FaceMesh
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        if not game_is_running:
            screen.fill((0, 0, 0))
            render_text('Game Over!', 64, (window_size[0] // 2, window_size[1] // 2 - 100))
            render_text(f'Score: {score}', 48, (window_size[0] // 2, window_size[1] // 2 - 30))
            render_text(f'High Score: {high_score}', 48, (window_size[0] // 2, window_size[1] // 2 + 30))
            render_text('Press SPACE to Restart', 36, (window_size[0] // 2, window_size[1] // 2 + 100))
            pygame.display.update()

            waiting_for_restart = True
            while waiting_for_restart:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        VID_CAP.release()
                        cv.destroyAllWindows()
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            reset_game()
                            waiting_for_restart = False
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Update high score on quit
                if score > high_score:
                    save_high_score(score)
                game_is_running = False
                
        if not game_is_running:
            break

        ret, frame = VID_CAP.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)
        frame = cv.resize(frame, window_size)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmark = results.multi_face_landmarks[0].landmark[94]
            y_pos = (landmark.y - 0.5) * 1.5 * window_size[1] + window_size[1] / 2
            bird_frame.centery = max(0, min(window_size[1] - bird_frame.height, int(y_pos)))

        # Update pipes
        for pf in pipe_frames:
            pf[0].x -= pipe_velocity()
            pf[1].x -= pipe_velocity()

        if pipe_frames and pipe_frames[0][0].right < 0:
            pipe_frames.popleft()

        # Draw webcam as background
        screen.fill((0, 0, 0))
        frame_surface = pygame.surfarray.make_surface(np.rot90(rgb_frame))
        screen.blit(frame_surface, (0, 0))

        # Draw bird
        screen.blit(bird_img, bird_frame)

        # Draw and check pipes
        passed_pipe = True
        for top, bottom in pipe_frames:
            screen.blit(pygame.transform.flip(pipe_img, False, True), top)
            screen.blit(pipe_img, bottom)

            if top.left <= bird_frame.centerx <= top.right:
                passed_pipe = False
                if not didUpdateScore:
                    score += 1
                    didUpdateScore = True

            if bird_frame.colliderect(top) or bird_frame.colliderect(bottom):
                game_is_running = False

        if passed_pipe:
            didUpdateScore = False

        # Spawn new pipes
        if pipeSpawnTimer == 0:
            top_pipe = pipe_template.copy()
            top_pipe.x = window_size[0]
            top_pipe.y = random.randint(-900, window_size[1] - space_between_pipes - 1000)

            bottom_pipe = pipe_template.copy()
            bottom_pipe.x = window_size[0]
            bottom_pipe.y = top_pipe.y + 1000 + space_between_pipes

            pipe_frames.append([top_pipe, bottom_pipe])

        pipeSpawnTimer += 1
        if pipeSpawnTimer >= time_between_pipe_spawn:
            pipeSpawnTimer = 0

        # UI
        render_text(f"Stage {stage}", 25, (50, 25))
        render_text(f"Score: {score}", 25, (50, 60))
        render_text(f"High Score: {high_score}", 25, (75, 95)) # Display high score

        # Increase difficulty
        if time.time() - game_clock >= 10:
            time_between_pipe_spawn = max(10, time_between_pipe_spawn * 5 // 6)
            stage += 1
            game_clock = time.time()
        
        # Check and update high score
        if score > high_score:
            high_score = score

        pygame.display.flip()
        clock.tick(FPS)

# Cleanup
if score > high_score:
    save_high_score(score)
VID_CAP.release()
cv.destroyAllWindows()
pygame.quit()
sys.exit()
