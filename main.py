import cv2
import mediapipe as mp
import pygame
import random
import sys
import numpy as np

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

meteors = []
METEOR_SPEED = 5
METEOR_SIZE = 40
SPAWN_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(SPAWN_EVENT, 1000)

RED = (255, 0, 0)
WHITE = (255, 255, 255)
GAME_OVER_FONT = pygame.font.SysFont("Arial", 50)
START_FONT = pygame.font.SysFont("Arial", 40)

cap = cv2.VideoCapture(0)

# Comprehensive removal of arm and shoulder landmarks
# This includes shoulder, elbow, and wrist indices for both left and right sides
disabled_pose_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

def is_collision(meteor, hitboxes):
    mx, my = meteor
    for hx, hy in hitboxes:
        if (mx - hx) ** 2 + (my - hy) ** 2 <= METEOR_SIZE ** 2:
            return True
    return False

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
        mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:

    started = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()
            if event.type == SPAWN_EVENT and started:
                side = random.choice(["left", "right"])
                if side == "left":
                    x = 0
                else:
                    x = WIDTH - METEOR_SIZE
                y = random.randint(HEIGHT // 3, 2 * HEIGHT // 3)
                meteors.append([x, y, side])

        keys = pygame.key.get_pressed()
        if not started and keys[pygame.K_RETURN]:
            started = True

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_hands = hands.process(rgb_frame)
        results_pose = pose.process(rgb_frame)

        hand_hitboxes = []
        body_hitboxes = []

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmark_x = int((1 - lm.x) * WIDTH)
                    landmark_y = int(lm.y * HEIGHT)
                    hand_hitboxes.append((landmark_x, landmark_y))

        if results_pose.pose_landmarks:
            for i, lm in enumerate(results_pose.pose_landmarks.landmark):
                if i not in disabled_pose_indices:
                    landmark_x = int((1 - lm.x) * WIDTH)
                    landmark_y = int(lm.y * HEIGHT)
                    body_hitboxes.append((landmark_x, landmark_y))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame_surface = pygame.surfarray.make_surface(frame)
        screen.blit(frame_surface, (0, 0))

        if not started:
            text_surface = START_FONT.render("Press Enter to Start", True, WHITE)
            screen.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, HEIGHT // 2))
        else:
            for meteor in meteors[:]:
                pygame.draw.circle(screen, RED, meteor[:2], METEOR_SIZE)
                if meteor[2] == "left":
                    meteor[0] += METEOR_SPEED
                else:
                    meteor[0] -= METEOR_SPEED

                if is_collision(meteor[:2], hand_hitboxes):
                    meteors.remove(meteor)
                elif is_collision(meteor[:2], body_hitboxes):
                    screen.fill(RED)
                    text_surface = GAME_OVER_FONT.render("Game Over", True, WHITE)
                    screen.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, HEIGHT // 2))
                    pygame.display.flip()
                    pygame.time.wait(2000)
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    sys.exit()
                elif meteor[0] < 0 or meteor[0] > WIDTH:
                    meteors.remove(meteor)

            for hx, hy in hand_hitboxes:
                pygame.draw.circle(screen, WHITE, (hx, hy), 5)

            for bx, by in body_hitboxes:
                pygame.draw.circle(screen, WHITE, (bx, by), 5)

        pygame.display.flip()
        clock.tick(30)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pygame.quit()