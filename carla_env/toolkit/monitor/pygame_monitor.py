import cv2
import carla
import atexit
import functools
import math
import os
import pygame
import numpy as np
import datetime
from carla_env.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="pygame_monitor")

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

class HelpText(object):
    def __init__(self, font, width, height):
        lines = "CarDreamer Eval".split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)

class DataTableRender(object):
    def __init__(self, font, width, pos_bottom_center=True):
        self.font = font
        self.screen_width = width
        self.pos_bottom_center = pos_bottom_center
        self.col_widths = [200, 150, 180, 100] # Label, Current, Benchmark, Status
        self.headers = ["Metric", "Value", "Threshold/Unit", "Status"]

    def render(self, display, table_data, height):
        if not table_data:
            return

        row_height = 20
        padding = 10
        total_height = (len(table_data) + 1) * row_height + padding * 2
        total_width = sum(self.col_widths) + padding * 2

        # Calculate Position (Center Top)
        x = (self.screen_width - total_width) // 2 if self.pos_bottom_center else 10
        y = 20 # Add static top padding

        # Draw background
        surface = pygame.Surface((total_width, total_height), pygame.SRCALPHA)
        pygame.draw.rect(surface, (0, 0, 0, 180), surface.get_rect())
        pygame.draw.rect(surface, (200, 200, 200), surface.get_rect(), 1) # Border
        display.blit(surface, (x, y))

        # Draw Headers
        curr_x = x + padding
        curr_y = y + padding
        for i, header in enumerate(self.headers):
            text_texture = self.font.render(header, True, (255, 215, 0)) # Gold
            display.blit(text_texture, (curr_x, curr_y))
            curr_x += self.col_widths[i]

        curr_y += row_height
        pygame.draw.line(display, (200, 200, 200), (x + padding, curr_y), (x + total_width - padding, curr_y), 1)
        curr_y += 5

        # Draw Data Rows
        for row in table_data:
            curr_x = x + padding
            for i, cell in enumerate(row):
                color = (255, 255, 255)
                # Status color overrides
                if i == 3:
                    if "EXCEEDED" in str(cell) or "DANGER" in str(cell): color = (255, 50, 50)
                    elif "OK" in str(cell) or "SAFE" in str(cell): color = (50, 255, 50)

                text_texture = self.font.render(str(cell), True, color)
                display.blit(text_texture, (curr_x, curr_y))
                curr_x += self.col_widths[i]
            curr_y += row_height

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.data_table = DataTableRender(self._font_mono, width, pos_bottom_center=True)
        self.episode_count = 1
        self.episode_reward = 0.0

    def reset_episode(self, episode_num):
        self.episode_count = episode_num
        self.episode_reward = 0.0

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, obs, info, clock):
        self._notifications.tick(None, clock)
        if not self._show_info:
            return

        # Parse info
        speed = info.get("speed_norm", 0.0) * 3.6 # km/h
        acc = info.get("acceleration", [0,0,0])
        acc_mag = np.linalg.norm(acc) if isinstance(acc, list) else 0.0
        loc = info.get("location", [0,0,0])

        # Controls
        ctrl_throttle = info.get("throttle", 0.0)
        ctrl_steer = info.get("steer", 0.0)
        ctrl_brake = info.get("brake", 0.0)

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % "CarDreamer Agent",
            'Mode:    % 20s' % info.get("drive_mode", "AUTOPILOT"),
            'Map:     % 20s' % info.get("town", "Unknown"),
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (info.get("compass", 0.0), ''),
            'Accel:   % 15.1f m/s2' % acc_mag,
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (loc[0], loc[1])),
            'Height:  % 18.0f m' % loc[2],
            ''
        ]

        self._info_text += [
            ('Throttle:', ctrl_throttle, 0.0, 1.0),
            ('Steer:', ctrl_steer, -1.0, 1.0),
            ('Brake:',    ctrl_brake, 0.0, 1.0),
            ('Reverse:',  False),
            # ('Hand brake:', False),
            ''
        ]

        # Add tasks specific info if available
        if "reward" in info:
             self.episode_reward += info["reward"] # Tích luỹ điểm để show lên dòng Acc Reward bên kia!
             
             self._info_text.append('')
             self._info_text.append('── REWARD BREAKDOWN ──')
             if 'r_speed' in info:
                 self._info_text.append(' T1_Collision:% 15.2f' % info.get("r_collision", 0.0))
                 self._info_text.append(' T1_Proximity:% 15.2f' % info.get("r_proximity", 0.0))
                 self._info_text.append(' T2_StopLight:% 15.2f' % info.get("r_traffic_light", 0.0))
                 self._info_text.append(' T2_StopSign: % 15.2f' % info.get("r_stop_sign", 0.0))
                 self._info_text.append(' T2_Standstil:% 15.2f' % info.get("r_standstill", 0.0))
                 self._info_text.append(' T2_Speeding: % 15.2f' % info.get("r_speeding", 0.0))
                 self._info_text.append(' T3_Waypoints:% 15.2f' % info.get("r_waypoints", 0.0))
                 self._info_text.append(' T3_Speed:    % 15.2f' % info.get("r_speed", 0.0))
                 self._info_text.append(' T3_OutLane:  % 15.2f' % info.get("r_out_of_lane", 0.0))
                 self._info_text.append(' T3_Center:   % 15.2f' % info.get("r_centerline", 0.0))
                 self._info_text.append(' T3_Heading:  % 15.2f' % info.get("r_heading", 0.0))
                 self._info_text.append(' T3_Dest:     % 15.2f' % info.get("r_destination", 0.0))
                 self._info_text.append(' T4_Smooth:   % 15.2f' % info.get("r_smoothness", 0.0))
                 self._info_text.append(' Time_Penalty:% 15.2f' % info.get("time_penalty", 0.0))
                         
        if "collision" in info:
             self._info_text.append('Collision:% 18s' % ("YES" if info["collision"] else "NO"))

        # Prepare Table Data
        self._table_data = []

        # 1. Episode / Reward
        # If 'reward' is not in info, default to 0.0
        r_val = info.get('reward', 0.0)
        self._table_data.append(["Episode", str(self.episode_count), "Count", "OK"])
        self._table_data.append(["Inst Reward", f"{r_val:.3f}", "Points", "OK"])
        self._table_data.append(["Acc Reward", f"{self.episode_reward:.3f}", "Episode Return", "OK"])

        # 2. Autonomous Research Variables (Speed, Controls, Tracking)
        speed_kmh = info.get("speed_norm", 0.0) * 3.6
        if speed_kmh < 0.5: speed_status = "STOPPED"
        elif speed_kmh > 45: speed_status = "FAST"
        else: speed_status = "CRUISING"
        self._table_data.append(["Speed", f"{speed_kmh:.1f}", "km/h", speed_status])

        throttle = info.get("throttle", 0.0)
        steer = info.get("steer", 0.0)
        brake = info.get("brake", 0.0)
        act_stat = "BRAKING" if brake > 0.05 else ("TURNING" if abs(steer) > 0.1 else "DRIVING")
        self._table_data.append(["Control", f"T:{throttle:.2f} S:{steer:.2f} B:{brake:.2f}", "Thr/Str/Brk", act_stat])

        wpt_dist = info.get("wpt_dist", 0.0)
        self._table_data.append(["Cross Track (WPT)", f"{wpt_dist:.2f}", "Meters", "SAFE" if wpt_dist < 2.0 else "DANGER"])

        # 3. Distance Tracking
        sum_dist = info.get("sum_travel_distance", 0.0)
        succ_dist = info.get("success_dist", 100.0)
        dist_status = "OK" if sum_dist < succ_dist else "SUCCESS"
        self._table_data.append(["Travel Dist", f"{sum_dist:.2f}", f"Target: {succ_dist:.2f} m", dist_status])

        # 3. Time Tracking
        time_elapsed = info.get("timesteps", 0)
        time_limit = info.get("time_limit", 1000)
        time_status = "OK" if time_elapsed < time_limit else "EXCEEDED"
        self._table_data.append(["Time Steps", str(time_elapsed), f"Limit: {time_limit}", time_status])

        # 4. Proximity & TTC
        ttc = info.get("ttc", 0.0)
        if ttc > 90.0 or ttc <= 0.0: ttc_str = "INF"
        else: ttc_str = f"{ttc:.2f} s"
        ttc_status = "DANGER" if 0 < ttc < 2.5 else "SAFE"
        self._table_data.append(["TTC", ttc_str, "Safe > 2.5s", ttc_status])

        dist_front = info.get("dist_to_front", -1.0)
        if dist_front < 0:
            df_str, df_stat = "CLEAR", "SAFE"
        else:
            df_str = f"{dist_front:.2f} m"
            df_stat = "DANGER" if dist_front < 10.0 else "SAFE"
        self._table_data.append(["Dist to Front", df_str, "Safe > 10m", df_stat])

        # 5. Traffic Light Status
        red_light = info.get("is_red_light", False)
        tl_str = "RED" if red_light else "GREEN/NONE"
        tl_stat = "DANGER" if red_light else "SAFE"
        self._table_data.append(["Traffic Light", tl_str, "Signal", tl_stat])

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((280, self.dim[1]), pygame.SRCALPHA)
            pygame.draw.rect(info_surface, (0, 0, 0, 150), info_surface.get_rect())
            display.blit(info_surface, (0, 0))

            v_offset = 10
            bar_h_offset = 130
            bar_width = 120
            bar_height = 8

            white = (255, 255, 255)
            green = (0, 200, 0)
            red   = (200, 0, 0)

            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    label = item[0]
                    value = item[1]
                    if isinstance(value, bool):
                        spacing = 15
                        surface = self._font_mono.render(f"{label:<{spacing}} {'■' if value else '□'}", True, white)
                        display.blit(surface, (8, v_offset))
                        item = None # Text already drawn
                    else:
                        min_val, max_val = item[2], item[3]

                        # Draw label
                        surface = self._font_mono.render(label, True, white)
                        display.blit(surface, (8, v_offset))

                        # Draw Border
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 5), (bar_width, bar_height))
                        pygame.draw.rect(display, white, rect_border, 1)

                        if label == "Steer:":
                            steer_val = max(min(float(value), 1.0), -1.0)
                            center_x = bar_h_offset + bar_width // 2
                            if steer_val >= 0:
                                fill_w = int((bar_width // 2) * steer_val)
                                rect_fill = pygame.Rect((center_x, v_offset + 5), (fill_w, bar_height))
                            else:
                                fill_w = int(-(bar_width // 2) * steer_val)
                                rect_fill = pygame.Rect((center_x - fill_w, v_offset + 5), (fill_w, bar_height))
                            pygame.draw.rect(display, green, rect_fill)

                        elif label == "Brake:":
                            brake_val = max(min(float(value), 1.0), 0.0)
                            fill_w = int(bar_width * brake_val)
                            rect_fill = pygame.Rect((bar_h_offset, v_offset + 5), (fill_w, bar_height))
                            pygame.draw.rect(display, red, rect_fill)

                        else:  # Throttle or anything else positive
                            val = max(min(float(value), 1.0), 0.0)
                            fill_w = int(bar_width * val)
                            rect_fill = pygame.Rect((bar_h_offset, v_offset + 5), (fill_w, bar_height))
                            pygame.draw.rect(display, green, rect_fill)

                        item = None # Text drawn manually above

                if item:  # At this point has to be a str
                    surface = self._font_mono.render(item, True, white)
                    display.blit(surface, (8, v_offset))
                v_offset += 18

            # Render Advanced HUD Table
            if hasattr(self, '_table_data'):
                self.data_table.render(display, self._table_data, self.dim[1])

        self._notifications.render(display)
        self.help.render(display)

class EnvMonitorPygame:
    def __init__(self, config):
        self._config = config

        # Headless mode check
        if not self._config.display.get("show_window", True):
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()
        pygame.font.init()

        # Determine display size
        self.width = self._config.display.get("width", 1280)
        self.height = self._config.display.get("height", 720)

        # For High-Res mode, we expect 'camera_display' key.
        if "image_size" in self._config.display:
             if isinstance(self._config.display.image_size, int):
                  self.width = self._config.display.image_size
                  self.height = int(self.width * 9 / 16) # Assume 16:9 aspect ratio roughly if single int
             elif isinstance(self._config.display.image_size, list):
                  pass

        # Set explicitly if hires
        if self._config.display.get("hires", False):
            self.width = 1280
            self.height = 720

        # Optimize: Use hardware surface and reduce flags for speed
        self.display = pygame.display.set_mode((self.width, self.height),
                                              pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.ASYNCBLIT)
        pygame.display.set_caption("CarDreamer Evaluation (Manual-style)")
        # Enable hardware acceleration
        os.environ['SDL_VIDEODRIVER'] = 'x11'

        self.hud = HUD(self.width, self.height)
        self.clock = pygame.time.Clock()

        # ULTRA SMOOTH: Frame timing optimization for corner smoothness
        self.last_render_time = 0
        self.frame_skip_threshold = 8  # Skip if too fast

        # Video Recording Setup (Deferred to reset)
        self._video_writer = None
        self._episode_count = 0
        self._save_video = self._config.display.get("save_video", False)
        self._task_name = self._config.get("task_name", "unknown_task")
        self._save_dir = os.path.join("evaluation_videos", self._task_name)
        if self._save_video:
            os.makedirs(self._save_dir, exist_ok=True)

        # Atexit
        atexit.register(self.close)

    def reset(self):
        """Called at start of episode to rotate video file."""
        log.info("Monitor: Resetting video recording...")

        if self._video_writer is not None:
            self._video_writer.release()
            log.info("Monitor: Released old video writer")

        if self._save_video:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # MJPG is much faster/robust than mp4v on systems with limited codecs
            filename = os.path.join(self._save_dir, f"episode_{self._episode_count:03d}_{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self._video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (self.width, self.height))
            log.info(f"Monitor: Started new recording to: {filename}")

        self._episode_count += 1
        self.hud.reset_episode(self._episode_count)

        log.info("Monitor: Reset complete.")

    @staticmethod
    def _clip_line_to_frame(p0, p1, w, h):
        """Clip a segment to image bounds using Liang-Barsky."""
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dx = x1 - x0
        dy = y1 - y0
        p = (-dx, dx, -dy, dy)
        q = (x0, (w - 1) - x0, y0, (h - 1) - y0)
        u1, u2 = 0.0, 1.0

        for pk, qk in zip(p, q):
            if pk == 0:
                if qk < 0: return None
                continue
            r = qk / pk
            if pk < 0:
                if r > u2: return None
                u1 = max(u1, r)
            else:
                if r < u1: return None
                u2 = min(u2, r)

        c0 = (int(round(x0 + u1 * dx)), int(round(y0 + u1 * dy)))
        c1 = (int(round(x0 + u2 * dx)), int(round(y0 + u2 * dy)))
        return c0, c1

    def _draw_pixel_waypoints(self, frame, pixel_wp,
                              line_color=(0, 255, 0),
                              edge_color=(0, 200, 255),
                              thickness=2):
        """Draw projected waypoints and keep off-screen segments visible via frame-edge clipping."""
        if frame is None or pixel_wp is None: return frame
        if not frame.flags.writeable: frame = frame.copy()

        pts = np.asarray(pixel_wp, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 1: return frame

        h, w = frame.shape[:2]
        xy = pts[:, :2]
        valid = np.isfinite(xy).all(axis=1)

        for i in range(len(xy) - 1):
            if not (valid[i] and valid[i + 1]): continue
            seg = self._clip_line_to_frame(xy[i], xy[i + 1], w, h)
            if seg is None: continue
            cv2.line(frame, seg[0], seg[1], line_color, thickness, cv2.LINE_AA)

        # SKIP DOTS: Only draw path lines, no individual waypoint dots
        # Remove dots to reduce visual clutter while keeping guidance path

        return frame

    def render(self, obs, info):
        # ULTRA FAST: Prioritize FPS over visual quality
        target_fps = 30  # Lower to 30 FPS for stable performance
        frame_time = self.clock.tick(target_fps)

        # Skip expensive operations every other frame
        if not hasattr(self, '_frame_counter'):
            self._frame_counter = 0
        self._frame_counter += 1

        # 1. Get Main Image (Hires Camera)
        render_keys = self._config.display.render_keys
        main_img = None

        # Logic to pick the best camera
        if 'camera_display' in obs:
            main_img = obs['camera_display']
        elif 'eval_camera_display' in obs:
            main_img = obs['eval_camera_display']
        elif 'camera_display' in info:
            main_img = info['camera_display']
        elif 'eval_camera_display' in info:
            main_img = info['eval_camera_display']
        elif 'camera' in obs:
             main_img = obs['camera']
        elif 'eval_camera' in obs:
             main_img = obs['eval_camera']

        # Logic to pick Map (BEV)
        map_img = None
        if 'birdeye_display' in obs:
             map_img = obs['birdeye_display']
        elif 'birdeye_display' in info:
             map_img = info['birdeye_display']
        elif 'birdeye_wpt' in obs:
             map_img = obs['birdeye_wpt']

        if main_img is not None:
             # ULTRA OPTIMIZE: Cache resize and use fastest interpolation
             if main_img.shape[1] != self.width or main_img.shape[0] != self.height:
                  # Use LINEAR for better quality with minimal performance cost
                  main_img = cv2.resize(main_img, (self.width, self.height),
                                      interpolation=cv2.INTER_LINEAR)

             # ---- ENABLE WAYPOINT PATH LINES (NO DOTS, JUST PATH) ----
             if "waypoints" in info and "ego_transform" in info:
                 wpts = info["waypoints"]
                 ego_tf = info["ego_transform"]

                 # Camera intrinsic matrix K (approximated for self.width x self.height, FOV 120)
                 cx = self.width / 2.0
                 cy = self.height / 2.0
                 f = cx / math.tan(120.0 * math.pi / 360.0)

                 pixel_wp = []
                 for wp in wpts:
                     wp_loc = carla.Location(float(wp[0]), float(wp[1]), 0.0)
                     local_loc = ego_tf.inverse_transform(wp_loc)

                     # Camera is at x=1.5, z=2.0 (offset according to configs/common.yaml)
                     cam_x = local_loc.x - 1.5
                     cam_y = local_loc.y
                     cam_z = local_loc.z - 2.0

                     # Ensure point is in front of camera
                     if cam_x > 0.1:
                         # CARLA Left-handed: +Y is right, +Z is up, +X is forward.
                         # u = cx + f * (cam_y / cam_x)
                         # v = cy - f * (cam_z / cam_x)
                         u = cx + f * (cam_y / cam_x)
                         v = cy - f * (cam_z / cam_x)
                         pixel_wp.append([u, v])

                 if len(pixel_wp) > 0:
                     main_img = self._draw_pixel_waypoints(main_img, pixel_wp)
             # ------------------------------------------

             surface = pygame.surfarray.make_surface(main_img.swapaxes(0, 1))
             self.display.blit(surface, (0, 0))

        # Draw BEV map picture-in-picture
        if map_img is not None:
             # Check format.
             map_surf = pygame.surfarray.make_surface(map_img.swapaxes(0, 1))
             # Resize map
             map_h = int(self.height * 0.3)
             map_w = int(map_h)
             map_surf = pygame.transform.smoothscale(map_surf, (map_w, map_h))
             self.display.blit(map_surf, (self.width - map_w - 20, 20))

        # Update HUD
        self.hud.simulation_time = getattr(self, "sim_time", 0.0) + 0.1
        self.hud.tick(obs, info, self.clock)
        self.hud.render(self.display)

        # ULTRA SMOOTH: Use update instead of flip for better sync
        # Only update dirty regions to reduce tearing/stuttering
        pygame.display.update()

        # Record Frame
        if self._video_writer is not None:
            # Pygame surface to numpy array (W, H, 3) -> (H, W, 3)
            # view3d sets (w, h, 3).
            # array3d returns (w, h, 3).
            # Opencv expects (h, w, 3) BGR.
            # Pygame is usually RGB.
            pixels = pygame.surfarray.array3d(self.display)
            pixels = pixels.swapaxes(0, 1) # (W, H, 3) -> (H, W, 3)
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            self._video_writer.write(pixels)

        # Handle events to keep window responsive and expose them
        self.last_events = pygame.event.get()
        for event in self.last_events:
             if event.type == pygame.QUIT:
                  self.close()

        # Check for episode end to finalize video immediately
        if info.get("terminal", False) or info.get("is_last", False):
             if self._video_writer is not None:
                 log.info("Episode terminal: Finalizing video.")
                 self._video_writer.release()
                 self._video_writer = None

    def close(self):
        if hasattr(self, '_video_writer') and self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        pygame.quit()
