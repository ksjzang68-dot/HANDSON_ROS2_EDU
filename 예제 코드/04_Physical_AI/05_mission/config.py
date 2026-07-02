# =========================================
# AGV Tracker — 공통 CONFIG
# =========================================
MASTER_SPEED = 35

MODEL_PATH        = "ai.pt"
CAM_INDEX         = 0
CAM_W, CAM_H      = 640, 480
CONF_THRESHOLD    = 0.4

WHEEL_DIAMETER_MM = 95
FIELD_MM          = 900
FIELD_PX          = 700
PREVIEW_W         = 320
PREVIEW_H         = 240
PANEL_W           = 260
WIN_W             = FIELD_PX + PANEL_W
WIN_H             = FIELD_PX + PREVIEW_H + 10
FPS               = 30
MAX_TRAIL         = 5000

BASE_SPEED        = MASTER_SPEED
STEER_GAIN        = 0.02
STEER_D_GAIN      = 0.01
MAX_STEER         = 40
DEAD_ZONE         = 30
LOST_TIMEOUT      = 1.0
YAW_KP            = 0.5

# 물체별 근접 판정 Y 임계값 (cy가 이 값보다 크면 grab)
CLOSE_Y_THRESH_MAP = {
    "can":     340,
    "paper":   380,
    "plastic": 390,   # 원하는 값으로 조정
}
CLOSE_Y_THRESH_DEFAULT = 370
STABLE_TRACK_SEC  = 2.5
APPROACH_DIST_MM  = 25.0
APPROACH_SPEED    = MASTER_SPEED

LOCK_STABLE_SEC   = 0.8
LOCK_POS_TOLERANCE = 160

DESTINATION_MAP = {
    "can":     (100,   800),
    "paper":   (800, 800),
    "plastic": (750, 100),
}
NAV_SPEED          = MASTER_SPEED
NAV_TURN_SPEED     = MASTER_SPEED
NAV_ARRIVE_THRESH  = 60
NAV_YAW_KP         = 0.5
RELEASE_DURATION   = 3.0
RELEASE_SPEED      = -50

TOTAL_CYCLES        = 3
SPIN_180_SPEED      = MASTER_SPEED+10
SPIN_180_TOLERANCE  = 5
RETURN_SPEED        = MASTER_SPEED
RETURN_TURN_SPEED   = MASTER_SPEED+10
RETURN_ARRIVE_THRESH = 80
RETURN_ORIGIN       = (0, 0)

# ── Colors ────────────────────────────────────────────────────
C_BG        = (6,   10,  15)
C_FIELD     = (7,   17,  29)
C_GRID      = (15,  35,  55)
C_BORDER    = (0,  180, 220)
C_TRAIL     = (0,  220, 255)
C_ROBOT     = (255, 107,  53)
C_START     = (46,  204, 113)
C_PANEL     = (10,  18,  30)
C_TEXT      = (180, 220, 240)
C_DIM       = (60,  100, 130)
C_ACCENT    = (0,   220, 255)
C_ORANGE    = (255, 107,  53)
C_RED       = (220,  60,  60)
C_GREEN     = (46,  204, 113)
C_WHITE     = (255, 255, 255)
C_PREVIEW   = (10,  18,  30)
C_LOCK      = (255, 220,   0)

DEST_COLORS = {
    "can":     (46,  204, 113),
    "paper":   (52,  152, 219),
    "plastic": (155,  89, 182),
}
