import cv2
import numpy as np
import time
from pyModbusTCP.client import ModbusClient

def detect_color(hsv, color_ranges):
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 0:  # 마스크 내에 색상 픽셀이 있는지 확인
            return color
    else_mask = cv2.inRange(hsv, np.array(else_range[0]), np.array(else_range[1]))
    if cv2.countNonZero(else_mask) > 0:
        return 'else'    
    return None

def rotate_frame(frame, angle):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h))
    return rotated

# 웹캠 초기화
cap = cv2.VideoCapture(1)  # 0번 카메라 장치 사용

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 초기 프레임 버퍼 비우기
for _ in range(5):
    cap.read()

color_ranges = {
    'red': ([0, 120, 70], [10, 255, 255]),
    'yellow': ([20, 100, 100], [40, 255, 255]),
    'green': ([40, 40, 40], [80, 255, 255]),
    'blue': ([100, 150, 0], [140, 255, 255])
}

# 'else' 범위 설정 (지정된 색상 범위를 제외한 범위)
else_range = ([0, 0, 0], [180, 255, 255])

color_codes = {'red': 1, 'yellow': 2, 'green': 3, 'blue': 4, 'else': 0}

# Modbus 클라이언트 초기화
c = ModbusClient(host="192.168.213.74", port=502, unit_id=1, auto_open=True)

# 창 이름 설정 및 크기 조정
window_name = 'Masked Frame'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 320, 240)  # 원하는 크기로 변경

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기 조정 (화면 축소)
    frame = cv2.resize(frame, (320, 240))  # 원하는 크기로 변경

    # ROI 설정
    x, y, w, h = 200, 130, 200, 270  # ROI 좌표 및 크기 (축소된 화면에 맞게 변경)
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ROI 외부를 어둡게 처리하기(mask)
    mask = np.zeros_like(frame)
    cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)

    # 마스크 적용
    masked_add = cv2.addWeighted(frame, 0.5, mask, 0.5, 0) 

    # ROI에 녹색 사각형 그리기
    cv2.rectangle(masked_add, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 색상 식별
    detected_color = detect_color(hsv_roi, color_ranges)
    if detected_color:
        print(f"Detected color: {detected_color}")
        # Modbus로 전송할 값
        modbus_value = color_codes[detected_color]
        print(f"Modbus Value: {modbus_value}")
        
        # Modbus 통신으로 값 전송
        if not c.write_single_register(128, modbus_value):
            print("Modbus write error")

    # 영상 시작 q로 종료    
    cv2.imshow(window_name, masked_add)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()