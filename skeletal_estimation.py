import cv2
import mediapipe as mp
import json

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ビデオファイルを開く
cap = cv2.VideoCapture('dance_sample1.mp4')

# ポーズデータを格納するリスト
pose_data = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 画像を処理
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # ポーズの検出
        results = pose.process(image)

        # フレームに結果を描画
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # ポーズのキーポイントを抽出
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility,
                })
            pose_data.append(keypoints)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 結果を表示
        cv2.imshow('Frame', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

# ポーズデータをJSONファイルに保存
with open('pose_data.json', 'w') as f:
    json.dump(pose_data, f)
