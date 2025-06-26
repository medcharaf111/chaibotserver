import cv2

backends = [
    ("Default", None),
    ("DirectShow", cv2.CAP_DSHOW),
    ("Video for Windows", cv2.CAP_VFW),
    ("Media Foundation", cv2.CAP_MSMF)
]

for backend_name, backend_flag in backends:
    for idx in range(3):
        print(f"\nTrying index {idx} with backend {backend_name}")
        if backend_flag is not None:
            cap = cv2.VideoCapture(idx, backend_flag)
        else:
            cap = cv2.VideoCapture(idx)
        ret, frame = cap.read()
        print(f"ret={ret}, frame is None={frame is None}")
        if ret and frame is not None:
            print("First pixel:", frame[0,0])
            cv2.imshow(f'Camera {idx} Backend {backend_name}', frame)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyAllWindows()
        cap.release() 