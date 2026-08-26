import cv2
import digit_recognizer as dr

img = cv2.imread("debug_crop_birth_date.jpg")
print("image:", None if img is None else img.shape)

try:
    t = dr._get_templates()
    print("templates ok:", sorted(t.keys()))
except Exception as e:
    print("TEMPLATE/FONT ERROR:", e)
    raise SystemExit

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary = dr._binarize(gray)
cv2.imwrite("diag_binary.jpg", binary)
print("black ratio:", round(1.0 - (binary > 0).mean(), 3))

comps = dr.segment_with_boxes(gray)
print("n comps:", len(comps))
for (x, y, w, h, blob) in comps:
    label, score = dr.classify_blob(blob)
    print(f"x={x:3d} y={y:3d} w={w:3d} h={h:3d} -> {label} {score:.2f}")

print("final:", dr.recognize_digits(img))