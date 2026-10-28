import gzip, sys, base64, numpy as np
from PIL import Image
v = np.frombuffer(gzip.open(sys.argv[1]).read(), np.uint8).reshape(181, 217, 181)  # z, y, x（Talairach、1 mm）
Z = 92  # z = +20 mm：側脳室の体部が大きく見える高さ
sl = v[Z]
IW, IH, PX = 176, 208, 1.25
# BrainWeb の組織番号 → アプリの組織（TIS の順：fat, wm, gm, csf, mus, les, bone。値は番号+1、0 は空気）
MAP = {0: 0, 1: 4, 2: 3, 3: 2, 4: 1, 5: 5, 6: 5, 7: 7, 8: 3, 9: 5}
lab = np.zeros((IH, IW), np.uint8)
for j in range(IH):
    for i in range(IW):
        xm, ym = (i + 0.5 - IW / 2) * PX, (j + 0.5 - IH / 2) * PX
        xi, yi = int(round(90 - xm)), int(round(108 - ym))   # 画像の左が患者の右（放射線科の表示）、上が前
        if 0 <= xi < 181 and 0 <= yi < 217: lab[j, i] = MAP[int(sl[yi, xi])]
# 病変（浮腫）：側脳室のまわりの白質に楕円を置く（BrainWeb の正常脳には病変がないため、アプリで描き足す）
csf = lab == 4
cy, cx = np.argwhere(csf[60:150, 60:116]).mean(0) + [60, 60]
LES = [(-22, -14, 6, 4.5), (25, -10, 5, 4), (-24, 18, 4.5, 4)]   # 側脳室の中心からのずれ（画素）と半径
yy, xx = np.mgrid[0:IH, 0:IW]
for dx, dy, rx, ry in LES:
    m = ((xx - (cx + dx)) / rx) ** 2 + ((yy - (cy + dy)) / ry) ** 2 < 1
    lab[m & (lab == 2)] = 6
print('ventricle center', cx, cy, 'counts', np.bincount(lab.ravel(), minlength=8))
# ランレングス（値, 個数 ≤ 255）を base64 にする
out = bytearray(); f = lab.ravel(); k = 0
while k < len(f):
    n = 1
    while k + n < len(f) and f[k + n] == f[k] and n < 255: n += 1
    out += bytes([f[k], n]); k += n
b64 = base64.b64encode(bytes(out)).decode()
open(sys.argv[2], 'w').write(b64); print('rle bytes', len(out), 'b64', len(b64))
pal = np.array([(0,0,0),(240,200,90),(235,235,235),(150,150,150),(70,140,220),(180,90,90),(200,120,255),(90,90,70)], np.uint8)
Image.fromarray(pal[lab]).resize((IW * 3, IH * 3), Image.NEAREST).save(sys.argv[3])
