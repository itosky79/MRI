import sys, base64, numpy as np
from PIL import Image
# Colin 27 高解像度版（2008）の離散の組織分類 colin27_cls_tal_hires.nii（NIfTI、int16、0.5 mm、362×434×362、Talairach）
# 座標は x = 0.5 i − 90.25、y = 0.5 j − 126.25、z = 0.5 k − 72.25（mm）。配列は z, y, x の順に並ぶ
v = np.memmap(sys.argv[1], np.int16, 'r', offset=352, shape=(362, 434, 362))
ZMM = 20.0   # z = +20 mm：側脳室の体部が大きく見える高さ（以前の BrainWeb の断面と同じ）
sl = np.rint(np.array(v[int(round((ZMM + 72.25) / 0.5))]) / 21.25).astype(int)   # NIfTI 版の組織番号は 21.25 倍で保存されている（21→1, 43→2, …, 255→12）
IW, IH, PX = 176, 208, 1.25
# Colin 27 の組織番号 → アプリの組織（TIS の順：fat, wm, gm, csf, mus, les, bone。値は番号+1、0 は空気）
# 1 脳脊髄液, 2 灰白質, 3 白質, 4 脂肪, 5 筋肉, 6 皮膚と筋肉, 7 頭蓋骨, 9 脂肪2, 10 硬膜, 11 骨髄, 12 血管
# 硬膜は結合組織として筋肉に、骨髄は脂肪髄として脂肪に、血管は流れる血液の信号が抜ける（flow void）として信号なしにする
MAP = {0: 0, 1: 4, 2: 3, 3: 2, 4: 1, 5: 5, 6: 5, 7: 7, 9: 1, 10: 5, 11: 1, 12: 0}
lut = np.zeros(16, np.uint8)
for k, m in MAP.items(): lut[k] = m
lab = np.zeros((IH, IW), np.uint8)
SUB = (np.arange(3) + 0.5) / 3 - 0.5   # 1 画素（1.25 mm）の中を 3×3 点で調べ、いちばん多い組織にする
for j in range(IH):
    for i in range(IW):
        votes = np.zeros(8, int)
        for sy in SUB:
            for sx in SUB:
                xm, ym = (i + 0.5 + sx - IW / 2) * PX, (j + 0.5 + sy - IH / 2) * PX
                x, y = -xm, -18 - ym   # 画像の左が患者の右（放射線科の表示）、上が前
                xi, yi = int(round((x + 90.25) / 0.5)), int(round((y + 126.25) / 0.5))
                votes[lut[sl[yi, xi]] if 0 <= xi < 362 and 0 <= yi < 434 else 0] += 1
        lab[j, i] = votes.argmax()
# 病変（浮腫）：側脳室のまわりの白質に楕円を置く（Colin 27 は正常脳で病変がないため、アプリで描き足す）
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
