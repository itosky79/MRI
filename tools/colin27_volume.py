"""Colin 27 の組織分類を 1.5 mm の立方格子にまとめ、アプリの組織番号のランレングスにする（任意の断面を切るため）。

使い方:  python3 tools/colin27_volume.py colin27_cls_tal_hires.nii head3d.js
格子は Talairach 座標で、番号 (k, j, i) の中心が x = X0 + 1.5 i、y = Y0 + 1.5 j、z = Z0 + 1.5 k（mm）。
値はアプリの TIS の番号 + 1（0 は空気）。1 格子（0.5 mm の 3×3×3 = 27 点）でいちばん多い組織にする。
病変（浮腫）は、側脳室のまわりの白質に楕円体を 3 つ置く（Colin 27 は正常脳のため。以前の 1 断面の描き足しと同じ位置）。
"""
import sys, base64, json
import numpy as np

v = np.memmap(sys.argv[1], np.int16, 'r', offset=352, shape=(362, 434, 362))   # z, y, x（0.5 mm）
lab = np.rint(np.asarray(v) / 21.25).astype(np.int8)
# Colin 27 の番号 → アプリの組織（TIS の順：fat, wm, gm, csf, mus, les, bone。値は番号+1）
MAP = {0: 0, 1: 4, 2: 3, 3: 2, 4: 1, 5: 5, 6: 5, 7: 7, 9: 1, 10: 5, 11: 1, 12: 0}
lut = np.zeros(16, np.int8)
for k, m in MAP.items(): lut[k] = m
app = lut[lab]
F = 3
nz, ny, nx = (s // F for s in app.shape)
blk = app[:nz * F, :ny * F, :nx * F].reshape(nz, F, ny, F, nx, F)
cnt = np.stack([(blk == c).sum(axis=(1, 3, 5)) for c in range(8)])          # 組織ごとの点の数
vol = cnt.argmax(0).astype(np.uint8)
# 0.5 mm の格子の中心は x = 0.5 i − 90.25。3 つまとめた中心は 0.5 (3 I + 1) − 90.25
X0, Y0, Z0 = 0.5 * 1 - 90.25, 0.5 * 1 - 126.25, 0.5 * 1 - 72.25
# 病変：Talairach z = +20 mm の側脳室の中心からのずれ（mm）と半径（以前の 1.25 mm 画素の値を mm に直したもの）
cz = 20.0
k = int(round((cz - Z0) / 1.5))
csf = np.argwhere(vol[k] == 4)
cy_mm, cx_mm = (csf[:, 0].mean() * 1.5 + Y0), (csf[:, 1].mean() * 1.5 + X0)
# 1 断面のときの楕円（画素：右向き dx、下向き dy、半径 rx, ry）。画像の右は患者の左（x の符号反転）、下は後ろ
LES = [(-22, -14, 6, 4.5), (25, -10, 5, 4), (-24, 18, 4.5, 4)]
zz, yy, xx = np.meshgrid(Z0 + 1.5 * np.arange(nz), Y0 + 1.5 * np.arange(ny), X0 + 1.5 * np.arange(nx), indexing='ij')
for dx, dy, rx, ry in LES:
    ex, ey = cx_mm - dx * 1.25, cy_mm - dy * 1.25
    m = ((xx - ex) / (rx * 1.25)) ** 2 + ((yy - ey) / (ry * 1.25)) ** 2 + ((zz - cz) / 6.0) ** 2 < 1
    vol[m & (vol == 2)] = 6
# 頭のある範囲だけに切り詰める
nzv = np.argwhere(vol > 0); lo = nzv.min(0); hi = nzv.max(0) + 1
vol = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
origin = [X0 + 1.5 * lo[2], Y0 + 1.5 * lo[1], Z0 + 1.5 * lo[0]]
f = vol.ravel(); out = bytearray(); i = 0
while i < len(f):
    n = 1
    while i + n < len(f) and f[i + n] == f[i] and n < 255: n += 1
    out += bytes([f[i], n]); i += n
print('shape (z,y,x)', vol.shape, 'origin', origin, 'counts', np.bincount(f, minlength=8), 'rle', len(out))
with open(sys.argv[2], 'w') as fo:
    fo.write('/* Colin 27 Average Brain, high-resolution version 2008 の組織分類を 1.5 mm の格子にまとめたもの（tools/colin27_volume.py）。\n'
             '   Copyright (C) 1993-2009 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University.\n'
             '   Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is hereby granted,\n'
             '   provided that the above copyright notice appear in all copies.\n'
             '   値はアプリの組織番号+1（0 は空気）。z, y, x の順に（値, 個数）のランレングス。病変（浮腫）はこのリポジトリで描き足したもの。 */\n')
    fo.write('const HEAD3D = ' + json.dumps({'nx': int(vol.shape[2]), 'ny': int(vol.shape[1]), 'nz': int(vol.shape[0]), 'd': 1.5,
              'x0': origin[0], 'y0': origin[1], 'z0': origin[2], 'rle': base64.b64encode(bytes(out)).decode()}) + ';\n')
