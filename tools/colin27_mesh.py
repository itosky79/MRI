"""Colin 27 の組織分類から、3D 表示用の頭皮・脳・脳室の面を作る。

使い方:  python3 tools/colin27_mesh.py colin27_cls_tal_hires.nii brain3d.js
必要なもの:  numpy, scipy, scikit-image, fast_simplification
座標はアプリの 3D シーンに合わせる（1 単位 = 5 mm）。
  x = 画像の右向き（患者の左が画像の右なので Talairach の x の符号を反転）、y = 前、z = 頭側（B0 の向き）。
  断面（Talairach z = +20 mm）が z = 0、画像の中心（x = 0, y = −18 mm）が原点になる。
"""
import sys, base64, json
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
import fast_simplification

U = 5.0
v = np.memmap(sys.argv[1], np.int16, 'r', offset=352, shape=(362, 434, 362))   # z, y, x（0.5 mm）
lab = np.rint(np.asarray(v) / 21.25).astype(np.int8)                         # 組織番号に戻す（21.25 倍で保存されている）
F = 4                                                                          # 2 mm に間引く
def down(mask):
    z, y, x = (s // F * F for s in mask.shape)
    m = mask[:z, :y, :x].reshape(z // F, F, y // F, F, x // F, F)
    return m.mean(axis=(1, 3, 5)).astype(np.float32)

head = down(lab > 0)
brain_b = (lab == 2) | (lab == 3)
brain_b = ndi.binary_fill_holes(ndi.binary_closing(brain_b, iterations=2))
brain = down(brain_b)
# 脳室：脳脊髄液を細いところで切り離し（オープニング 3 回）、側脳室の高さの中央にあるかたまりを選んで、
# 切った分を脳脊髄液の中だけで戻す
csf = lab == 1
cc, n = ndi.label(ndi.binary_opening(csf, iterations=3))
cent = np.array(ndi.center_of_mass(cc > 0, cc, range(1, n + 1)))            # z, y, x の番号
tal = np.stack([0.5 * cent[:, 2] - 90.25, 0.5 * cent[:, 1] - 126.25, 0.5 * cent[:, 0] - 72.25], 1)
sizes = ndi.sum(cc > 0, cc, range(1, n + 1))
score = np.linalg.norm(tal - [0, -15, 15], axis=1) - (sizes > 40000) * 1000   # 20 mL 以上で中央に近いもの
keep = cc == 1 + int(np.argmin(score))
keep = ndi.binary_dilation(keep, iterations=3, mask=csf)
vent = down(keep)

def mesh(vol, sigma, target):
    vol = ndi.gaussian_filter(np.pad(vol, 2), sigma)
    verts, faces, _, _ = measure.marching_cubes(vol, 0.5)
    verts -= 2
    if len(faces) > target:
        verts, faces = fast_simplification.simplify(verts.astype(np.float32), faces.astype(np.int32), 1 - target / len(faces))
    k, j, i = verts[:, 0], verts[:, 1], verts[:, 2]                   # 2 mm の格子の番号（ブロックの中心）
    xt = 0.5 * (F * i + (F - 1) / 2) - 90.25
    yt = 0.5 * (F * j + (F - 1) / 2) - 126.25
    zt = 0.5 * (F * k + (F - 1) / 2) - 72.25
    p = np.stack([-xt / U, (yt + 18) / U, (zt - 20) / U], 1)
    faces = faces[:, [0, 2, 1]]                                       # x の反転で裏返った面の向きを戻す
    q = np.round(p * 500).astype(np.int16)
    return {'p': base64.b64encode(q.tobytes()).decode(), 'i': base64.b64encode(faces.astype(np.uint16 if len(p) < 65536 else np.uint32).tobytes()).decode(),
            'n32': bool(len(p) >= 65536), 'nv': int(len(p)), 'nf': int(len(faces))}

out = {'head': mesh(head, 1.2, 9000), 'brain': mesh(brain, 0.9, 30000), 'vent': mesh(vent, 0.8, 4000)}
for k, m in out.items(): print(k, m['nv'], m['nf'])
with open(sys.argv[2], 'w') as f:
    f.write('/* Colin 27 Average Brain, high-resolution version 2008 の組織分類から作った頭皮・脳・脳室の面（tools/colin27_mesh.py）。\n'
            '   Copyright (C) 1993-2009 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University.\n'
            '   Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is hereby granted,\n'
            '   provided that the above copyright notice appear in all copies.\n'
            '   座標は 1 単位 = 5 mm を 500 倍した int16。面は頂点番号の 3 つ組。 */\n')
    f.write('const BRAIN3D = ' + json.dumps(out) + ';\n')
