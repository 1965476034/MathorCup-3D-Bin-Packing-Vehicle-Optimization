from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import 问题二_求解器 as solver

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, '问题二最佳方案摆放图及空间载重利用率数据')
PNG_DIR = os.path.join(OUT_DIR, '货车图片')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

TRUCK_NAME_ASCII = {
    '车型1': '车型1',
    '车型2': '车型2',
}

BOX_COLORS = {
    'G1': '#4C78A8',
    'G2': '#F58518',
    'G3': '#E45756',
    'G4': '#54A24B',
    'G5': '#B279A2',
}

PAPER_DPI = 220
FIGSIZE = (13, 9)


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f'没有可写入的数据: {path}')
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 3D PNG 渲染模块（仅本地 matplotlib）
# ============================================================

def cuboid_faces(x: int, y: int, z: int, dx: int, dy: int, dz: int):
    p000 = (x, y, z)
    p100 = (x + dx, y, z)
    p110 = (x + dx, y + dy, z)
    p010 = (x, y + dy, z)
    p001 = (x, y, z + dz)
    p101 = (x + dx, y, z + dz)
    p111 = (x + dx, y + dy, z + dz)
    p011 = (x, y + dy, z + dz)
    return [
        [p000, p100, p110, p010],
        [p001, p101, p111, p011],
        [p000, p100, p101, p001],
        [p010, p110, p111, p011],
        [p000, p010, p011, p001],
        [p100, p110, p111, p101],
    ]


def draw_truck_wireframe(ax, L: int, W: int, H: int) -> None:
    corners = [
        (0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0),
        (0, 0, H), (L, 0, H), (L, W, H), (0, W, H),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        xs = [corners[i][0], corners[j][0]]
        ys = [corners[i][1], corners[j][1]]
        zs = [corners[i][2], corners[j][2]]
        ax.plot(xs, ys, zs, color='black', linewidth=0.9)


def style_3d_axes(ax) -> None:
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        except Exception:
            pass
    try:
        ax.xaxis._axinfo['grid']['linewidth'] = 0.0
        ax.yaxis._axinfo['grid']['linewidth'] = 0.0
        ax.zaxis._axinfo['grid']['linewidth'] = 0.0
    except Exception:
        pass


def render_truck_png(packer: solver.SurfacePacker, out_png: str, title: str) -> Dict[str, object]:
    fig = plt.figure(figsize=FIGSIZE, dpi=PAPER_DPI)
    ax = fig.add_subplot(111, projection='3d')

    for p in sorted(packer.placed, key=lambda box: (box.z, box.x, box.y, box.h)):
        faces = cuboid_faces(p.x, p.y, p.z, p.l, p.w, p.h)
        poly = Poly3DCollection(
            faces,
            facecolors=BOX_COLORS.get(p.type_code, '#999999'),
            edgecolors='k',
            linewidths=0.18,
            alpha=0.68,
        )
        ax.add_collection3d(poly)

    L, W, H = packer.truck.L, packer.truck.W, packer.truck.effective_H
    draw_truck_wireframe(ax, L, W, H)
    style_3d_axes(ax)

    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    ax.set_box_aspect((L, W, H))
    ax.set_xlabel('x轴：从右后到前方（厘米）', labelpad=8)
    ax.set_ylabel('y轴：从右侧到左侧（厘米）', labelpad=8)
    ax.set_zlabel('z轴：从下到上（厘米）', labelpad=8)
    ax.view_init(elev=23, azim=-58)

    util = packer.utilization()
    truck_label = TRUCK_NAME_ASCII.get(packer.truck.name, packer.truck.name)
    counts_text = '  '.join(f'{code}={packer.loaded_counts[code]}' for code in solver.ITEM_ORDER)
    info_text = (
        f"车型：{truck_label}\n"
        f"装载件数：{len(packer.placed)}\n"
        f"有效空间利用率：{util['空间利用率_eff']:.4f}\n"
        f"载重利用率：{util['载重利用率']:.4f}\n"
        f"单车运输成本：{packer.truck.trip_cost:.0f} 元\n"
        f"已装体积：{util['已装货物体积_m3']:.3f} 立方米\n"
        f"已装重量：{util['已装货物重量_kg']:.1f} 千克\n"
        f"各类货物数量：{counts_text}"
    )

    legend_handles = [
        Patch(facecolor=BOX_COLORS[code], edgecolor='k', label=code, alpha=0.68)
        for code in solver.ITEM_ORDER
        if packer.loaded_counts.get(code, 0) > 0
    ]
    if legend_handles:
        fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.985, 0.88), frameon=True)

    fig.text(
        0.81,
        0.43,
        info_text,
        ha='left',
        va='top',
        fontsize=10.5,
        bbox=dict(boxstyle='round,pad=0.45', facecolor='white', edgecolor='#666666', alpha=0.95),
    )

    fig.suptitle(title, y=0.97, fontsize=14)
    plt.tight_layout(rect=(0.0, 0.0, 0.79, 0.95))
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)

    return {
        '车型': packer.truck.name,
        '单车运输成本': packer.truck.trip_cost,
        '装载件数': len(packer.placed),
        '空间利用率_eff': util['空间利用率_eff'],
        '载重利用率': util['载重利用率'],
        '已装货物体积_m3': util['已装货物体积_m3'],
        '已装货物重量_kg': util['已装货物重量_kg'],
    }


def export_pngs_for_scheme(packers: List[solver.SurfacePacker], scheme_slug: str, scheme_title: str) -> List[Dict[str, object]]:
    scheme_dir = os.path.join(PNG_DIR, scheme_slug)
    os.makedirs(scheme_dir, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    for i, packer in enumerate(packers, start=1):
        filename = f'{scheme_slug}_第{i:02d}辆车.png'
        abs_path = os.path.join(scheme_dir, filename)
        title = f'{scheme_title} - 第{i}辆车'
        stats = render_truck_png(packer, abs_path, title)
        manifest_rows.append({
            '方案标识': scheme_slug,
            '方案名称': scheme_title,
            '车辆序号': i,
            '图片文件名': filename,
            '图片相对路径': os.path.relpath(abs_path, OUT_DIR),
            **stats,
        })
    return manifest_rows


def bundle_to_overview_row(bundle: Dict[str, object]) -> Dict[str, object]:
    packers = bundle['packers']
    type_counts = bundle['truck_type_counts']
    scheme_name_map = {
        'Q2(1)_min_total_trucks': '问题2（1）总车辆数最少方案',
        'Q2(1)_min_total_trucks_fallback': '问题2（1）总车辆数最少方案',
        'Q2(2)_min_total_cost': '问题2（2）总运输成本最低方案',
        'Q2(2)_min_total_cost_fallback': '问题2（2）总运输成本最低方案',
        'Q2(2)_min_total_cost_upper_bound': '问题2（2）总运输成本最低方案',
        'Q2(2)_min_total_cost_stage1': '问题2（2）总运输成本最低方案',
    }
    return {
        '方案名称': scheme_name_map.get(bundle['scheme_name'], bundle['scheme_name']),
        '总车辆数': bundle['truck_count'],
        '车型1数量': type_counts.get('车型1', 0),
        '车型2数量': type_counts.get('车型2', 0),
        '总运输成本': bundle['total_cost'],
        '最小空间利用率_eff': bundle['min_空间利用率_eff'],
        '平均空间利用率_eff': bundle['avg_空间利用率_eff'],
        '最小载重利用率': bundle['min_载重利用率'],
        '平均载重利用率': bundle['avg_载重利用率'],
        '最小最小利用率': bundle['min_min利用率'],
        '平均最小利用率': bundle['avg_min利用率'],
        '总装载体积_m3': sum(p.total_volume for p in packers) / 1e6,
        '总装载重量_kg': sum(p.total_weight for p in packers),
    }


def vehicle_summary_rows(bundle: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    scheme_name_map = {
        'Q2(1)_min_total_trucks': '问题2（1）总车辆数最少方案',
        'Q2(1)_min_total_trucks_fallback': '问题2（1）总车辆数最少方案',
        'Q2(2)_min_total_cost': '问题2（2）总运输成本最低方案',
        'Q2(2)_min_total_cost_fallback': '问题2（2）总运输成本最低方案',
        'Q2(2)_min_total_cost_upper_bound': '问题2（2）总运输成本最低方案',
        'Q2(2)_min_total_cost_stage1': '问题2（2）总运输成本最低方案',
    }
    for idx, packer in enumerate(bundle['packers'], start=1):
        util = packer.utilization()
        row = {
            '方案名称': scheme_name_map.get(bundle['scheme_name'], bundle['scheme_name']),
            '车辆序号': idx,
            '车型': packer.truck.name,
            '单车运输成本': packer.truck.trip_cost,
            '装载件数': len(packer.placed),
            '空间利用率_raw': util['空间利用率_raw'],
            '空间利用率_eff': util['空间利用率_eff'],
            '载重利用率': util['载重利用率'],
            '已装货物体积_m3': util['已装货物体积_m3'],
            '已装货物重量_kg': util['已装货物重量_kg'],
        }
        for code in solver.ITEM_ORDER:
            row[f'{code}_数量'] = packer.loaded_counts[code]
        rows.append(row)
    return rows


def result_json_view(result: Dict[str, object]) -> Dict[str, object]:
    scheme_name_map = {
        'Q2(1)_min_total_trucks': '问题2（1）总车辆数最少方案',
        'Q2(1)_min_total_trucks_fallback': '问题2（1）总车辆数最少方案',
        'Q2(2)_min_total_cost': '问题2（2）总运输成本最低方案',
        'Q2(2)_min_total_cost_fallback': '问题2（2）总运输成本最低方案',
        'Q2(2)_min_total_cost_upper_bound': '问题2（2）总运输成本最低方案',
        'Q2(2)_min_total_cost_stage1': '问题2（2）总运输成本最低方案',
    }
    comparison = result['comparison']
    out: Dict[str, object] = {
        '方案对比': {
            '总车辆数最少方案_车辆数': comparison['min_total_trucks_vehicle_count'],
            '总车辆数最少方案_总成本': comparison['min_total_trucks_total_cost'],
            '总车辆数最少方案_车型数量': comparison['min_total_trucks_type_counts'],
            '总成本最低方案_车辆数': comparison['min_total_cost_vehicle_count'],
            '总成本最低方案_总成本': comparison['min_total_cost_total_cost'],
            '总成本最低方案_车型数量': comparison['min_total_cost_type_counts'],
            '两方案成本差额_总车辆数最少减总成本最低': comparison['cost_gap_min_trucks_minus_min_cost'],
            '两方案车辆数差额_总成本最低减总车辆数最少': comparison['vehicle_gap_min_cost_minus_min_trucks'],
        }
    }
    key_name_map = {
        'min_total_trucks': '总车辆数最少方案',
        'min_total_cost': '总运输成本最低方案',
    }
    for key in ('min_total_trucks', 'min_total_cost'):
        bundle = result[key]
        out[key_name_map[key]] = {
            '方案名称': scheme_name_map.get(bundle['scheme_name'], bundle['scheme_name']),
            '车辆数': bundle['truck_count'],
            '总运输成本': bundle['total_cost'],
            '车型数量': bundle['truck_type_counts'],
            '最小空间利用率_eff': bundle['min_空间利用率_eff'],
            '平均空间利用率_eff': bundle['avg_空间利用率_eff'],
            '最小载重利用率': bundle['min_载重利用率'],
            '平均载重利用率': bundle['avg_载重利用率'],
            '最小最小利用率': bundle['min_min利用率'],
            '平均最小利用率': bundle['avg_min利用率'],
            '车辆明细': [
                {
                    '车型': p.truck.name,
                    '单车运输成本': p.truck.trip_cost,
                    '利用率': p.utilization(),
                    '装载数量': p.loaded_counts,
                    '装载件数': len(p.placed),
                }
                for p in bundle['packers']
            ],
        }
    return out


if __name__ == '__main__':
    result = solver.solve_question2(fragile_fixed=False)
    catalog = result['catalog']

    scheme_min_trucks = result['min_total_trucks']
    scheme_min_cost = result['min_total_cost']

    # 明细坐标方案导出
    write_csv(
        os.path.join(OUT_DIR, '问题2_总车辆数最少方案_装载明细.csv'),
        solver.solution_rows_from_packers(scheme_min_trucks['packers'], catalog),
    )
    write_csv(
        os.path.join(OUT_DIR, '问题2_总运输成本最低方案_装载明细.csv'),
        solver.solution_rows_from_packers(scheme_min_cost['packers'], catalog),
    )

    # 方案概览与逐车汇总
    overview_rows = [
        bundle_to_overview_row(scheme_min_trucks),
        bundle_to_overview_row(scheme_min_cost),
    ]
    write_csv(os.path.join(OUT_DIR, '问题2_方案概览.csv'), overview_rows)

    vehicle_rows = []
    vehicle_rows.extend(vehicle_summary_rows(scheme_min_trucks))
    vehicle_rows.extend(vehicle_summary_rows(scheme_min_cost))
    write_csv(os.path.join(OUT_DIR, '问题2_逐车汇总.csv'), vehicle_rows)

    # JSON 汇总
    with open(os.path.join(OUT_DIR, '问题2_结果汇总.json'), 'w', encoding='utf-8') as f:
        json.dump(result_json_view(result), f, ensure_ascii=False, indent=2)

    # PNG 导出
    png_manifest: List[Dict[str, object]] = []
    png_manifest.extend(export_pngs_for_scheme(scheme_min_trucks['packers'], '总车辆数最少方案', '问题2（1）总车辆数最少方案'))
    png_manifest.extend(export_pngs_for_scheme(scheme_min_cost['packers'], '总运输成本最低方案', '问题2（2）总运输成本最低方案'))
    write_csv(os.path.join(OUT_DIR, '问题2_图片清单.csv'), png_manifest)

    print('已生成问题2输出文件：')
    for name in [
        '问题2_总车辆数最少方案_装载明细.csv',
        '问题2_总运输成本最低方案_装载明细.csv',
        '问题2_方案概览.csv',
        '问题2_逐车汇总.csv',
        '问题2_结果汇总.json',
        '问题2_图片清单.csv',
    ]:
        print(f'- {os.path.join(OUT_DIR, name)}')
    print(f'- 货车图片目录：{PNG_DIR}')
