from __future__ import annotations

import copy
import math

import csv
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# 全局参数
SAFETY_GAP_CM = 3
BEARING_LIMIT_KG_PER_M2 = 500.0
BEARING_LIMIT_KG_PER_CM2 = BEARING_LIMIT_KG_PER_M2 / 10000.0  # 0.05 kg / cm^2
MAX_SURFACES = 400
CANDIDATE_SURFACE_LIMIT = 140
DIRECTIONAL_PROJECTION_RATIO = 1.0  # 保守处理：定向件必须整底完全投影在支撑面内
ITEM_ORDER = ("G1", "G2", "G3", "G4", "G5")
FRAGILE_CODE = "G3"
STANDARD_SUPPORT_BUILDERS = ("G1", "G2")
FRAGILE_PLATFORM_LOOKAHEAD_DEPTH = 4
FRAGILE_TOP_LINEAR_BONUS = 4.0
FRAGILE_TOP_QUADRATIC_BONUS = 6.0
FRAGILE_NEAR_TOP_BONUS = 2.0
FRAGILE_NEAR_TOP_MARGIN_CM = 20
COMPOSITE_SURFACE_PAIR_LIMIT = 120


# 一、数据结构


@dataclass(frozen=True)
class TruckType:
    name: str
    L: int
    W: int
    H: int
    max_weight: float
    trip_cost: float

    @property
    def effective_H(self) -> int:
        return self.H - SAFETY_GAP_CM

    @property
    def raw_volume(self) -> int:
        return self.L * self.W * self.H

    @property
    def eff_volume(self) -> int:
        return self.L * self.W * self.effective_H


@dataclass(frozen=True)
class ItemType:
    code: str
    category: str  # standard / fragile / directional
    dims: Tuple[int, int, int]
    weight: float
    quantity: int

    @property
    def volume(self) -> int:
        a, b, c = self.dims
        return a * b * c


@dataclass
class PlacedBox:
    pid: int
    type_code: str
    category: str
    weight: float
    x: int
    y: int
    z: int
    l: int
    w: int
    h: int
    orientation_name: str
    parent_pid: Optional[int] = None
    support_pids: Tuple[int, ...] = ()
    load_on_top: float = 0.0

    @property
    def volume(self) -> int:
        return self.l * self.w * self.h

    @property
    def top_area_cm2(self) -> int:
        return self.l * self.w

    @property
    def bearing_capacity(self) -> float:
        if self.category == "fragile":
            return 0.0
        return BEARING_LIMIT_KG_PER_CM2 * self.top_area_cm2


@dataclass(frozen=True)
class Surface:
    x: int
    y: int
    z: int
    L: int
    W: int
    parent_pid: Optional[int] = None
    support_pids: Tuple[int, ...] = ()
    source_surfaces: Tuple["Surface", ...] = ()

    @property
    def area(self) -> int:
        return self.L * self.W


@dataclass(frozen=True)
class PackingPolicy:
    name: str
    mode: str  # balanced / volume / weight / product
    category_bonus: Dict[str, float] = field(default_factory=dict)


# 二、基础数据（从附件中提取）
def build_problem_data() -> Tuple[List[TruckType], Dict[str, ItemType]]:
    def _find_file(filename: str) -> Path:
        candidates = [
            Path(filename),
            Path.cwd() / filename,
            Path("/mnt/data") / filename,   # 当前对话附件环境
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"未找到附件文件: {filename}")

    truck_file = _find_file("附件1_车型数据.csv")
    cargo_file = _find_file("附件1_货物数据.csv")

    trucks = []
    with open(truck_file, "r", encoding="gbk", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 保持原来名称风格：车型1、车型2
            truck_name = row["车型名称"].split("_")[0] if row["车型名称"] else f"车型{row['车型编号']}"
            trucks.append(
                TruckType(
                    truck_name,
                    int(row["长(cm)"]),
                    int(row["宽(cm)"]),
                    int(row["高(cm)"]),
                    int(row["额定载重(kg)"]),
                    float(row["单次运输成本(元)"]),
                )
            )

    type_mapping = {
        "标准件": "standard",
        "易碎件": "fragile",
        "定向件": "directional",
    }

    item_types = {}
    with open(cargo_file, "r", encoding="gbk", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row["货物编号"]
            item_types[item_id] = ItemType(
                item_id,
                type_mapping[row["类型"]],
                (
                    int(row["长(cm)"]),
                    int(row["宽(cm)"]),
                    int(row["高(cm)"]),
                ),
                float(row["单重(kg)"]),
                int(row["数量"]),
            )

    return trucks, item_types


# 三、姿态枚举与计数工具
@lru_cache(maxsize=None)
def unique_orientations(dims: Tuple[int, int, int], category: str, fragile_fixed: bool = False) -> Tuple[Tuple[int, int, int, str], ...]:
    if category == "directional":
        return ((dims[0], dims[1], dims[2], f"{dims[0]}x{dims[1]}x{dims[2]}"),)

    if category == "fragile" and fragile_fixed:
        return ((dims[0], dims[1], dims[2], f"{dims[0]}x{dims[1]}x{dims[2]}"),)

    out = []
    for p in set(permutations(dims, 3)):
        out.append((p[0], p[1], p[2], f"{p[0]}x{p[1]}x{p[2]}"))

    # 优先大底面、低高度
    out.sort(key=lambda x: (-(x[0] * x[1]), x[2], -max(x[0], x[1]), -min(x[0], x[1])))
    return tuple(out)


def counts_from_catalog(catalog: Dict[str, ItemType]) -> Dict[str, int]:
    return {code: catalog[code].quantity for code in ITEM_ORDER}


def counts_tuple(counts: Dict[str, int]) -> Tuple[int, ...]:
    return tuple(counts.get(code, 0) for code in ITEM_ORDER)


def counts_dict(counts_t: Tuple[int, ...]) -> Dict[str, int]:
    return {code: counts_t[i] for i, code in enumerate(ITEM_ORDER)}


def subtract_counts(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    out = {}
    for code in ITEM_ORDER:
        v = a.get(code, 0) - b.get(code, 0)
        if v < 0:
            raise ValueError(f"计数相减出现负数: {code}")
        out[code] = v
    return out


def is_zero_counts(counts: Dict[str, int]) -> bool:
    return all(counts.get(code, 0) == 0 for code in ITEM_ORDER)


# 四、下界：最少车辆数的理论下界
def total_remaining_volume(counts: Dict[str, int], catalog: Dict[str, ItemType]) -> int:
    return sum(counts[code] * catalog[code].volume for code in ITEM_ORDER)


def total_remaining_weight(counts: Dict[str, int], catalog: Dict[str, ItemType]) -> float:
    return sum(counts[code] * catalog[code].weight for code in ITEM_ORDER)


def volume_weight_lower_bound(counts: Dict[str, int], catalog: Dict[str, ItemType], truck: TruckType) -> int:
    total_vol = total_remaining_volume(counts, catalog)
    total_wt = total_remaining_weight(counts, catalog)
    return max(
        math.ceil(total_vol / truck.eff_volume),
        math.ceil(total_wt / truck.max_weight),
    )


# 五、核心装箱器：表面分割 + 货类计数驱动
class SurfacePacker:
    def __init__(
        self,
        truck: TruckType,
        catalog: Dict[str, ItemType],
        remaining_counts: Dict[str, int],
        policy: PackingPolicy,
        fragile_fixed: bool = False,
        projection_ratio: float = DIRECTIONAL_PROJECTION_RATIO,
    ):
        self.truck = truck
        self.catalog = catalog
        self.policy = policy
        self.fragile_fixed = fragile_fixed
        self.projection_ratio = projection_ratio

        self.remaining_counts = {code: remaining_counts.get(code, 0) for code in ITEM_ORDER}
        self.loaded_counts = {code: 0 for code in ITEM_ORDER}

        self.placed: List[PlacedBox] = []
        self.by_pid: Dict[int, PlacedBox] = {}
        self.next_pid = 1

        self.total_weight = 0.0
        self.total_volume = 0

        # 初始唯一表面：车厢底板
        self.surfaces: List[Surface] = [Surface(0, 0, 0, truck.L, truck.W, None)]

        # 同一装箱状态下，大量候选会反复查询“该表面还能否继续抬高成 G3 平台”。
        # 用轻量缓存避免在每次打分时重复枚举。
        self._top_fragile_fit_cache: Dict[Tuple[int, int, int], bool] = {}
        self._fragile_final_top_cache: Dict[Tuple[int, int, int], int] = {}
        self._future_fragile_platform_cache: Dict[Tuple[int, int, int, int, int, Optional[int], Tuple[int, ...]], int] = {}
        self._recursive_fragile_top_cache: Dict[Tuple[int, int, int, int], int] = {}

    # 基础几何与稳定性检查
    def rectangle_overlap_area(
        self,
        ax: int,
        ay: int,
        al: int,
        aw: int,
        bx: int,
        by: int,
        bl: int,
        bw: int,
    ) -> int:
        x_overlap = max(0, min(ax + al, bx + bl) - max(ax, bx))
        y_overlap = max(0, min(ay + aw, by + bw) - max(ay, by))
        return x_overlap * y_overlap

    def centroid_inside_parent(self, x: int, y: int, l: int, w: int, parent: PlacedBox) -> bool:
        cx = x + l / 2.0
        cy = y + w / 2.0
        return (parent.x <= cx <= parent.x + parent.l) and (parent.y <= cy <= parent.y + parent.w)

    def projection_ratio_ok(self, spec: ItemType, x: int, y: int, l: int, w: int, parent: Optional[PlacedBox]) -> bool:
        if parent is None:
            return True

        overlap = self.rectangle_overlap_area(x, y, l, w, parent.x, parent.y, parent.l, parent.w)
        ratio = overlap / (l * w)

        # 定向件新增“投影面积约束”：采用保守的“整底完整支撑”策略
        if spec.category == "directional" and ratio + 1e-12 < self.projection_ratio:
            return False

        # 若支撑体本身是定向件，则上层货物同样要求完整落在其投影范围内，避免悬挑
        if parent.category == "directional" and ratio + 1e-12 < self.projection_ratio:
            return False

        return True

    def surface_support_boxes(self, surface: Surface) -> List[PlacedBox]:
        support_boxes: List[PlacedBox] = []
        if surface.support_pids:
            for pid in surface.support_pids:
                box = self.by_pid.get(pid)
                if box is not None:
                    support_boxes.append(box)
            return support_boxes
        if surface.parent_pid is not None:
            parent = self.by_pid.get(surface.parent_pid)
            if parent is not None:
                support_boxes.append(parent)
        return support_boxes

    def surface_is_standard_support(self, surface: Surface) -> bool:
        boxes = self.surface_support_boxes(surface)
        return bool(boxes) and all(box.category == "standard" for box in boxes)

    def compute_support_loads(self, item_weight: float, surface: Surface, x: int, y: int, l: int, w: int) -> Optional[Dict[int, float]]:
        boxes = self.surface_support_boxes(surface)
        if not boxes:
            return {}
        if len(boxes) == 1:
            return {boxes[0].pid: item_weight}

        overlaps: Dict[int, int] = {}
        total_overlap = 0
        for box in boxes:
            ov = self.rectangle_overlap_area(x, y, l, w, box.x, box.y, box.l, box.w)
            if ov > 0:
                overlaps[box.pid] = ov
                total_overlap += ov
        if total_overlap < l * w:
            return None
        return {pid: item_weight * ov / total_overlap for pid, ov in overlaps.items()}

    def can_support(self, spec: ItemType, surface: Surface, l: int, w: int) -> bool:
        support_boxes = self.surface_support_boxes(surface)

        # 落地总是允许
        if not support_boxes:
            return True

        # 任一支撑箱为易碎件都不允许
        if any(box.category == "fragile" for box in support_boxes):
            return False

        # 组合面仅给 G3 使用；且其支撑箱必须全部为标准件
        if len(support_boxes) >= 2:
            if spec.category != "fragile":
                return False
            if not all(box.category == "standard" for box in support_boxes):
                return False
        else:
            parent = support_boxes[0]
            if spec.category == "fragile" and parent.category != "standard":
                return False
            if not self.centroid_inside_parent(surface.x, surface.y, l, w, parent):
                return False
            if not self.projection_ratio_ok(spec, surface.x, surface.y, l, w, parent):
                return False

        load_map = self.compute_support_loads(spec.weight, surface, surface.x, surface.y, l, w)
        if load_map is None:
            return False

        # 承重链约束：组合面按覆盖面积分摊重量
        for pid, delta in load_map.items():
            cur = self.by_pid.get(pid)
            while cur is not None:
                if cur.load_on_top + delta > cur.bearing_capacity + 1e-12:
                    return False
                cur = self.by_pid.get(cur.parent_pid)
        return True

    def apply_load_update(self, load_map: Dict[int, float]) -> None:
        for pid, delta in load_map.items():
            cur = self.by_pid.get(pid)
            while cur is not None:
                cur.load_on_top += delta
                cur = self.by_pid.get(cur.parent_pid)

    def reset_state_caches(self) -> None:
        self._top_fragile_fit_cache.clear()
        self._fragile_final_top_cache.clear()
        self._future_fragile_platform_cache.clear()
        self._recursive_fragile_top_cache.clear()

    def surface_cache_key(self, surface: Surface) -> Tuple[int, int, int, int, int, Optional[int], Tuple[int, ...]]:
        return (surface.x, surface.y, surface.z, surface.L, surface.W, surface.parent_pid, tuple(surface.support_pids))

    def remaining_fragile_exists(self) -> bool:
        return self.remaining_counts.get(FRAGILE_CODE, 0) > 0

    def remaining_standard_support_builder_exists(self) -> bool:
        return any(self.remaining_counts.get(code, 0) > 0 for code in STANDARD_SUPPORT_BUILDERS)

    def best_fragile_final_top_z_for_top(self, L: int, W: int, top_z: int) -> int:
        key = (L, W, top_z)
        if key in self._fragile_final_top_cache:
            return self._fragile_final_top_cache[key]

        fragile = self.catalog[FRAGILE_CODE]
        best_final_top = -1
        for fl, fw, fh, _ in unique_orientations(fragile.dims, fragile.category, self.fragile_fixed):
            if fl <= L and fw <= W and top_z + fh <= self.truck.effective_H:
                best_final_top = max(best_final_top, top_z + fh)
        self._fragile_final_top_cache[key] = best_final_top
        return best_final_top

    def _best_recursive_fragile_top_z_from_top(self, L: int, W: int, top_z: int, depth: int) -> int:
        key = (L, W, top_z, depth)
        if key in self._recursive_fragile_top_cache:
            return self._recursive_fragile_top_cache[key]

        best = self.best_fragile_final_top_z_for_top(L, W, top_z)
        if depth <= 0 or top_z >= self.truck.effective_H or not self.remaining_standard_support_builder_exists():
            self._recursive_fragile_top_cache[key] = best
            return best

        for code in STANDARD_SUPPORT_BUILDERS:
            if self.remaining_counts.get(code, 0) <= 0:
                continue
            spec = self.catalog[code]
            for nl, nw, nh, _ in unique_orientations(spec.dims, spec.category, self.fragile_fixed):
                if nl > L or nw > W:
                    continue
                if top_z + nh > self.truck.effective_H:
                    continue
                child = self._best_recursive_fragile_top_z_from_top(nl, nw, top_z + nh, depth - 1)
                best = max(best, child)

        self._recursive_fragile_top_cache[key] = best
        return best

    def top_can_host_fragile(self, L: int, W: int, top_z: int) -> bool:
        key = (L, W, top_z)
        if key in self._top_fragile_fit_cache:
            return self._top_fragile_fit_cache[key]
        ok = self.best_fragile_final_top_z_for_top(L, W, top_z) >= 0
        self._top_fragile_fit_cache[key] = ok
        return ok

    def merged_adjacent_rect(self, x1: int, y1: int, L1: int, W1: int, x2: int, y2: int, L2: int, W2: int) -> Optional[Tuple[int, int, int, int]]:
        if x1 == x2 and L1 == L2 and (y1 + W1 == y2 or y2 + W2 == y1):
            return (x1, min(y1, y2), L1, W1 + W2)
        if y1 == y2 and W1 == W2 and (x1 + L1 == x2 or x2 + L2 == x1):
            return (min(x1, x2), y1, L1 + L2, W1)
        return None

    def best_fragile_top_from_existing_neighbors(self, x: int, y: int, L: int, W: int, top_z: int, exclude_pids: Tuple[int, ...] = ()) -> int:
        best = -1
        for other in self.surfaces:
            if other.z != top_z or other.parent_pid is None or other.parent_pid in exclude_pids:
                continue
            parent = self.by_pid.get(other.parent_pid)
            if parent is None or parent.category != "standard":
                continue
            merged = self.merged_adjacent_rect(x, y, L, W, other.x, other.y, other.L, other.W)
            if merged is None:
                continue
            _, _, mL, mW = merged
            best = max(best, self.best_fragile_final_top_z_for_top(mL, mW, top_z))
        return best

    def best_fragile_top_from_same_level_sibling_after_builder(
        self,
        surface: Surface,
        code: str,
        l: int,
        w: int,
        h: int,
    ) -> int:
        """估计“先抬当前柱、再抬同层相邻柱”后形成更高 G3 支撑面的潜力。

        这是针对“G2 只堆一层就开始放 G3”问题的关键修正：
        当前代码原本只能看到“当前柱抬高后与已经同高的邻柱拼面”，
        却看不到“相邻柱下一步也能同步抬高到同一层”这一常见情形。
        因而会在较低层过早放置 G3。这里显式地把这类两步平台构造
        （当前柱 + 同层相邻柱）纳入前瞻评估。
        """
        if self.remaining_counts.get(code, 0) < 2:
            return -1

        spec = self.catalog[code]
        top_z = surface.z + h
        best = -1

        current_support = tuple(sorted(surface.support_pids)) if surface.support_pids else (() if surface.parent_pid is None else (surface.parent_pid,))

        for other in self.surfaces:
            if other is surface:
                continue
            if other.z != surface.z:
                continue

            other_support_boxes = self.surface_support_boxes(other)
            if not other_support_boxes or not all(box.category == "standard" for box in other_support_boxes):
                continue

            other_support = tuple(sorted(other.support_pids)) if other.support_pids else (() if other.parent_pid is None else (other.parent_pid,))
            if other_support == current_support:
                # 同一父面的再摆放潜力已经由“同一支撑面再补一件”逻辑覆盖。
                continue

            if l > other.L or w > other.W:
                continue
            if other.z + h > self.truck.effective_H:
                continue
            if not self.can_support(spec, other, l, w):
                continue

            merged = self.merged_adjacent_rect(surface.x, surface.y, l, w, other.x, other.y, l, w)
            if merged is None:
                continue
            _, _, mL, mW = merged
            best = max(best, self.best_fragile_final_top_z_for_top(mL, mW, top_z))

        return best

    def best_future_fragile_top_after_builder(self, surface: Surface, code: str, l: int, w: int, h: int) -> int:
        top_z = surface.z + h
        best = self._best_recursive_fragile_top_z_from_top(l, w, top_z, FRAGILE_PLATFORM_LOOKAHEAD_DEPTH - 1)

        # 情形 A：当前柱抬高后，已经存在同高邻柱，可立即形成更高拼面。
        best = max(best, self.best_fragile_top_from_existing_neighbors(surface.x, surface.y, l, w, top_z))

        # 情形 B：当前支撑面本身还足够宽/长，再补一件同类标准件后形成更高拼面。
        if self.remaining_counts.get(code, 0) >= 2:
            if 2 * l <= surface.L:
                best = max(best, self.best_fragile_final_top_z_for_top(2 * l, w, top_z))
            if 2 * w <= surface.W:
                best = max(best, self.best_fragile_final_top_z_for_top(l, 2 * w, top_z))

        # 情形 C：当前柱先抬高一步，随后把“同层相邻柱”也抬到同一高度，
        # 再在更高层形成 G3 支撑面。这个分支正是用于抑制“G2 只放一层就开始放 G3”。
        best = max(best, self.best_fragile_top_from_same_level_sibling_after_builder(surface, code, l, w, h))
        return best

    def best_future_fragile_top_z_from_surface(self, surface: Surface) -> int:
        key = self.surface_cache_key(surface)
        if key in self._future_fragile_platform_cache:
            return self._future_fragile_platform_cache[key]

        if not self.remaining_fragile_exists() or not self.remaining_standard_support_builder_exists():
            self._future_fragile_platform_cache[key] = -1
            return -1

        best_final_top = -1
        for code in STANDARD_SUPPORT_BUILDERS:
            if self.remaining_counts.get(code, 0) <= 0:
                continue
            spec = self.catalog[code]
            if self.total_weight + spec.weight > self.truck.max_weight + 1e-12:
                continue

            for l, w, h, _ in unique_orientations(spec.dims, spec.category, self.fragile_fixed):
                if l > surface.L or w > surface.W:
                    continue
                if surface.z + h > self.truck.effective_H:
                    continue
                if not self.can_support(spec, surface, l, w):
                    continue

                final_top = self._best_recursive_fragile_top_z_from_top(
                    l, w, surface.z + h, FRAGILE_PLATFORM_LOOKAHEAD_DEPTH - 1
                )
                if final_top >= 0:
                    best_final_top = max(best_final_top, final_top)

        self._future_fragile_platform_cache[key] = best_final_top
        return best_final_top

    def generate_fragile_composite_surfaces(self, candidate_surfaces: List[Surface]) -> List[Surface]:
        standard_surfaces = [
            s for s in candidate_surfaces
            if s.parent_pid is not None and self.surface_is_standard_support(s)
        ][:COMPOSITE_SURFACE_PAIR_LIMIT]

        best_by_key: Dict[Tuple[int, int, int, int, int, Tuple[int, ...]], Surface] = {}
        for i, s1 in enumerate(standard_surfaces):
            p1 = self.by_pid.get(s1.parent_pid)
            if p1 is None or p1.category != "standard":
                continue
            for s2 in standard_surfaces[i + 1 :]:
                p2 = self.by_pid.get(s2.parent_pid)
                if p2 is None or p2.category != "standard" or p2.pid == p1.pid:
                    continue
                if s1.z != s2.z:
                    continue

                merged: Optional[Surface] = None
                # 纵向拼接：同 x / L，相邻 y
                if s1.x == s2.x and s1.L == s2.L and (s1.y + s1.W == s2.y or s2.y + s2.W == s1.y):
                    merged = Surface(
                        x=s1.x,
                        y=min(s1.y, s2.y),
                        z=s1.z,
                        L=s1.L,
                        W=s1.W + s2.W,
                        parent_pid=None,
                        support_pids=tuple(sorted((p1.pid, p2.pid))),
                        source_surfaces=(s1, s2),
                    )
                # 横向拼接：同 y / W，相邻 x
                elif s1.y == s2.y and s1.W == s2.W and (s1.x + s1.L == s2.x or s2.x + s2.L == s1.x):
                    merged = Surface(
                        x=min(s1.x, s2.x),
                        y=s1.y,
                        z=s1.z,
                        L=s1.L + s2.L,
                        W=s1.W,
                        parent_pid=None,
                        support_pids=tuple(sorted((p1.pid, p2.pid))),
                        source_surfaces=(s1, s2),
                    )
                if merged is None:
                    continue
                if self.best_fragile_final_top_z_for_top(merged.L, merged.W, merged.z) < 0:
                    continue
                ckey = (merged.x, merged.y, merged.z, merged.L, merged.W, merged.support_pids)
                best_by_key[ckey] = merged
        return list(best_by_key.values())

    def best_supported_fragile_choice(self, candidate_surfaces: List[Surface]) -> Tuple[Optional[Tuple[ItemType, Surface, int, int, int, str]], int]:
        if self.remaining_counts.get(FRAGILE_CODE, 0) <= 0:
            return None, -1

        spec = self.catalog[FRAGILE_CODE]
        if self.total_weight + spec.weight > self.truck.max_weight + 1e-12:
            return None, -1

        best_choice = None
        best_key = None
        best_final_top = -1

        for surface in candidate_surfaces:
            support_boxes = self.surface_support_boxes(surface)
            if support_boxes and not all(box.category == "standard" for box in support_boxes):
                continue
            # 顶层优先阶段只考虑“可抬高后的高位平台”；地板情况交给通用搜索处理。
            if not support_boxes:
                continue

            for l, w, h, orient_name in unique_orientations(spec.dims, spec.category, self.fragile_fixed):
                if l > surface.L or w > surface.W:
                    continue
                if surface.z + h > self.truck.effective_H:
                    continue
                if not self.can_support(spec, surface, l, w):
                    continue

                final_top = surface.z + h
                composite_bonus = 1 if len(surface.support_pids) >= 2 else 0
                key = (final_top, composite_bonus) + self.candidate_key(spec, surface, l, w, h)
                if best_key is None or key > best_key:
                    best_key = key
                    best_choice = (spec, surface, l, w, h, orient_name)
                    best_final_top = final_top

        return best_choice, best_final_top

    def best_platform_builder_choice(self, candidate_surfaces: List[Surface]) -> Tuple[Optional[Tuple[ItemType, Surface, int, int, int, str]], int]:
        if not self.remaining_fragile_exists() or not self.remaining_standard_support_builder_exists():
            return None, -1

        best_choice = None
        best_key = None
        best_future_final_top = -1

        for code in STANDARD_SUPPORT_BUILDERS:
            if self.remaining_counts.get(code, 0) <= 0:
                continue
            spec = self.catalog[code]
            if self.total_weight + spec.weight > self.truck.max_weight + 1e-12:
                continue

            for surface in candidate_surfaces:
                for l, w, h, orient_name in unique_orientations(spec.dims, spec.category, self.fragile_fixed):
                    if l > surface.L or w > surface.W:
                        continue
                    if surface.z + h > self.truck.effective_H:
                        continue
                    if not self.can_support(spec, surface, l, w):
                        continue

                    future_final_top = self.best_future_fragile_top_after_builder(surface, code, l, w, h)
                    if future_final_top < 0:
                        continue

                    key = (future_final_top,) + self.candidate_key(spec, surface, l, w, h)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_choice = (spec, surface, l, w, h, orient_name)
                        best_future_final_top = future_final_top

        return best_choice, best_future_final_top

    def select_toplayer_fragile_priority_choice(self, candidate_surfaces: List[Surface]) -> Optional[Tuple[ItemType, Surface, int, int, int, str]]:
        if not self.remaining_fragile_exists():
            return None

        fragile_surfaces = list(candidate_surfaces) + self.generate_fragile_composite_surfaces(candidate_surfaces)
        fragile_choice, fragile_final_top = self.best_supported_fragile_choice(fragile_surfaces)
        builder_choice, builder_future_final_top = self.best_platform_builder_choice(candidate_surfaces)

        if fragile_choice is None and builder_choice is None:
            return None
        if fragile_choice is None:
            return builder_choice
        if builder_choice is None:
            return fragile_choice

        # 只要还能明显抬高 G3 的最终顶高，就继续造平台；
        # 否则直接把 G3 放到当前更高的位置。
        if builder_future_final_top > fragile_final_top + 1:
            return builder_choice
        return fragile_choice

    # 表面维护
    def split_surface(self, surface: Surface, l: int, w: int) -> List[Surface]:
        # 切法 A
        a_surfaces: List[Surface] = []
        if surface.L - l > 0:
            a_surfaces.append(Surface(surface.x + l, surface.y, surface.z, surface.L - l, surface.W, surface.parent_pid))
        if surface.W - w > 0:
            a_surfaces.append(Surface(surface.x, surface.y + w, surface.z, l, surface.W - w, surface.parent_pid))
        score_a = sum(s.area for s in a_surfaces) - abs((surface.L - l) - l) - abs((surface.W - w) - w)

        # 切法 B
        b_surfaces: List[Surface] = []
        if surface.W - w > 0:
            b_surfaces.append(Surface(surface.x, surface.y + w, surface.z, surface.L, surface.W - w, surface.parent_pid))
        if surface.L - l > 0:
            b_surfaces.append(Surface(surface.x + l, surface.y, surface.z, surface.L - l, w, surface.parent_pid))
        score_b = sum(s.area for s in b_surfaces) - abs((surface.L - l) - l) - abs((surface.W - w) - w)

        return a_surfaces if score_a >= score_b else b_surfaces

    def subtract_rect_from_surface(self, surface: Surface, rx: int, ry: int, rl: int, rw: int) -> List[Surface]:
        ix0 = max(surface.x, rx)
        iy0 = max(surface.y, ry)
        ix1 = min(surface.x + surface.L, rx + rl)
        iy1 = min(surface.y + surface.W, ry + rw)
        if ix0 >= ix1 or iy0 >= iy1:
            return [surface]

        out: List[Surface] = []
        # left / right strips
        if ix0 > surface.x:
            out.append(Surface(surface.x, surface.y, surface.z, ix0 - surface.x, surface.W, surface.parent_pid))
        if ix1 < surface.x + surface.L:
            out.append(Surface(ix1, surface.y, surface.z, surface.x + surface.L - ix1, surface.W, surface.parent_pid))
        # bottom / top strips inside the overlapped x-span
        mid_L = ix1 - ix0
        if iy0 > surface.y:
            out.append(Surface(ix0, surface.y, surface.z, mid_L, iy0 - surface.y, surface.parent_pid))
        if iy1 < surface.y + surface.W:
            out.append(Surface(ix0, iy1, surface.z, mid_L, surface.y + surface.W - iy1, surface.parent_pid))
        return [s for s in out if s.L > 0 and s.W > 0]

    def _merge_surfaces_once(self, surfaces: List[Surface]) -> List[Surface]:
        # 先横向合并
        grouped: Dict[Tuple[int, Optional[int], int, int], List[Surface]] = {}
        for s in surfaces:
            grouped.setdefault((s.z, s.parent_pid, s.y, s.W), []).append(s)

        merged_h: List[Surface] = []
        for key, group in grouped.items():
            group = sorted(group, key=lambda s: (s.x, s.L))
            cur = group[0]
            for nxt in group[1:]:
                if cur.x + cur.L == nxt.x:
                    cur = Surface(cur.x, cur.y, cur.z, cur.L + nxt.L, cur.W, cur.parent_pid)
                else:
                    merged_h.append(cur)
                    cur = nxt
            merged_h.append(cur)

        # 再纵向合并
        grouped2: Dict[Tuple[int, Optional[int], int, int], List[Surface]] = {}
        for s in merged_h:
            grouped2.setdefault((s.z, s.parent_pid, s.x, s.L), []).append(s)

        merged_v: List[Surface] = []
        for key, group in grouped2.items():
            group = sorted(group, key=lambda s: (s.y, s.W))
            cur = group[0]
            for nxt in group[1:]:
                if cur.y + cur.W == nxt.y:
                    cur = Surface(cur.x, cur.y, cur.z, cur.L, cur.W + nxt.W, cur.parent_pid)
                else:
                    merged_v.append(cur)
                    cur = nxt
            merged_v.append(cur)

        return merged_v

    def prune_surfaces(self) -> None:
        surfaces = [
            s for s in self.surfaces
            if s.L > 0 and s.W > 0 and s.z <= self.truck.effective_H
        ]

        # 去除被包含表面
        surfaces.sort(key=lambda s: (-s.area, s.z, s.x, s.y))
        kept: List[Surface] = []
        for s in surfaces:
            covered = False
            for k in kept:
                if (
                    s.parent_pid == k.parent_pid
                    and s.z == k.z
                    and s.x >= k.x
                    and s.y >= k.y
                    and s.x + s.L <= k.x + k.L
                    and s.y + s.W <= k.y + k.W
                ):
                    covered = True
                    break
            if not covered:
                kept.append(s)

        # 进行两轮邻接面合并
        for _ in range(2):
            kept = self._merge_surfaces_once(kept)

        kept.sort(key=lambda s: (s.z, s.x, s.y, -s.area))
        self.surfaces = kept[:MAX_SURFACES]

    # 目标与评分
    def utilization(self) -> Dict[str, float]:
        return {
            "空间利用率_raw": self.total_volume / self.truck.raw_volume,
            "空间利用率_eff": self.total_volume / self.truck.eff_volume,
            "载重利用率": self.total_weight / self.truck.max_weight,
            "已装货物体积_m3": self.total_volume / 1e6,
            "已装货物重量_kg": self.total_weight,
        }

    def truck_score_tuple(self) -> Tuple[float, float, float, float, float]:
        util = self.utilization()
        v = util["空间利用率_eff"]
        w = util["载重利用率"]
        return (
            min(v, w),
            (v + w) / 2.0,
            -abs(v - w),
            v * w,
            len(self.placed),
        )

    def candidate_key(self, spec: ItemType, surface: Surface, l: int, w: int, h: int) -> Tuple[float, ...]:
        new_v = (self.total_volume + l * w * h) / self.truck.eff_volume
        new_w = (self.total_weight + spec.weight) / self.truck.max_weight
        fill = (l * w) / (surface.L * surface.W)
        cat_bonus = self.policy.category_bonus.get(spec.category, 0.0)
        gap = abs(new_v - new_w)

        support_boxes = self.surface_support_boxes(surface)
        has_standard_support = bool(support_boxes) and all(box.category == "standard" for box in support_boxes)
        is_composite_support = len(surface.support_pids) >= 2

        fragile_support_bonus = 0.0
        standard_platform_bonus = 0.0
        top_surface_penalty = 0.0
        top_layer_bonus = 0.0
        future_stack_penalty = 0.0
        shape_pref = -float(h)

        if spec.category == "fragile":
            shape_pref = float(h)
            placed_top_z = surface.z + h
            top_ratio = placed_top_z / self.truck.effective_H
            top_layer_bonus = (
                FRAGILE_TOP_LINEAR_BONUS * top_ratio
                + FRAGILE_TOP_QUADRATIC_BONUS * (top_ratio ** 2)
            )
            if placed_top_z >= self.truck.effective_H - FRAGILE_NEAR_TOP_MARGIN_CM:
                top_layer_bonus += FRAGILE_NEAR_TOP_BONUS

            if has_standard_support:
                fragile_support_bonus = 1.2 + (0.35 if is_composite_support else 0.0)
                if not is_composite_support:
                    future_top_z = self.best_future_fragile_top_z_from_surface(surface)
                    if future_top_z > placed_top_z + 1:
                        gap_ratio = (future_top_z - placed_top_z) / self.truck.effective_H
                        future_stack_penalty = -(2.8 + 3.5 * gap_ratio)
            elif self.remaining_standard_support_builder_exists():
                future_stack_penalty = -1.5

        if spec.category == "standard" and self.remaining_fragile_exists():
            top_z = surface.z + h
            future_fragile_top = self.best_future_fragile_top_after_builder(surface, spec.code, l, w, h)
            if future_fragile_top >= 0:
                standard_platform_bonus = 1.0 + 2.8 * (future_fragile_top / self.truck.effective_H)
                if future_fragile_top > top_z + 1:
                    standard_platform_bonus += 0.8
            elif surface.z == 0:
                standard_platform_bonus = 0.1

        if (
            support_boxes
            and len(support_boxes) == 1
            and support_boxes[0].category == "standard"
            and spec.category == "directional"
            and self.remaining_fragile_exists()
        ):
            top_surface_penalty = -0.7

        if self.policy.mode == "balanced":
            main = (min(new_v, new_w), (new_v + new_w) / 2.0, -gap, new_v * new_w)
        elif self.policy.mode == "volume":
            main = (new_v, min(new_v, new_w), (new_v + new_w) / 2.0, -gap)
        elif self.policy.mode == "weight":
            main = (new_w, min(new_v, new_w), (new_v + new_w) / 2.0, -gap)
        elif self.policy.mode == "product":
            main = (new_v * new_w, min(new_v, new_w), (new_v + new_w) / 2.0, -gap)
        else:
            raise ValueError(f"未知策略模式: {self.policy.mode}")

        return main + (
            fragile_support_bonus,
            standard_platform_bonus,
            top_surface_penalty,
            top_layer_bonus,
            future_stack_penalty,
            shape_pref,
            fill,
            cat_bonus,
            -surface.z,
            -surface.x,
            -surface.y,
        )

    # 放置操作
    def place(self, spec: ItemType, surface: Surface, l: int, w: int, h: int, orient_name: str) -> None:
        pid = self.next_pid
        self.next_pid += 1

        support_pids = tuple(surface.support_pids) if surface.support_pids else (() if surface.parent_pid is None else (surface.parent_pid,))
        support_parent = surface.parent_pid if not surface.support_pids else None
        placed = PlacedBox(
            pid=pid,
            type_code=spec.code,
            category=spec.category,
            weight=spec.weight,
            x=surface.x,
            y=surface.y,
            z=surface.z,
            l=l,
            w=w,
            h=h,
            orientation_name=orient_name,
            parent_pid=support_parent,
            support_pids=support_pids,
        )

        self.placed.append(placed)
        self.by_pid[pid] = placed
        self.total_weight += spec.weight
        self.total_volume += placed.volume
        self.remaining_counts[spec.code] -= 1
        self.loaded_counts[spec.code] += 1

        load_map = self.compute_support_loads(spec.weight, surface, placed.x, placed.y, l, w)
        if load_map is None:
            raise RuntimeError("组合支撑面重量分配失败。")
        self.apply_load_update(load_map)

        if surface.source_surfaces:
            for src in surface.source_surfaces:
                if src in self.surfaces:
                    self.surfaces.remove(src)
                    self.surfaces.extend(self.subtract_rect_from_surface(src, placed.x, placed.y, l, w))
        else:
            self.surfaces.remove(surface)
            self.surfaces.extend(self.split_surface(surface, l, w))

        if spec.category != "fragile":
            top_z = surface.z + h
            if top_z <= self.truck.effective_H:
                self.surfaces.append(Surface(surface.x, surface.y, top_z, l, w, pid))

        self.prune_surfaces()
        self.reset_state_caches()

    # 单车装箱主过程
    def pack(self) -> "SurfacePacker":
        while self.surfaces and not is_zero_counts(self.remaining_counts):
            self.surfaces.sort(key=lambda s: (s.z, s.x, s.y, -s.area))
            base_candidate_surfaces = self.surfaces[:CANDIDATE_SURFACE_LIMIT]

            priority_choice = self.select_toplayer_fragile_priority_choice(base_candidate_surfaces)
            if priority_choice is not None:
                spec, surface, l, w, h, orient_name = priority_choice
                self.place(spec, surface, l, w, h, orient_name)
                continue

            best_choice = None
            best_key = None

            for code in ITEM_ORDER:
                if self.remaining_counts.get(code, 0) <= 0:
                    continue
                spec = self.catalog[code]

                if self.total_weight + spec.weight > self.truck.max_weight + 1e-12:
                    continue

                for surface in base_candidate_surfaces:
                    for l, w, h, orient_name in unique_orientations(spec.dims, spec.category, self.fragile_fixed):
                        if l > surface.L or w > surface.W:
                            continue
                        if surface.z + h > self.truck.effective_H:
                            continue
                        if not self.can_support(spec, surface, l, w):
                            continue

                        key = self.candidate_key(spec, surface, l, w, h)
                        if best_key is None or key > best_key:
                            best_key = key
                            best_choice = (spec, surface, l, w, h, orient_name)

            if best_choice is None:
                break

            spec, surface, l, w, h, orient_name = best_choice
            self.place(spec, surface, l, w, h, orient_name)

        return self


# 六、策略集
SINGLE_TRUCK_POLICIES: Tuple[PackingPolicy, ...] = (
    PackingPolicy(
        "balanced_platform",
        "balanced",
        {"standard": 2.0, "directional": 1.0, "fragile": 0.0},
    ),
    PackingPolicy(
        "balanced_directional",
        "balanced",
        {"directional": 2.0, "standard": 1.0, "fragile": 0.0},
    ),
    PackingPolicy(
        "product_mix",
        "product",
        {"standard": 1.5, "directional": 1.5, "fragile": 0.2},
    ),
    PackingPolicy(
        "volume_first",
        "volume",
        {"standard": 1.5, "directional": 1.0, "fragile": 0.1},
    ),
    PackingPolicy(
        "weight_first",
        "weight",
        {"standard": 1.0, "directional": 1.0, "fragile": 0.5},
    ),
)

MIN_TRUCK_POLICIES: Tuple[PackingPolicy, ...] = (
    PackingPolicy(
        "cover_volume",
        "volume",
        {"standard": 1.8, "directional": 1.0, "fragile": 0.2},
    ),
    PackingPolicy(
        "cover_balanced",
        "balanced",
        {"standard": 2.0, "directional": 1.0, "fragile": 0.0},
    ),
    PackingPolicy(
        "cover_product",
        "product",
        {"standard": 1.3, "directional": 1.3, "fragile": 0.0},
    ),
    PackingPolicy(
        "fragile_rescue",
        "volume",
        {"fragile": 2.0, "standard": 1.0, "directional": 0.5},
    ),
    PackingPolicy(
        "heavy_mix",
        "weight",
        {"directional": 1.5, "standard": 1.0, "fragile": 0.2},
    ),
)


# 七、候选模式生成、单车择优、最少车辆搜索
def compare_balance_packers(packer: SurfacePacker) -> Tuple[float, float, float, float, float]:
    util = packer.utilization()
    v = util["空间利用率_eff"]
    w = util["载重利用率"]
    return (min(v, w), (v + w) / 2.0, -abs(v - w), v * w, len(packer.placed))


def compare_cover_packers(packer: SurfacePacker) -> Tuple[float, float, float, float, int]:
    util = packer.utilization()
    v = util["空间利用率_eff"]
    w = util["载重利用率"]
    return (v, min(v, w), w, v * w, len(packer.placed))


def pack_single_truck_bestof(
    counts: Dict[str, int],
    catalog: Dict[str, ItemType],
    truck: TruckType,
    fragile_fixed: bool = False,
) -> SurfacePacker:
    best = None
    best_key = None

    for policy in SINGLE_TRUCK_POLICIES:
        packer = SurfacePacker(truck, catalog, counts, policy, fragile_fixed=fragile_fixed).pack()
        key = compare_balance_packers(packer)
        if best_key is None or key > best_key:
            best_key = key
            best = packer

    return best


def generate_candidate_packers(
    counts: Dict[str, int],
    catalog: Dict[str, ItemType],
    truck: TruckType,
    fragile_fixed: bool = False,
    top_k: int = 4,
) -> List[SurfacePacker]:
    candidates = []
    best_by_pattern: Dict[Tuple[int, ...], Tuple[Tuple[float, ...], SurfacePacker]] = {}

    for policy in MIN_TRUCK_POLICIES:
        packer = SurfacePacker(truck, catalog, counts, policy, fragile_fixed=fragile_fixed).pack()
        if len(packer.placed) == 0:
            continue
        pattern = counts_tuple(packer.loaded_counts)
        score = compare_cover_packers(packer)
        old = best_by_pattern.get(pattern)
        if old is None or score > old[0]:
            best_by_pattern[pattern] = (score, packer)

    ranked = sorted(best_by_pattern.values(), key=lambda x: x[0], reverse=True)
    return [packer for _, packer in ranked[:top_k]]


def greedy_pack_all(
    counts: Dict[str, int],
    catalog: Dict[str, ItemType],
    truck: TruckType,
    fragile_fixed: bool = False,
) -> List[SurfacePacker]:
    remaining = {code: counts.get(code, 0) for code in ITEM_ORDER}
    solution: List[SurfacePacker] = []

    while not is_zero_counts(remaining):
        candidates = generate_candidate_packers(remaining, catalog, truck, fragile_fixed=fragile_fixed, top_k=1)
        if not candidates:
            raise RuntimeError("当前启发式未能再装入任何货物，请检查参数设置。")
        chosen = candidates[0]
        solution.append(chosen)
        remaining = subtract_counts(remaining, chosen.loaded_counts)

    return solution


def can_finish_with_n_trucks(
    counts: Dict[str, int],
    catalog: Dict[str, ItemType],
    truck: TruckType,
    n: int,
    fragile_fixed: bool = False,
    top_k: int = 4,
    memo: Optional[Dict[Tuple[int, Tuple[int, ...]], Tuple[bool, Optional[List[SurfacePacker]]]]] = None,
    candidate_cache: Optional[Dict[Tuple[Tuple[int, ...], str], List[SurfacePacker]]] = None,
) -> Tuple[bool, Optional[List[SurfacePacker]]]:
    if memo is None:
        memo = {}
    if candidate_cache is None:
        candidate_cache = {}

    state = counts_tuple(counts)
    key = (n, state)
    if key in memo:
        return memo[key]

    if all(v == 0 for v in state):
        memo[key] = (True, [])
        return memo[key]

    if n == 0:
        memo[key] = (False, None)
        return memo[key]

    if volume_weight_lower_bound(counts, catalog, truck) > n:
        memo[key] = (False, None)
        return memo[key]

    ckey = (state, truck.name)
    if ckey not in candidate_cache:
        candidate_cache[ckey] = generate_candidate_packers(
            counts,
            catalog,
            truck,
            fragile_fixed=fragile_fixed,
            top_k=top_k,
        )
    candidates = candidate_cache[ckey]

    for packer in candidates:
        new_counts = subtract_counts(counts, packer.loaded_counts)
        ok, tail = can_finish_with_n_trucks(
            new_counts,
            catalog,
            truck,
            n - 1,
            fragile_fixed=fragile_fixed,
            top_k=top_k,
            memo=memo,
            candidate_cache=candidate_cache,
        )
        if ok:
            memo[key] = (True, [packer] + (tail or []))
            return memo[key]

    memo[key] = (False, None)
    return memo[key]


def solve_min_trucks_single_type(
    counts: Dict[str, int],
    catalog: Dict[str, ItemType],
    truck: TruckType,
    fragile_fixed: bool = False,
    max_search_gap: int = 3,
) -> List[SurfacePacker]:
    """固定一种车型时，求“尽量少的车辆数”。

    说明：
    - 先用贪心得到一个可行上界；
    - 只有当“上界 - 下界”不大时，才启动有限深度搜索压缩车辆数；
    - 对于本题放大10倍后的数据，若 gap 过大，直接返回贪心解更稳健，
      这样可以保证程序在比赛环境中运行时间可控。
    """
    lb = volume_weight_lower_bound(counts, catalog, truck)
    greedy_solution = greedy_pack_all(counts, catalog, truck, fragile_fixed=fragile_fixed)
    ub = len(greedy_solution)

    if lb == ub:
        return greedy_solution

    if ub - lb > max_search_gap:
        return greedy_solution

    memo: Dict[Tuple[int, Tuple[int, ...]], Tuple[bool, Optional[List[SurfacePacker]]]] = {}
    candidate_cache: Dict[Tuple[Tuple[int, ...], str], List[SurfacePacker]] = {}

    for n in range(lb, ub + 1):
        ok, sol = can_finish_with_n_trucks(
            counts,
            catalog,
            truck,
            n,
            fragile_fixed=fragile_fixed,
            top_k=4,
            memo=memo,
            candidate_cache=candidate_cache,
        )
        if ok and sol is not None:
            return sol

    return greedy_solution


# 八、导出与结果整理
def uid_width_for_catalog(catalog: Dict[str, ItemType]) -> int:
    return max(4, max(len(str(catalog[code].quantity)) for code in ITEM_ORDER))


def solution_rows_from_packers(packers: List[SurfacePacker], catalog: Dict[str, ItemType]) -> List[Dict[str, object]]:
    width = uid_width_for_catalog(catalog)
    next_index = {code: 1 for code in ITEM_ORDER}
    rows: List[Dict[str, object]] = []
    category_name_map = {
        "standard": "标准件",
        "fragile": "易碎件",
        "directional": "定向件",
    }

    for truck_idx, packer in enumerate(packers, start=1):
        pid_to_uid: Dict[int, str] = {}
        for p in packer.placed:
            uid = f"{p.type_code}_{next_index[p.type_code]:0{width}d}"
            next_index[p.type_code] += 1
            pid_to_uid[p.pid] = uid

        for p in packer.placed:
            rows.append(
                {
                    "车辆编号": truck_idx,
                    "车型": packer.truck.name,
                    "货物唯一编号": pid_to_uid[p.pid],
                    "货物类型": p.type_code,
                    "货物类别": category_name_map.get(p.category, p.category),
                    "x坐标_右后到前_cm": p.x,
                    "y坐标_右到左_cm": p.y,
                    "z坐标_下到上_cm": p.z,
                    "放置长度_x方向_cm": p.l,
                    "放置宽度_y方向_cm": p.w,
                    "放置高度_z方向_cm": p.h,
                    "摆放姿态": p.orientation_name,
                    "支撑对象编号": (
                        "组合支撑:" + "|".join(pid_to_uid[pid] for pid in p.support_pids)
                        if len(p.support_pids) >= 2
                        else (pid_to_uid.get(p.parent_pid, "地板") if p.parent_pid is not None else "地板")
                    ),
                    "货物重量_kg": p.weight,
                    "货物体积_cm3": p.volume,
                }
            )
    return rows


def summary_rows(title: str, packers: List[SurfacePacker]) -> List[Dict[str, object]]:
    rows = []
    for idx, packer in enumerate(packers, start=1):
        util = packer.utilization()
        row = {
            "方案": title,
            "车辆序号": idx,
            "车型": packer.truck.name,
            "装载件数": len(packer.placed),
            "空间利用率_raw": util["空间利用率_raw"],
            "空间利用率_eff": util["空间利用率_eff"],
            "载重利用率": util["载重利用率"],
            "已装货物体积_m3": util["已装货物体积_m3"],
            "已装货物重量_kg": util["已装货物重量_kg"],
        }
        for code in ITEM_ORDER:
            row[f"{code}_数量"] = packer.loaded_counts[code]
        rows.append(row)
    return rows


def print_solution_summary(title: str, packers: List[SurfacePacker]) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"总车辆数: {len(packers)}")
    total_vol = sum(p.total_volume for p in packers) / 1e6
    total_wt = sum(p.total_weight for p in packers)
    print(f"总装载体积: {total_vol:.3f} m^3")
    print(f"总装载重量: {total_wt:.1f} kg")
    print()

    for i, p in enumerate(packers, start=1):
        util = p.utilization()
        print(f"车辆 {i} | {p.truck.name} | 件数 {len(p.placed)}")
        print(f"  空间利用率(raw): {util['空间利用率_raw']:.4f}")
        print(f"  空间利用率(eff): {util['空间利用率_eff']:.4f}")
        print(f"  载重利用率     : {util['载重利用率']:.4f}")
        print(f"  已装体积       : {util['已装货物体积_m3']:.3f} m^3")
        print(f"  已装重量       : {util['已装货物重量_kg']:.1f} kg")
        print(
            "  货类计数       : "
            + ", ".join(f"{code}={p.loaded_counts[code]}" for code in ITEM_ORDER)
        )
        print()


# 九、问题2：多车型组合配送（独立于问题1的精确求解）
PATTERN_POLICIES_Q2: Tuple[PackingPolicy, ...] = (
    PackingPolicy("cover_volume", "volume", {"standard": 1.8, "directional": 1.0, "fragile": 0.2}),
    PackingPolicy("cover_balanced", "balanced", {"standard": 2.0, "directional": 1.0, "fragile": 0.0}),
    PackingPolicy("directional_bias", "volume", {"directional": 2.5, "standard": 0.8, "fragile": 0.3}),
    PackingPolicy("heavy_mix", "weight", {"directional": 1.5, "standard": 1.0, "fragile": 0.2}),
)

# 比原版更丰富的截断网格，有利于生成更多高质量单车模式，
# 再由 MILP 做“最少车/最低成本”精确覆盖。
CAP_GRID_Q2: Tuple[Optional[int], ...] = (None, 30, 80, 150, 300)

_PATTERN_CACHE_Q2: Dict[Tuple[str, Tuple[int, ...], bool], List[SurfacePacker]] = {}

# 运行时间控制：对 Q2 的模式库做“主评分 + 性价比”双通道裁剪，
# 同时缩短若干辅助 MILP 的时限，避免最少成本分支因为时间上限返回较差可行解。
Q2_TRUCK_LIBRARY_KEEP_MAIN = 90
Q2_TRUCK_LIBRARY_KEEP_COST = 60
Q2_JOINT_LIBRARY_KEEP_MAIN = 150
Q2_JOINT_LIBRARY_KEEP_COST = 100
Q2_TIME_LIMIT_MIN_TRUCKS_STAGE1 = 90.0
Q2_TIME_LIMIT_MIN_TRUCKS_STAGE2 = 45.0
Q2_TIME_LIMIT_COST_STAGE1 = 35.0
Q2_TIME_LIMIT_COST_STAGE2 = 35.0
Q2_TIME_LIMIT_BALANCE = 30.0
Q2_TIME_LIMIT_THRESHOLD = 10.0
Q2_TIME_LIMIT_CHEAPER_CHECK = 8.0
Q2_TIME_LIMIT_CHEAPER_SEARCH = 15.0


def _pattern_subsets_q2() -> Tuple[Tuple[str, ...], ...]:
    subsets: List[Tuple[str, ...]] = []
    for r in (1, 2, 3):
        subsets.extend(combinations(ITEM_ORDER, r))
    subsets.extend(
        [
            ("G1", "G2", "G4", "G5"),
            ("G2", "G3", "G4", "G5"),
            ("G1", "G2", "G3", "G5"),
            ("G1", "G2", "G3", "G4"),
            ("G1", "G3", "G4"),
            ITEM_ORDER,
        ]
    )
    return tuple(subsets)


PATTERN_SUBSETS_Q2: Tuple[Tuple[str, ...], ...] = _pattern_subsets_q2()


def _masked_counts_q2(counts: Dict[str, int], subset: Sequence[str], cap: Optional[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for code in ITEM_ORDER:
        if code not in subset:
            out[code] = 0
        elif cap is None:
            out[code] = counts[code]
        else:
            out[code] = min(counts[code], cap)
    return out


def _pattern_key_q2(packer: SurfacePacker) -> Tuple[str, Tuple[int, ...]]:
    return packer.truck.name, tuple(packer.loaded_counts[code] for code in ITEM_ORDER)


@lru_cache(maxsize=None)
def _pattern_min_fill_from_key_q2(
    loaded_g1: int,
    loaded_g2: int,
    loaded_g3: int,
    loaded_g4: int,
    loaded_g5: int,
    truck_name: str,
) -> float:
    trucks, catalog = build_problem_data()
    truck = next(t for t in trucks if t.name == truck_name)
    total_vol = 0
    total_wt = 0.0
    for code, qty in zip(ITEM_ORDER, (loaded_g1, loaded_g2, loaded_g3, loaded_g4, loaded_g5)):
        spec = catalog[code]
        total_vol += qty * spec.volume
        total_wt += qty * spec.weight
    v = total_vol / truck.eff_volume
    w = total_wt / truck.max_weight
    return min(v, w)



def pattern_min_fill_q2(packer: SurfacePacker) -> float:
    key = tuple(packer.loaded_counts[code] for code in ITEM_ORDER)
    return _pattern_min_fill_from_key_q2(*key, packer.truck.name)



def pattern_score_tuple_q2(packer: SurfacePacker) -> Tuple[float, float, float, float, int, int, float]:
    util = packer.utilization()
    v = util["空间利用率_eff"]
    w = util["载重利用率"]
    mix = sum(1 for code in ITEM_ORDER if packer.loaded_counts[code] > 0)
    return (
        min(v, w),
        (v + w) / 2.0,
        v * w,
        v,
        mix,
        len(packer.placed),
        -packer.truck.trip_cost,
    )



def pattern_secondary_score_q2(packer: SurfacePacker) -> float:
    util = packer.utilization()
    v = util["空间利用率_eff"]
    w = util["载重利用率"]
    mix = sum(1 for code in ITEM_ORDER if packer.loaded_counts[code] > 0)
    return 1000.0 * min(v, w) + 250.0 * (v * w) + 20.0 * mix + 0.1 * len(packer.placed)



def sort_solution_by_fullness_q2(packers: List[SurfacePacker]) -> List[SurfacePacker]:
    return sorted(packers, key=pattern_score_tuple_q2, reverse=True)



def pattern_cost_efficiency_q2(packer: SurfacePacker) -> Tuple[float, float, float, float, int, float]:
    util = packer.utilization()
    v = util["空间利用率_eff"]
    w = util["载重利用率"]
    trip_cost = max(1.0, float(packer.truck.trip_cost))
    return (
        packer.total_volume / trip_cost,
        min(v, w),
        packer.total_weight / trip_cost,
        v * w,
        len(packer.placed),
        -packer.truck.trip_cost,
    )



def _clip_pattern_library_q2(
    packers: List[SurfacePacker],
    keep_main: int,
    keep_cost: int,
    seed_packers: Optional[Sequence[SurfacePacker]] = None,
) -> List[SurfacePacker]:
    seed_packers = tuple(seed_packers or ())
    unique: Dict[Tuple[str, Tuple[int, ...]], SurfacePacker] = {}
    for packer in list(seed_packers) + list(packers):
        key = _pattern_key_q2(packer)
        old = unique.get(key)
        if old is None or pattern_score_tuple_q2(packer) > pattern_score_tuple_q2(old):
            unique[key] = packer

    ranked_main = sorted(unique.values(), key=pattern_score_tuple_q2, reverse=True)
    ranked_cost = sorted(unique.values(), key=pattern_cost_efficiency_q2, reverse=True)

    out: List[SurfacePacker] = []
    seen: set[Tuple[str, Tuple[int, ...]]] = set()

    def add_from(seq: Sequence[SurfacePacker], limit: Optional[int] = None) -> None:
        taken = 0
        for packer in seq:
            key = _pattern_key_q2(packer)
            if key in seen:
                continue
            seen.add(key)
            out.append(packer)
            taken += 1
            if limit is not None and taken >= limit:
                break

    add_from(seed_packers)
    add_from(ranked_main, keep_main)
    add_from(ranked_cost, keep_cost)
    return sorted(out, key=pattern_score_tuple_q2, reverse=True)



def generate_pattern_library_q2(
    counts: Dict[str, int],
    catalog: Dict[str, ItemType],
    truck: TruckType,
    fragile_fixed: bool = False,
) -> List[SurfacePacker]:
    cache_key = (truck.name, counts_tuple(counts), fragile_fixed)
    if cache_key in _PATTERN_CACHE_Q2:
        return _PATTERN_CACHE_Q2[cache_key]

    best_by_pattern: Dict[Tuple[str, Tuple[int, ...]], SurfacePacker] = {}

    greedy_solution = greedy_pack_all(counts, catalog, truck, fragile_fixed=fragile_fixed)
    for packer in greedy_solution:
        best_by_pattern[_pattern_key_q2(packer)] = packer

    for subset in PATTERN_SUBSETS_Q2:
        for cap in CAP_GRID_Q2:
            masked = _masked_counts_q2(counts, subset, cap)
            if not any(masked.values()):
                continue
            for policy in PATTERN_POLICIES_Q2:
                packer = SurfacePacker(truck, catalog, masked, policy, fragile_fixed=fragile_fixed).pack()
                if not packer.placed:
                    continue
                key = _pattern_key_q2(packer)
                old = best_by_pattern.get(key)
                if old is None or pattern_score_tuple_q2(packer) > pattern_score_tuple_q2(old):
                    best_by_pattern[key] = packer

    library = _clip_pattern_library_q2(
        list(best_by_pattern.values()),
        keep_main=Q2_TRUCK_LIBRARY_KEEP_MAIN,
        keep_cost=Q2_TRUCK_LIBRARY_KEEP_COST,
        seed_packers=greedy_solution,
    )
    _PATTERN_CACHE_Q2[cache_key] = library
    return library



def generate_joint_pattern_library_q2(
    counts: Dict[str, int],
    catalog: Dict[str, ItemType],
    trucks: Sequence[TruckType],
    fragile_fixed: bool = False,
) -> List[SurfacePacker]:
    all_patterns: List[SurfacePacker] = []
    for truck in trucks:
        all_patterns.extend(generate_pattern_library_q2(counts, catalog, truck, fragile_fixed=fragile_fixed))
    best_by_pattern: Dict[Tuple[str, Tuple[int, ...]], SurfacePacker] = {}
    for packer in all_patterns:
        key = _pattern_key_q2(packer)
        old = best_by_pattern.get(key)
        if old is None or pattern_score_tuple_q2(packer) > pattern_score_tuple_q2(old):
            best_by_pattern[key] = packer
    return _clip_pattern_library_q2(
        list(best_by_pattern.values()),
        keep_main=Q2_JOINT_LIBRARY_KEEP_MAIN,
        keep_cost=Q2_JOINT_LIBRARY_KEEP_COST,
    )



def _build_count_matrix_q2(packers: List[SurfacePacker]) -> np.ndarray:
    return np.array([[packer.loaded_counts[code] for packer in packers] for code in ITEM_ORDER], dtype=float)



def _target_vector_q2(counts: Dict[str, int]) -> np.ndarray:
    return np.array([counts[code] for code in ITEM_ORDER], dtype=float)



def _truck_count_vector_q2(packers: List[SurfacePacker]) -> np.ndarray:
    return np.ones(len(packers), dtype=float)



def _cost_vector_q2(packers: List[SurfacePacker]) -> np.ndarray:
    return np.array([packer.truck.trip_cost for packer in packers], dtype=float)



def _type_indicator_q2(packers: List[SurfacePacker], truck_name: str) -> np.ndarray:
    return np.array([1.0 if packer.truck.name == truck_name else 0.0 for packer in packers], dtype=float)



def _run_milp_exact_cover_q2(
    counts: Dict[str, int],
    packers: List[SurfacePacker],
    objective: np.ndarray,
    exact_truck_count: Optional[int] = None,
    exact_total_cost: Optional[float] = None,
    exact_type_counts: Optional[Dict[str, int]] = None,
    max_total_cost: Optional[float] = None,
    time_limit: float = 90.0,
):
    if not packers:
        return None

    matrix = _build_count_matrix_q2(packers)
    target = _target_vector_q2(counts)

    constraints: List[LinearConstraint] = [LinearConstraint(matrix, target, target)]

    if exact_truck_count is not None:
        ones = _truck_count_vector_q2(packers).reshape(1, -1)
        target_vec = np.array([float(exact_truck_count)])
        constraints.append(LinearConstraint(ones, target_vec, target_vec))

    if exact_total_cost is not None:
        costs = _cost_vector_q2(packers).reshape(1, -1)
        target_vec = np.array([float(exact_total_cost)])
        constraints.append(LinearConstraint(costs, target_vec, target_vec))

    if max_total_cost is not None:
        costs = _cost_vector_q2(packers).reshape(1, -1)
        lb = np.array([-np.inf], dtype=float)
        ub = np.array([float(max_total_cost)], dtype=float)
        constraints.append(LinearConstraint(costs, lb, ub))

    if exact_type_counts:
        for truck_name, wanted in exact_type_counts.items():
            ind = _type_indicator_q2(packers, truck_name).reshape(1, -1)
            target_vec = np.array([float(wanted)])
            constraints.append(LinearConstraint(ind, target_vec, target_vec))

    bounds = Bounds(lb=np.zeros(len(packers)), ub=np.full(len(packers), np.inf))
    integrality = np.ones(len(packers), dtype=int)
    return milp(
        c=objective,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": float(time_limit)},
    )



def _extract_integer_coeffs_q2(result) -> Optional[np.ndarray]:
    if result is None or result.x is None:
        return None
    coeffs = np.rint(result.x).astype(int)
    if np.max(np.abs(result.x - coeffs)) > 1e-5:
        return None
    return coeffs



def _coeffs_to_solution_q2(packers: List[SurfacePacker], coeffs: np.ndarray) -> List[SurfacePacker]:
    out: List[SurfacePacker] = []
    for idx, times in enumerate(coeffs.tolist()):
        if times <= 0:
            continue
        for _ in range(times):
            out.append(copy.deepcopy(packers[idx]))
    return sort_solution_by_fullness_q2(out)



def _solution_counts_q2(packers: List[SurfacePacker]) -> Dict[str, int]:
    acc = {code: 0 for code in ITEM_ORDER}
    for packer in packers:
        for code in ITEM_ORDER:
            acc[code] += packer.loaded_counts[code]
    return acc



def _solution_matches_target_q2(packers: List[SurfacePacker], counts: Dict[str, int]) -> bool:
    return _solution_counts_q2(packers) == {code: counts[code] for code in ITEM_ORDER}



def scheme_total_cost(packers: List[SurfacePacker]) -> float:
    return float(sum(packer.truck.trip_cost for packer in packers))



def _rounded_cost_q2(value: float) -> int:
    return int(round(float(value)))



def _solution_satisfies_constraints_q2(
    packers: List[SurfacePacker],
    counts: Dict[str, int],
    exact_truck_count: Optional[int] = None,
    exact_total_cost: Optional[float] = None,
    exact_type_counts: Optional[Dict[str, int]] = None,
    max_total_cost: Optional[float] = None,
) -> bool:
    if not _solution_matches_target_q2(packers, counts):
        return False
    if exact_truck_count is not None and len(packers) != int(exact_truck_count):
        return False
    total_cost = _rounded_cost_q2(scheme_total_cost(packers))
    if exact_total_cost is not None and total_cost != _rounded_cost_q2(exact_total_cost):
        return False
    if max_total_cost is not None and total_cost > _rounded_cost_q2(max_total_cost):
        return False
    if exact_type_counts:
        type_counts = scheme_truck_type_counts(packers)
        for truck_name, wanted in exact_type_counts.items():
            if type_counts.get(truck_name, 0) != int(wanted):
                return False
    return True



def _extract_valid_solution_from_result_q2(
    result,
    packers: List[SurfacePacker],
    counts: Dict[str, int],
    exact_truck_count: Optional[int] = None,
    exact_total_cost: Optional[float] = None,
    exact_type_counts: Optional[Dict[str, int]] = None,
    max_total_cost: Optional[float] = None,
    require_success: bool = False,
) -> Optional[List[SurfacePacker]]:
    if result is None or result.x is None:
        return None
    if require_success and not bool(getattr(result, "success", False)):
        return None
    coeffs = _extract_integer_coeffs_q2(result)
    if coeffs is None:
        return None
    solution = _coeffs_to_solution_q2(packers, coeffs)
    if not _solution_satisfies_constraints_q2(
        solution,
        counts,
        exact_truck_count=exact_truck_count,
        exact_total_cost=exact_total_cost,
        exact_type_counts=exact_type_counts,
        max_total_cost=max_total_cost,
    ):
        return None
    return solution



def _truck_cost_step_q2(trucks: Sequence[TruckType]) -> int:
    step = 0
    for truck in trucks:
        step = math.gcd(step, int(round(float(truck.trip_cost))))
    return max(1, step)



def scheme_truck_type_counts(packers: List[SurfacePacker]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for packer in packers:
        counts[packer.truck.name] = counts.get(packer.truck.name, 0) + 1
    return counts



def scheme_fill_stats(packers: List[SurfacePacker]) -> Dict[str, float]:
    if not packers:
        return {
            "min_空间利用率_eff": 0.0,
            "avg_空间利用率_eff": 0.0,
            "min_载重利用率": 0.0,
            "avg_载重利用率": 0.0,
            "min_min利用率": 0.0,
            "avg_min利用率": 0.0,
        }
    v_list = [p.utilization()["空间利用率_eff"] for p in packers]
    w_list = [p.utilization()["载重利用率"] for p in packers]
    min_list = [min(v, w) for v, w in zip(v_list, w_list)]
    return {
        "min_空间利用率_eff": float(min(v_list)),
        "avg_空间利用率_eff": float(sum(v_list) / len(v_list)),
        "min_载重利用率": float(min(w_list)),
        "avg_载重利用率": float(sum(w_list) / len(w_list)),
        "min_min利用率": float(min(min_list)),
        "avg_min利用率": float(sum(min_list) / len(min_list)),
    }



def build_scheme_bundle_q2(name: str, packers: List[SurfacePacker]) -> Dict[str, object]:
    type_counts = scheme_truck_type_counts(packers)
    fill_stats = scheme_fill_stats(packers)
    return {
        "scheme_name": name,
        "packers": sort_solution_by_fullness_q2(packers),
        "truck_count": len(packers),
        "total_cost": scheme_total_cost(packers),
        "truck_type_counts": type_counts,
        **fill_stats,
    }



def _fallback_greedy_multi_type_q2(
    counts: Dict[str, int],
    catalog: Dict[str, ItemType],
    trucks: Sequence[TruckType],
    objective: str,
    fragile_fixed: bool = False,
) -> List[SurfacePacker]:
    remaining = {code: counts[code] for code in ITEM_ORDER}
    solution: List[SurfacePacker] = []

    while not is_zero_counts(remaining):
        best: Optional[SurfacePacker] = None
        best_key = None
        for truck in trucks:
            candidates = generate_candidate_packers(remaining, catalog, truck, fragile_fixed=fragile_fixed, top_k=2)
            for cand in candidates:
                util = cand.utilization()
                v = util["空间利用率_eff"]
                w = util["载重利用率"]
                if objective == "cost":
                    key = (
                        cand.total_volume / truck.trip_cost,
                        cand.total_weight / truck.trip_cost,
                        min(v, w),
                        v,
                        -truck.trip_cost,
                        len(cand.placed),
                    )
                else:
                    key = (
                        cand.total_volume,
                        cand.total_weight,
                        min(v, w),
                        v,
                        -truck.trip_cost,
                        len(cand.placed),
                    )
                if best_key is None or key > best_key:
                    best_key = key
                    best = cand
        if best is None:
            raise RuntimeError("多车型贪心回退未能继续装载，建议检查候选模式。")
        solution.append(best)
        remaining = subtract_counts(remaining, best.loaded_counts)

    return sort_solution_by_fullness_q2(solution)



def _find_any_solution_with_cost_cap_q2(
    counts: Dict[str, int],
    library: List[SurfacePacker],
    cost_cap: int,
    time_limit: float,
) -> Optional[List[SurfacePacker]]:
    if cost_cap < 0:
        return None
    result = _run_milp_exact_cover_q2(
        counts,
        library,
        objective=np.zeros(len(library), dtype=float),
        exact_truck_count=None,
        exact_total_cost=None,
        exact_type_counts=None,
        max_total_cost=float(cost_cap),
        time_limit=time_limit,
    )
    return _extract_valid_solution_from_result_q2(
        result,
        library,
        counts,
        max_total_cost=float(cost_cap),
    )



def _find_min_feasible_cost_under_upper_q2(
    counts: Dict[str, int],
    library: List[SurfacePacker],
    upper_cost: int,
    cost_step: int,
) -> Tuple[int, Optional[List[SurfacePacker]]]:
    upper_cost = _rounded_cost_q2(upper_cost)
    cheaper_cap = upper_cost - cost_step
    if cheaper_cap < 0:
        return upper_cost, None

    quick_solution = _find_any_solution_with_cost_cap_q2(
        counts,
        library,
        cheaper_cap,
        time_limit=Q2_TIME_LIMIT_CHEAPER_CHECK,
    )
    if quick_solution is None:
        return upper_cost, None

    caps = list(range(0, upper_cost + cost_step, cost_step))
    lo = 0
    hi = len(caps) - 1
    best_cap = upper_cost
    best_solution: Optional[List[SurfacePacker]] = None

    while lo <= hi:
        mid = (lo + hi) // 2
        cap = caps[mid]
        solution = _find_any_solution_with_cost_cap_q2(
            counts,
            library,
            cap,
            time_limit=Q2_TIME_LIMIT_CHEAPER_SEARCH,
        )
        if solution is not None:
            best_cap = cap
            best_solution = solution
            hi = mid - 1
        else:
            lo = mid + 1

    if best_solution is None:
        return upper_cost, None
    return _rounded_cost_q2(scheme_total_cost(best_solution)), best_solution



def _highest_feasible_fill_threshold_q2(
    counts: Dict[str, int],
    library: List[SurfacePacker],
    exact_truck_count: Optional[int] = None,
    exact_total_cost: Optional[float] = None,
    exact_type_counts: Optional[Dict[str, int]] = None,
) -> float:
    values = sorted({pattern_min_fill_q2(packer) for packer in library})
    if not values:
        return 0.0

    lo = 0
    hi = len(values) - 1
    best = 0.0
    while lo <= hi:
        mid = (lo + hi) // 2
        threshold = values[mid]
        filtered = [packer for packer in library if pattern_min_fill_q2(packer) + 1e-9 >= threshold]
        result = _run_milp_exact_cover_q2(
            counts,
            filtered,
            objective=np.zeros(len(filtered), dtype=float),
            exact_truck_count=exact_truck_count,
            exact_total_cost=exact_total_cost,
            exact_type_counts=exact_type_counts,
            time_limit=Q2_TIME_LIMIT_THRESHOLD,
        )
        solution = _extract_valid_solution_from_result_q2(
            result,
            filtered,
            counts,
            exact_truck_count=exact_truck_count,
            exact_total_cost=exact_total_cost,
            exact_type_counts=exact_type_counts,
        )
        if solution is not None:
            best = threshold
            lo = mid + 1
            continue
        hi = mid - 1
    return best



def optimize_balance_with_constraints_q2(
    counts: Dict[str, int],
    library: List[SurfacePacker],
    exact_truck_count: Optional[int] = None,
    exact_total_cost: Optional[float] = None,
    exact_type_counts: Optional[Dict[str, int]] = None,
) -> List[SurfacePacker]:
    threshold = _highest_feasible_fill_threshold_q2(
        counts,
        library,
        exact_truck_count=exact_truck_count,
        exact_total_cost=exact_total_cost,
        exact_type_counts=exact_type_counts,
    )
    filtered = [packer for packer in library if pattern_min_fill_q2(packer) + 1e-9 >= threshold]
    if not filtered:
        filtered = library

    objective = -np.array([pattern_secondary_score_q2(packer) for packer in filtered], dtype=float)
    result = _run_milp_exact_cover_q2(
        counts,
        filtered,
        objective=objective,
        exact_truck_count=exact_truck_count,
        exact_total_cost=exact_total_cost,
        exact_type_counts=exact_type_counts,
        time_limit=Q2_TIME_LIMIT_BALANCE,
    )
    solution = _extract_valid_solution_from_result_q2(
        result,
        filtered,
        counts,
        exact_truck_count=exact_truck_count,
        exact_total_cost=exact_total_cost,
        exact_type_counts=exact_type_counts,
    )
    if solution is not None:
        return solution

    fallback_result = _run_milp_exact_cover_q2(
        counts,
        library,
        objective=-np.array([pattern_secondary_score_q2(packer) for packer in library], dtype=float),
        exact_truck_count=exact_truck_count,
        exact_total_cost=exact_total_cost,
        exact_type_counts=exact_type_counts,
        time_limit=Q2_TIME_LIMIT_BALANCE,
    )
    fallback_solution = _extract_valid_solution_from_result_q2(
        fallback_result,
        library,
        counts,
        exact_truck_count=exact_truck_count,
        exact_total_cost=exact_total_cost,
        exact_type_counts=exact_type_counts,
    )
    if fallback_solution is not None:
        return fallback_solution

    return []



def solve_min_total_trucks_q2(
    counts: Dict[str, int],
    catalog: Dict[str, ItemType],
    trucks: Sequence[TruckType],
    fragile_fixed: bool = False,
) -> Dict[str, object]:
    library = generate_joint_pattern_library_q2(counts, catalog, trucks, fragile_fixed=fragile_fixed)
    result1 = _run_milp_exact_cover_q2(
        counts,
        library,
        objective=np.ones(len(library), dtype=float),
        exact_truck_count=None,
        exact_total_cost=None,
        time_limit=Q2_TIME_LIMIT_MIN_TRUCKS_STAGE1,
    )
    stage1_solution = _extract_valid_solution_from_result_q2(result1, library, counts)
    if stage1_solution is None:
        fallback = _fallback_greedy_multi_type_q2(counts, catalog, trucks, objective="truck_count", fragile_fixed=fragile_fixed)
        return build_scheme_bundle_q2("Q2(1)_min_total_trucks_fallback", fallback)

    truck_count = len(stage1_solution)

    costs = _cost_vector_q2(library)
    result2 = _run_milp_exact_cover_q2(
        counts,
        library,
        objective=costs,
        exact_truck_count=truck_count,
        exact_total_cost=None,
        time_limit=Q2_TIME_LIMIT_MIN_TRUCKS_STAGE2,
    )
    stage2_solution = _extract_valid_solution_from_result_q2(
        result2,
        library,
        counts,
        exact_truck_count=truck_count,
    )
    if stage2_solution is None:
        return build_scheme_bundle_q2("Q2(1)_min_total_trucks", stage1_solution)

    min_cost_at_min_trucks = scheme_total_cost(stage2_solution)
    balanced = optimize_balance_with_constraints_q2(
        counts,
        library,
        exact_truck_count=truck_count,
        exact_total_cost=min_cost_at_min_trucks,
    )
    if balanced:
        return build_scheme_bundle_q2("Q2(1)_min_total_trucks", balanced)

    return build_scheme_bundle_q2("Q2(1)_min_total_trucks", stage2_solution)



def solve_min_total_cost_q2(
    counts: Dict[str, int],
    catalog: Dict[str, ItemType],
    trucks: Sequence[TruckType],
    fragile_fixed: bool = False,
    upper_bound_bundle: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    library = generate_joint_pattern_library_q2(counts, catalog, trucks, fragile_fixed=fragile_fixed)
    costs = _cost_vector_q2(library)

    fallback_packers = _fallback_greedy_multi_type_q2(counts, catalog, trucks, objective="cost", fragile_fixed=fragile_fixed)
    best_bundle = build_scheme_bundle_q2("Q2(2)_min_total_cost_fallback", fallback_packers)
    if upper_bound_bundle is not None and upper_bound_bundle["total_cost"] < best_bundle["total_cost"]:
        best_bundle = build_scheme_bundle_q2("Q2(2)_min_total_cost_upper_bound", upper_bound_bundle["packers"])

    # 先做一轮较短的“成本最小化”MILP；只有当它真的给出了更低成本的可行方案时才采纳。
    result1 = _run_milp_exact_cover_q2(
        counts,
        library,
        objective=costs,
        exact_truck_count=None,
        exact_total_cost=None,
        time_limit=Q2_TIME_LIMIT_COST_STAGE1,
    )
    stage1_solution = _extract_valid_solution_from_result_q2(result1, library, counts)
    stage1_proven_optimal = stage1_solution is not None and bool(getattr(result1, "success", False))
    if stage1_solution is not None:
        stage1_bundle = build_scheme_bundle_q2("Q2(2)_min_total_cost_stage1", stage1_solution)
        if stage1_bundle["total_cost"] < best_bundle["total_cost"]:
            best_bundle = stage1_bundle

    cost_step = _truck_cost_step_q2(trucks)
    upper_cost = _rounded_cost_q2(best_bundle["total_cost"])

    # 如果当前最好成本已经有可行上界，先快速检查“再便宜一个步长”是否可行；
    # 不可行则直接返回该上界，避免因为 MILP 超时把更贵方案误标成最低成本。
    if not (stage1_proven_optimal and _rounded_cost_q2(best_bundle["total_cost"]) == upper_cost):
        cheaper_cost, cheaper_solution = _find_min_feasible_cost_under_upper_q2(
            counts,
            library,
            upper_cost,
            cost_step,
        )
        if cheaper_solution is not None and cheaper_cost < upper_cost:
            best_bundle = build_scheme_bundle_q2("Q2(2)_min_total_cost", cheaper_solution)
            upper_cost = cheaper_cost
        elif upper_bound_bundle is not None and _rounded_cost_q2(upper_bound_bundle["total_cost"]) == upper_cost:
            # 当前库里没有找到比上界更便宜的方案，直接返回这个已知更优可行解。
            return build_scheme_bundle_q2("Q2(2)_min_total_cost", upper_bound_bundle["packers"])

    target_cost = _rounded_cost_q2(best_bundle["total_cost"])
    result2 = _run_milp_exact_cover_q2(
        counts,
        library,
        objective=np.ones(len(library), dtype=float),
        exact_truck_count=None,
        exact_total_cost=target_cost,
        time_limit=Q2_TIME_LIMIT_COST_STAGE2,
    )
    stage2_solution = _extract_valid_solution_from_result_q2(
        result2,
        library,
        counts,
        exact_total_cost=target_cost,
    )
    if stage2_solution is None:
        return build_scheme_bundle_q2("Q2(2)_min_total_cost", best_bundle["packers"])

    truck_count = len(stage2_solution)
    balanced = optimize_balance_with_constraints_q2(
        counts,
        library,
        exact_truck_count=truck_count,
        exact_total_cost=target_cost,
    )
    if balanced:
        return build_scheme_bundle_q2("Q2(2)_min_total_cost", balanced)

    return build_scheme_bundle_q2("Q2(2)_min_total_cost", stage2_solution)



def compare_q2_schemes(min_trucks_bundle: Dict[str, object], min_cost_bundle: Dict[str, object]) -> Dict[str, object]:
    return {
        "min_total_trucks_vehicle_count": int(min_trucks_bundle["truck_count"]),
        "min_total_trucks_total_cost": float(min_trucks_bundle["total_cost"]),
        "min_total_trucks_type_counts": dict(min_trucks_bundle["truck_type_counts"]),
        "min_total_cost_vehicle_count": int(min_cost_bundle["truck_count"]),
        "min_total_cost_total_cost": float(min_cost_bundle["total_cost"]),
        "min_total_cost_type_counts": dict(min_cost_bundle["truck_type_counts"]),
        "cost_gap_min_trucks_minus_min_cost": float(min_trucks_bundle["total_cost"] - min_cost_bundle["total_cost"]),
        "vehicle_gap_min_cost_minus_min_trucks": int(min_cost_bundle["truck_count"] - min_trucks_bundle["truck_count"]),
    }



def solve_question2(fragile_fixed: bool = False) -> Dict[str, object]:
    trucks, catalog = build_problem_data()
    counts = counts_from_catalog(catalog)

    min_trucks_bundle = solve_min_total_trucks_q2(counts, catalog, trucks, fragile_fixed=fragile_fixed)
    min_cost_bundle = solve_min_total_cost_q2(
        counts,
        catalog,
        trucks,
        fragile_fixed=fragile_fixed,
        upper_bound_bundle=min_trucks_bundle,
    )

    return {
        "catalog": catalog,
        "min_total_trucks": min_trucks_bundle,
        "min_total_cost": min_cost_bundle,
        "comparison": compare_q2_schemes(min_trucks_bundle, min_cost_bundle),
    }



def print_scheme_summary_q2(title: str, bundle: Dict[str, object]) -> None:
    packers = bundle["packers"]
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"总车辆数: {bundle['truck_count']}")
    print(f"总运输成本: {bundle['total_cost']:.0f} 元")
    print("车型分布: " + ", ".join(f"{k}={v}" for k, v in sorted(bundle["truck_type_counts"].items())))
    print(f"最差 min(满容,满载): {bundle['min_min利用率']:.4f}")
    print(f"平均 min(满容,满载): {bundle['avg_min利用率']:.4f}")
    print()
    for i, p in enumerate(packers, start=1):
        util = p.utilization()
        print(f"车辆 {i} | {p.truck.name} | 件数 {len(p.placed)} | 成本 {p.truck.trip_cost:.0f}")
        print(f"  空间利用率(eff): {util['空间利用率_eff']:.4f}")
        print(f"  载重利用率     : {util['载重利用率']:.4f}")
        print(f"  已装体积       : {util['已装货物体积_m3']:.3f} m^3")
        print(f"  已装重量       : {util['已装货物重量_kg']:.1f} kg")
        print("  货类计数       : " + ", ".join(f"{code}={p.loaded_counts[code]}" for code in ITEM_ORDER))
        print()


if __name__ == "__main__":
    result = solve_question2(fragile_fixed=False)
    print_scheme_summary_q2("Q2(1) 两车型下总车辆数最少", result["min_total_trucks"])
    print_scheme_summary_q2("Q2(2) 两车型下总运输成本最低", result["min_total_cost"])
    print("=" * 78)
    print("方案对比")
    print("=" * 78)
    for k, v in result["comparison"].items():
        print(f"{k}: {v}")
