import os
import tempfile
import logging
import numpy as np
import trimesh
import gradio as gr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("3d-cutter")

MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024

_TEMP_FILES: set[str] = set()


def _cleanup_temp_files() -> None:
    """Удаляет старые временные файлы от предыдущих операций."""
    for path in list(_TEMP_FILES):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            log.warning("Не удалось удалить временный файл %s: %s", path, e)
        _TEMP_FILES.discard(path)


def _track_temp_file(path: str) -> str:
    _TEMP_FILES.add(path)
    return path


def load_txt_mesh(file_path: str) -> trimesh.Trimesh:
    vertices, faces = [], []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
            else:
                try:
                    vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except (ValueError, IndexError):
                    continue

    if not vertices:
        raise ValueError("В файле не найдено ни одной вершины")

    verts = np.asarray(vertices, dtype=float)
    if faces:
        return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=int))
    try:
        return trimesh.Trimesh(vertices=verts).convex_hull
    except Exception:
        return trimesh.Trimesh(vertices=verts)


def load_model(file_path: str) -> trimesh.Trimesh:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return load_txt_mesh(file_path)

    loaded = trimesh.load(file_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError("Пустая сцена в файле")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded

def save_txt_mesh(mesh: trimesh.Trimesh, path: str):
    with open(path, "w") as f:
        f.write("# Cut mesh by 3D Cutter\n")
        f.write(f"# vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}\n\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for face in mesh.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def export_preview(mesh: trimesh.Trimesh) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as f:
            path = _track_temp_file(f.name)
        mesh.export(path)
        return path
    except Exception:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as f:
            path = _track_temp_file(f.name)
        mesh.export(path)
        return path


EMPTY_INFO = "**Шаг 2.** Здесь появится информация о модели после её загрузки."


def analyze_model(file):
    _cleanup_temp_files()

    if file is None:
        return EMPTY_INFO, None, 0.0, 0.0, 0.0, None

    try:
        file_size = os.path.getsize(file.name)
    except OSError as e:
        return f"**Ошибка чтения файла:** `{e}`", None, 0.0, 0.0, 0.0, None

    if file_size > MAX_FILE_SIZE:
        return (
            f"**Файл слишком велик:** {file_size / (1024 * 1024):.1f} МБ (макс. {MAX_FILE_SIZE_MB} МБ).",
            None,
            0.0,
            0.0,
            0.0,
            None,
        )

    try:
        mesh = load_model(file.name)
    except Exception as e:
        return f"**Ошибка чтения файла:** `{e}`", None, 0.0, 0.0, 0.0, None

    bmin, bmax = mesh.bounds
    center = mesh.centroid
    size = bmax - bmin

    try:
        volume = f"{mesh.volume:.4f}"
    except Exception:
        volume = "—"

    watertight = "да" if mesh.is_watertight else "нет"
    watertight_warn = "" if mesh.is_watertight else "\n\n⚠️ Модель не замкнута — объём и обрезка могут быть неточными."

    info = f"""**Шаг 2.** Информация о модели:

| | X | Y | Z |
|:---|---:|---:|---:|
| **Минимум** | {bmin[0]:.3f} | {bmin[1]:.3f} | {bmin[2]:.3f} |
| **Максимум** | {bmax[0]:.3f} | {bmax[1]:.3f} | {bmax[2]:.3f} |
| **Размер** | {size[0]:.3f} | {size[1]:.3f} | {size[2]:.3f} |
| **Центр** | {center[0]:.3f} | {center[1]:.3f} | {center[2]:.3f} |

Вершин: **{len(mesh.vertices):,}** · Граней: **{len(mesh.faces):,}** · Объём: **{volume}** · Замкнута: **{watertight}**{watertight_warn}
"""
    preview = export_preview(mesh)
    return info, preview, float(center[0]), float(center[1]), float(center[2]), mesh


def preset_x():   return 1.0, 0.0, 0.0
def preset_y():   return 0.0, 1.0, 0.0
def preset_z():   return 0.0, 0.0, 1.0
def preset_xy():  return 1.0, 1.0, 0.0
def preset_xyz(): return 1.0, 1.0, 1.0


def _apply_single_cut(mesh: trimesh.Trimesh, px, py, pz, nx, ny, nz, keep_positive):
    normal = np.array([nx, ny, nz], dtype=float)
    if np.linalg.norm(normal) < 1e-9:
        raise ValueError("Нормаль не может быть нулевой.")
    normal /= np.linalg.norm(normal)

    point = np.array([px, py, pz], dtype=float)

    bmin, bmax = mesh.bounds
    corners = np.array([
        [bmin[0], bmin[1], bmin[2]], [bmax[0], bmin[1], bmin[2]],
        [bmin[0], bmax[1], bmin[2]], [bmax[0], bmax[1], bmin[2]],
        [bmin[0], bmin[1], bmax[2]], [bmax[0], bmin[1], bmax[2]],
        [bmin[0], bmax[1], bmax[2]], [bmax[0], bmax[1], bmax[2]],
    ])
    dists = (corners - point) @ normal
    if np.all(dists > 0) or np.all(dists < 0):
        raise ValueError(
            "Плоскость не пересекает модель.\n\n"
            f"Границы: min `({bmin[0]:.3f}, {bmin[1]:.3f}, {bmin[2]:.3f})`, "
            f"max `({bmax[0]:.3f}, {bmax[1]:.3f}, {bmax[2]:.3f})`\n\n"
            f"Центр: `({mesh.centroid[0]:.3f}, {mesh.centroid[1]:.3f}, {mesh.centroid[2]:.3f})`\n\n"
            "Выберите точку внутри границ модели."
        )

    cut = mesh.slice_plane(plane_origin=point, plane_normal=normal if keep_positive else -normal)
    if cut is None or len(cut.vertices) == 0:
        raise ValueError("Результат обрезки пустой.")
    return cut


def cut_mesh(file, current_mesh, px, py, pz, nx, ny, nz, keep_positive):
    _cleanup_temp_files()

    if file is None and current_mesh is None:
        return None, None, None, None, None, "**Шаг 5.** Сначала загрузите файл модели."

    try:
        if current_mesh is not None:
            mesh = current_mesh
        else:
            if file is None:
                return None, None, None, None, None, "**Шаг 5.** Сначала загрузите файл модели."
            file_size = os.path.getsize(file.name)
            if file_size > MAX_FILE_SIZE:
                return (
                    None, None, None, None, None,
                    f"**Файл слишком велик:** {file_size / (1024 * 1024):.1f} МБ (макс. {MAX_FILE_SIZE_MB} МБ).",
                )
            mesh = load_model(file.name)
    except OSError as e:
        return None, None, None, None, None, f"**Ошибка чтения файла:** `{e}`"
    except Exception as e:
        return None, None, None, None, None, f"**Ошибка загрузки:** `{e}`"

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        return None, None, None, None, None, "**Не удалось распознать меш в файле.**"

    try:
        cut = _apply_single_cut(mesh, px, py, pz, nx, ny, nz, keep_positive)
    except ValueError as e:
        return None, None, None, None, mesh, f"**Ошибка обрезки:** `{e}`"
    except Exception as e:
        return None, None, None, None, mesh, f"**Ошибка обрезки:** `{e}`"

    txt_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        txt_path = _track_temp_file(f.name)
    save_txt_mesh(cut, txt_path)

    stl_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as f:
            stl_path = _track_temp_file(f.name)
        cut.export(stl_path)
    except Exception:
        stl_path = None

    obj_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as f:
            obj_path = _track_temp_file(f.name)
        cut.export(obj_path)
    except Exception:
        obj_path = None

    ply_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as f:
            ply_path = _track_temp_file(f.name)
        cut.export(ply_path)
    except Exception:
        ply_path = None

    glb_path = export_preview(cut)

    status = (
        "**Шаг 5. Готово.**\n\n"
        "| | Было | Стало |\n"
        "|:---|---:|---:|\n"
        f"| Вершин | {len(mesh.vertices):,} | **{len(cut.vertices):,}** |\n"
        f"| Граней | {len(mesh.faces):,} | **{len(cut.faces):,}** |\n\n"
        f"Плоскость: точка `({px}, {py}, {pz})`, нормаль `({nx}, {ny}, {nz})`"
    )
    return txt_path, stl_path, obj_path, ply_path, glb_path, glb_path, cut, status


def cut_mesh_batch(file, current_mesh, planes_text, keep_positive):
    """Последовательная обрезка по нескольким плоскостям.

    planes_text — текст с одной плоскостью на строку, форматы:
    "px py pz nx ny nz" или "px,py,pz,nx,ny,nz"
    """
    _cleanup_temp_files()

    if file is None and current_mesh is None:
        return None, None, None, None, None, "**Шаг 5.** Сначала загрузите файл модели."

    try:
        if current_mesh is not None:
            mesh = current_mesh
        else:
            if file is None:
                return None, None, None, None, None, "**Шаг 5.** Сначала загрузите файл модели."
            file_size = os.path.getsize(file.name)
            if file_size > MAX_FILE_SIZE:
                return (
                    None, None, None, None, None,
                    f"**Файл слишком велик:** {file_size / (1024 * 1024):.1f} МБ (макс. {MAX_FILE_SIZE_MB} МБ).",
                )
            mesh = load_model(file.name)
    except OSError as e:
        return None, None, None, None, None, f"**Ошибка чтения файла:** `{e}`"
    except Exception as e:
        return None, None, None, None, None, f"**Ошибка загрузки:** `{e}`"

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        return None, None, None, None, None, "**Не удалось распознать меш в файле.**"

    planes = []
    for line_num, line in enumerate(planes_text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) != 6:
            return None, None, None, None, mesh, (
                f"**Ошибка в строке {line_num}:** ожидалось 6 чисел (px py pz nx ny nz), "
                f"получено {len(parts)}"
            )
        try:
            px, py, pz, nx, ny, nz = (float(p) for p in parts)
        except ValueError as e:
            return None, None, None, None, mesh, (
                f"**Ошибка в строке {line_num}:** неверное число (`{e}`)"
            )
        if all(v == 0 for v in (px, py, pz, nx, ny, nz)):
            continue
        planes.append((px, py, pz, nx, ny, nz))

    if not planes:
        return None, None, None, None, mesh, "**Список плоскостей пуст.**"

    current = mesh
    total_vertices = len(current.vertices)
    total_faces = len(current.faces)

    for i, (px, py, pz, nx, ny, nz) in enumerate(planes):
        try:
            current = _apply_single_cut(current, px, py, pz, nx, ny, nz, keep_positive)
        except ValueError as e:
            return None, None, None, None, current, (
                f"**Ошибка обрезки плоскости #{i + 1}:** `{e}`"
            )
        except Exception as e:
            return None, None, None, None, current, (
                f"**Ошибка обрезки плоскости #{i + 1}:** `{e}`"
            )

    txt_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        txt_path = _track_temp_file(f.name)
    save_txt_mesh(current, txt_path)

    stl_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as f:
            stl_path = _track_temp_file(f.name)
        current.export(stl_path)
    except Exception:
        stl_path = None

    obj_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as f:
            obj_path = _track_temp_file(f.name)
        current.export(obj_path)
    except Exception:
        obj_path = None

    ply_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as f:
            ply_path = _track_temp_file(f.name)
        current.export(ply_path)
    except Exception:
        ply_path = None

    glb_path = export_preview(current)

    status = (
        "**Шаг 5. Готово (пакетная обрезка).**\n\n"
        "| | Было | Стало |\n"
        "|:---|---:|---:|\n"
        f"| Вершин | {total_vertices:,} | **{len(current.vertices):,}** |\n"
        f"| Граней | {total_faces:,} | **{len(current.faces):,}** |\n\n"
        f"Плоскостей применено: **{len(planes)}**"
    )
    return txt_path, stl_path, obj_path, ply_path, glb_path, glb_path, current, status


def reset_mesh(file):
    """Сброс состояния меша к исходной модели из файла."""
    if file is None:
        return None, None, None, "**Нет загруженной модели для сброса.**"
    try:
        mesh = load_model(file.name)
        preview = export_preview(mesh)
        return mesh, preview, preview, "**Состояние сброшено к исходной модели.**"
    except Exception as e:
        return None, None, None, f"**Ошибка сброса:** `{e}`"

CUSTOM_CSS = """
/* ============ Общий фон ============ */
html, body,
body > div,
gradio-app,
.gradio-container,
.gradio-container > .main {
    background: #EEF1F5 !important;
    border: none !important;
}
footer { display: none !important; }

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 32px 24px !important;
}

/* ============ Хедер ============ */
#app-header {
    padding: 0 0 20px 0;
    margin-bottom: 16px;
    border-bottom: 1px solid #DDE2EA;
}
#app-header h1 {
    color: #111827;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
}
#app-header p {
    color: #6B7280;
    font-size: 0.9rem;
    margin: 6px 0 0 0;
}

/* ============ Карточки ============ */
.card-block {
    padding: 20px 22px !important;
    margin-bottom: 16px !important;
    background: #FFFFFF !important;
    border: 1px solid #DDE2EA !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
    overflow: hidden !important;
}
.card-block:last-child {
    margin-bottom: 0 !important;
}

/* ============ Заголовки шагов ============ */
.step-title p,
.step-title strong {
    color: #111827 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
    letter-spacing: -0.01em;
}

/* ============ Подсказки ============ */
.hint-text p {
    color: #6B7280 !important;
    font-size: 0.85rem !important;
    margin: 12px 0 6px 0 !important;
}

/* ============ Поля ввода ============ */
input[type="number"],
input[type="text"],
textarea {
    background: #F3F4F6 !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    color: #111827 !important;
    padding: 9px 12px !important;
    font-size: 0.9rem !important;
    box-shadow: none !important;
}
input[type="number"]:hover,
input[type="text"]:hover {
    background: #EEF0F3 !important;
}
input[type="number"]:focus,
input[type="text"]:focus {
    background: #FFFFFF !important;
    border-color: #111827 !important;
    outline: none !important;
}

/* Обёртки полей — без фона */
.gr-input, .gr-number, [data-testid="number-input"] {
    background: transparent !important;
    border: none !important;
}

/* Подписи полей */
label, label > span, .block-title, .block-label {
    background: transparent !important;
    color: #6B7280 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* ============ Кнопки-пресеты ============ */
.preset-row button,
button.gr-button-secondary {
    background: #F3F4F6 !important;
    color: #374151 !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 8px 14px !important;
    box-shadow: none !important;
    transition: all 0.15s ease;
}
.preset-row button:hover,
button.gr-button-secondary:hover {
    background: #E5E7EB !important;
    color: #111827 !important;
}

/* ============ Главная кнопка ============ */
.cut-btn button,
button.gr-button-primary {
    background: #111827 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 14px 24px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    transition: all 0.15s ease;
    width: 100% !important;
}
.cut-btn button:hover,
button.gr-button-primary:hover {
    background: #1F2937 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
}

/* ============ Зона загрузки файла ============ */
.file-preview,
.file-drop,
[data-testid="file"] {
    background: #F9FAFB !important;
    border: 2px dashed #C7D2DE !important;
    border-radius: 12px !important;
    transition: all 0.15s ease;
}
.file-preview:hover,
.file-drop:hover,
[data-testid="file"]:hover {
    background: #F3F4F6 !important;
    border-color: #9CA3AF !important;
}

/* ============ Превью моделей ============ */
.viewer-box,
.viewer-box > div,
.viewer-box .empty,
.viewer-box .icon-wrap,
.viewer-box canvas {
    background: #F9FAFB !important;
}
.viewer-box {
    border: 1px solid #DDE2EA !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ============ Скачивание файлов ============ */
.card-block .file-preview,
.card-block .download-row .file-preview,
.card-block .download-row [data-testid="file"] {
    background: #F9FAFB !important;
    border: 1.5px solid #DDE2EA !important;
    border-radius: 10px !important;
    padding: 12px !important;
    min-height: 90px !important;
    transition: all 0.15s ease;
}
.card-block .file-preview:hover,
.card-block .download-row [data-testid="file"]:hover {
    background: #F3F4F6 !important;
    border-color: #C7D2DE !important;
}

/* ============ Радио-кнопки ============ */
fieldset {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
fieldset > span:first-child,
fieldset > legend {
    display: none !important;
}

fieldset label {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 6px 0 !important;
    margin: 0 !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: #374151 !important;
    font-size: 0.9rem !important;
    cursor: pointer;
}
fieldset label:hover {
    background: transparent !important;
    color: #111827 !important;
}

fieldset input[type="radio"] {
    width: 16px !important;
    height: 16px !important;
    margin: 0 !important;
    padding: 0 !important;
    flex-shrink: 0 !important;
    appearance: none !important;
    -webkit-appearance: none !important;
    background: #FFFFFF !important;
    border: 1.5px solid #C7D2DE !important;
    border-radius: 50% !important;
    position: relative !important;
    cursor: pointer;
    box-shadow: none !important;
}
fieldset input[type="radio"]:checked {
    background: #FFFFFF !important;
    border-color: #111827 !important;
}
fieldset input[type="radio"]:checked::after {
    content: "" !important;
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    width: 8px !important;
    height: 8px !important;
    background: #111827 !important;
    border-radius: 50% !important;
}

/* ============ Текст ============ */
.gradio-container p,
.gradio-container td,
.gradio-container th,
.gradio-container strong,
.gradio-container span {
    color: #111827;
}

.gradio-container table {
    font-size: 0.9rem;
    color: #111827;
}
.gradio-container table th {
    color: #6B7280 !important;
    font-weight: 500 !important;
}

/* ============ Футер ============ */
#app-footer {
    color: #9CA3AF;
    font-size: 0.8rem;
    text-align: center;
    padding: 24px 0 8px 0;
}

/* ============ Адаптивность ============ */
@media (max-width: 768px) {
    .gradio-container {
        padding: 16px 12px !important;
    }

    .card-block {
        padding: 16px 18px !important;
        margin-bottom: 12px !important;
        overflow: hidden !important;
    }

    #app-header h1 {
        font-size: 1.35rem !important;
    }

    #app-header p {
        font-size: 0.85rem !important;
    }

    .step-title p,
    .step-title strong {
        font-size: 0.9rem !important;
    }

    .hint-text p {
        font-size: 0.8rem !important;
        margin: 8px 0 4px 0 !important;
    }

    input[type="number"],
    input[type="text"],
    textarea {
        font-size: 0.85rem !important;
        padding: 8px 10px !important;
    }

    .preset-row button,
    button.gr-button-secondary {
        font-size: 0.8rem !important;
        padding: 6px 10px !important;
    }

    .cut-btn button,
    button.gr-button-primary {
        font-size: 0.9rem !important;
        padding: 12px 16px !important;
    }

    /* На маленьких экранах колонки идут одной строкой */
    .gradio-container > .main .gr-column {
        width: 100% !important;
    }

    .gradio-container .gr-row {
        flex-wrap: wrap !important;
        gap: 8px !important;
    }

    .gradio-container .gr-row > * {
        flex: 1 1 200px !important;
        min-width: 150px !important;
        box-sizing: border-box !important;
    }

    button.gr-button-primary,
    button.gr-button-secondary {
        width: auto !important;
        box-sizing: border-box !important;
    }

    .gradio-container table {
        font-size: 0.8rem !important;
        width: 100% !important;
        table-layout: fixed !important;
    }

    .gradio-container table th,
    .gradio-container table td {
        padding: 4px 6px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
}

@media (max-width: 480px) {
    .gradio-container {
        padding: 12px 8px !important;
    }

    #app-header h1 {
        font-size: 1.2rem !important;
    }

    .card-block {
        padding: 14px 16px !important;
        overflow: hidden !important;
    }

    .viewer-box {
        height: 240px !important;
    }

    input[type="number"],
    input[type="text"] {
        font-size: 0.8rem !important;
    }

    .preset-row {
        flex-wrap: wrap !important;
        gap: 6px !important;
    }

    .preset-row button {
        flex: 1 1 45% !important;
        font-size: 0.75rem !important;
        padding: 5px 8px !important;
    }

    .gradio-container .gr-row > * {
        flex: 1 1 120px !important;
        min-width: 100px !important;
    }
}
"""

HEAD_HTML = """
<script>
(function() {
    var u = new URL(window.location.href);
    if (u.searchParams.get('__theme') !== 'light') {
        u.searchParams.set('__theme', 'light');
        window.location.replace(u.toString());
    }
})();
</script>
"""


with gr.Blocks(
    title="3D Cutter",
) as demo:

    gr.HTML(
        """
        <div id="app-header">
            <h1>3D Cutter</h1>
            <p>Обрезка 3D-моделей по плоскости. Экспорт в TXT / STL / OBJ / PLY / GLB</p>
        </div>
        """
    )

    # просмотр
    with gr.Column(elem_classes="card-block"):
        gr.Markdown(
            "**Просмотр моделей**",
            container=False,
            elem_classes="step-title",
        )
        with gr.Row():
            original_preview = gr.Model3D(
                label="Оригинал",
                clear_color=[0.976, 0.980, 0.984, 1.0],
                height=320,
                elem_classes="viewer-box",
            )
            result_preview = gr.Model3D(
                label="Результат",
                clear_color=[0.976, 0.980, 0.984, 1.0],
                height=320,
                elem_classes="viewer-box",
            )

    # загрузка
    with gr.Column(elem_classes="card-block"):
        gr.Markdown(
            "**Шаг 1.** Загрузите модель",
            container=False,
            elem_classes="step-title",
        )
        file_input = gr.File(
            label="",
            file_types=[".stl", ".obj", ".ply", ".glb", ".gltf", ".txt"],
        )
        model_info = gr.Markdown(
            EMPTY_INFO,
            container=False,
            elem_classes="hint-text info-box",
        )

    # скрытое состояние текущего меша (для последовательной обрезки)
    current_mesh_state = gr.State(None)

    # выбор плоскости
    with gr.Column(elem_classes="card-block"):
        gr.Markdown(
            "**Шаг 2.** Задайте плоскость разреза",
            container=False,
            elem_classes="step-title",
        )

        gr.Markdown(
            "**2.1.** Быстрый выбор нормали:",
            container=False,
            elem_classes="hint-text",
        )
        with gr.Row(elem_classes="preset-row"):
            btn_x = gr.Button("X", size="sm", variant="secondary")
            btn_y = gr.Button("Y", size="sm", variant="secondary")
            btn_z = gr.Button("Z", size="sm", variant="secondary")
            btn_xy = gr.Button("XY", size="sm", variant="secondary")
            btn_xyz = gr.Button("XYZ", size="sm", variant="secondary")

        gr.Markdown(
            "**2.2.** Точка на плоскости:",
            container=False,
            elem_classes="hint-text",
        )
        with gr.Row():
            px = gr.Number(value=0.0, label="X", precision=3)
            py = gr.Number(value=0.0, label="Y", precision=3)
            pz = gr.Number(value=0.0, label="Z", precision=3)

        btn_center = gr.Button(
            "Подставить центр модели",
            size="sm",
            variant="secondary",
        )

        gr.Markdown(
            "**2.3.** Вектор нормали:",
            container=False,
            elem_classes="hint-text",
        )
        with gr.Row():
            nx = gr.Number(value=1.0, label="nx", precision=3)
            ny = gr.Number(value=0.0, label="ny", precision=3)
            nz = gr.Number(value=0.0, label="nz", precision=3)

        gr.Markdown(
            "**2.4.** Какую часть оставить:",
            container=False,
            elem_classes="hint-text",
        )
        keep_side = gr.Radio(
            choices=[
                ("Сторону, куда смотрит нормаль", True),
                ("Противоположную сторону", False),
            ],
            value=True,
            show_label=False,
            container=False,
        )

    # обрезка
    with gr.Column(elem_classes="card-block"):
        gr.Markdown(
            "**Шаг 3.** Обрезать модель",
            container=False,
            elem_classes="step-title",
        )

        gr.Markdown(
            "**3.1.** Последовательная обрезка (одна плоскость):",
            container=False,
            elem_classes="hint-text",
        )
        btn_cut = gr.Button(
            "Обрезать",
            variant="primary",
            size="lg",
            elem_classes="cut-btn",
        )

        gr.Markdown(
            "**3.2.** Пакетная обрезка (несколько плоскостей подряд):",
            container=False,
            elem_classes="hint-text",
        )
        planes_text = gr.Textbox(
            label="Плоскости (по одной на строку: px py pz nx ny nz)",
            placeholder="0 0 0 1 0 0\n0 0 0 0 1 0\n0 0 0 0 0 1",
            lines=5,
        )
        btn_cut_batch = gr.Button(
            "Обрезать по всем плоскостям",
            variant="secondary",
            size="sm",
        )

        btn_reset = gr.Button(
            "Сбросить состояние",
            variant="secondary",
            size="sm",
        )

        status = gr.Markdown(
            "Загрузите модель, задайте плоскость и нажмите «Обрезать».",
            container=False,
            elem_classes="hint-text",
        )

    # скачивание
    with gr.Column(elem_classes="card-block"):
        gr.Markdown(
            "**Шаг 4.** Скачать результат",
            container=False,
            elem_classes="step-title",
        )
        with gr.Row():
            out_txt = gr.File(label="TXT", interactive=False)
            out_stl = gr.File(label="STL", interactive=False)
            out_obj = gr.File(label="OBJ", interactive=False)
        with gr.Row():
            out_ply = gr.File(label="PLY", interactive=False)
            out_glb = gr.File(label="GLB", interactive=False)

    gr.HTML('<div id="app-footer">3D Cutter · Python, Gradio, trimesh</div>')

    file_input.change(
        fn=analyze_model,
        inputs=file_input,
        outputs=[model_info, original_preview, px, py, pz, current_mesh_state],
    )

    btn_center.click(
        fn=analyze_model,
        inputs=file_input,
        outputs=[model_info, original_preview, px, py, pz, current_mesh_state],
    )

    btn_x.click(fn=preset_x, outputs=[nx, ny, nz])
    btn_y.click(fn=preset_y, outputs=[nx, ny, nz])
    btn_z.click(fn=preset_z, outputs=[nx, ny, nz])
    btn_xy.click(fn=preset_xy, outputs=[nx, ny, nz])
    btn_xyz.click(fn=preset_xyz, outputs=[nx, ny, nz])

    btn_cut.click(
        fn=cut_mesh,
        inputs=[file_input, current_mesh_state, px, py, pz, nx, ny, nz, keep_side],
        outputs=[out_txt, out_stl, out_obj, out_ply, out_glb, result_preview, current_mesh_state, status],
    ).then(
        fn=None,
        js="() => window.scrollTo({top: 0, behavior: 'smooth'})",
    )

    btn_cut_batch.click(
        fn=cut_mesh_batch,
        inputs=[file_input, current_mesh_state, planes_text, keep_side],
        outputs=[out_txt, out_stl, out_obj, out_ply, out_glb, result_preview, current_mesh_state, status],
    ).then(
        fn=None,
        js="() => window.scrollTo({top: 0, behavior: 'smooth'})",
    )

    btn_reset.click(
        fn=reset_mesh,
        inputs=[file_input],
        outputs=[current_mesh_state, original_preview, result_preview, status],
    )

if __name__ == "__main__":
    import os
    _cleanup_temp_files()
    demo.launch(
        server_name="0.0.0.0",
        theme=gr.themes.Soft(primary_hue="slate", neutral_hue="slate"),
        css=CUSTOM_CSS,
        head=HEAD_HTML,
    )