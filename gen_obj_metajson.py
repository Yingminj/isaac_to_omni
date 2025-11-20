import os, json, argparse
from collections import defaultdict

def find_meta_files(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn == "meta.json" or fn.endswith("_meta.json"):
                out.append(os.path.join(dp, fn))
    return out

def get_inst_and_cls(k, v):
    meta = v.get("meta", {}) if isinstance(v, dict) else {}
    oid = meta.get("oid")
    if not oid:
        # try instance_path basename
        ip = meta.get("instance_path")
        if ip:
            oid = os.path.splitext(os.path.basename(ip))[0]
        else:
            oid = k
    cls = meta.get("class_name")
    if not cls:
        # infer from oid
        tmp = oid
        if tmp.startswith("real-"):
            tmp = tmp[len("real-"):]
        cls = tmp.split("-")[-1].split("_")[0]
    return oid, cls, meta

def build(data_root):
    files = find_meta_files(data_root)
    classes = defaultdict(set)
    instances = {}
    # stats[class_name][source-split]
    def make_stat():
        return {
            "phocal-train": 0, "phocal-test": 0,
            "omniobject3d-train": 0, "omniobject3d-test": 0,
            "google_scan-train": 0, "google_scan-test": 0,
            "real-train": 0, "real-test": 0
        }
    stats = defaultdict(make_stat)

    def infer_source_from_oid(oid):
        o = oid.lower()
        if o.startswith("phocal-"):
            return "phocal"
        if o.startswith("omniobject3d-"):
            return "omniobject3d"
        if o.startswith("google_scan-") or o.startswith("google-scan-") or o.startswith("google_scan-"):
            return "google_scan"
        # treat others (including real- and owndata) as real
        return "real"

    for f in files:
        try:
            j = json.load(open(f, "r"))
        except Exception as e:
            # Fail fast and surface the problematic file instead of silently skipping.
            raise RuntimeError(f"Failed to load JSON from '{f}': {e}") from e
        objs = j.get("objects", {}) or {}
        for k, v in objs.items():
            oid, cls, meta = get_inst_and_cls(k, v)
            classes[cls].add(oid)
            # determine split from file path
            f_low = f.lower()
            if "/train/" in f_low:
                split = "train"
            elif "/test/" in f_low:
                split = "test"
            elif "/val/" in f_low:
                split = "test"
            else:
                # unknown -> treat as train
                split = "train"
            src = infer_source_from_oid(oid)
            stats[cls][f"{src}-{split}"] += 1
            if oid not in instances:
                # gather available fields
                inst_path = meta.get("instance_path") or ""
                # dimensions / bbox fallback
                dims = meta.get("bbox_side_len") or meta.get("scale") or []
                # keep numeric list
                dimensions = [float(x) for x in dims] if isinstance(dims, (list,tuple)) else []
                name = os.path.splitext(os.path.basename(inst_path))[0] if inst_path else oid
                instances[oid] = {
                    "object_id": oid,
                    "source": "owndata",
                    "name": name,
                    "obj_path": inst_path,
                    "tag": {
                        "datatype": "real",
                        "sceneChanger": False,
                        "symmetry": {"any": False, "x": "none", "y": "none", "z": "none"},
                        "materialOptions": ["raw", "diffuse"],
                        "upAxis": ["y"]
                    },
                    "class_label": None,   # fill later
                    "class_name": cls,
                    "dimensions": dimensions
                }
    # build class_list with labels
    class_items = sorted(classes.items(), key=lambda x: x[0])
    class_list = []
    for i, (cls_name, inst_set) in enumerate(class_items, start=1):
        class_list.append({
            "name": cls_name,
            "label": i,
            "instance_ids": sorted(list(inst_set)),
            "stat": stats.get(cls_name, make_stat())
        })
        for inst in inst_set:
            if inst in instances:
                instances[inst]["class_label"] = i
    return {"class_list": class_list, "instance_dict": instances}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="datasets/basic1117-2/Replicator_03")
    p.add_argument("--out", default="configs/obj_meta_generated.json")
    args = p.parse_args()
    meta = build(args.data_root)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.out} classes={len(meta['class_list'])} instances={len(meta['instance_dict'])}")

if __name__ == "__main__":
    main()