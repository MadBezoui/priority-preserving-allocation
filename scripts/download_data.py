"""Download original UCI archives, hash them, and prepare chronological traces."""
from pathlib import Path
import hashlib
import json
import zipfile
import datetime
import sys
import requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
SOURCES = {
    "electricity": {
        "url": "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
        "doi": "10.24432/C58C86", "creator": "Artur Trindade (2015)",
        "expected_sha256": "f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1",
        "license": "CC BY 4.0", "page": "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014"},
    "bike": {
        "url": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "doi": "10.24432/C5W894", "creator": "Hadi Fanaee-T (2013)",
        "expected_sha256": "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
        "license": "CC BY 4.0", "page": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"},
}


def download(name, meta):
    target = RAW / (name + ".zip")
    fresh = not target.exists()
    if not target.exists():
        partial = target.with_suffix(".partial")
        with requests.get(meta["url"], stream=True, timeout=(30, 180)) as r:
            r.raise_for_status()
            with partial.open("wb") as f:
                for chunk in r.iter_content(1024*1024):
                    f.write(chunk)
        # Check archive integrity before making cache authoritative.
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None:
                raise RuntimeError("corrupt dataset archive")
        partial.rename(target)
    digest = hashlib.sha256()
    with target.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != meta["expected_sha256"]:
        raise RuntimeError(f"{name}: archive checksum mismatch; refusing changed inputs")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    old_path = ROOT / "data/source_manifest.json"
    old = json.loads(old_path.read_text()).get(name,{}) if old_path.exists() else {}
    retrieved = now if fresh else old.get("retrieved_utc")
    return dict(meta, sha256=digest.hexdigest(), bytes=target.stat().st_size,
                retrieved_utc=retrieved, verified_utc=now, cache_reused=not fresh)


def electricity():
    path = RAW / "electricity.zip"
    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist() if n.endswith("LD2011_2014.txt") and not n.startswith("__MACOSX"))
        sums, count, columns = None, 0, None
        with z.open(member) as f:
            for chunk in pd.read_csv(f, sep=";", decimal=",", index_col=0, chunksize=4096):
                train = chunk.loc[(chunk.index >= "2012-01-01") & (chunk.index < "2013-01-01")]
                if len(train):
                    values = train.to_numpy(float)
                    sums = values.sum(axis=0) if sums is None else sums + values.sum(axis=0)
                    count += len(train)
                    columns = list(chunk.columns)
        means = sums / count
        chosen = np.flatnonzero(means > 0)[:32]
        selected = [columns[i] for i in chosen]
        test, timestamps = [], []
        with z.open(member) as f:
            for chunk in pd.read_csv(f, sep=";", decimal=",", index_col=0,
                                     usecols=[0] + [int(i+1) for i in chosen], chunksize=4096):
                sub = chunk.loc[(chunk.index >= "2013-01-01") & (chunk.index < "2014-01-01")]
                if len(sub):
                    test.append(sub.to_numpy(float) / means[chosen])
                    timestamps.extend(sub.index)
        data = np.concatenate(test)
        assert np.isfinite(data).all() and (data >= 0).all()
        np.savez_compressed(PROCESSED / "electricity.npz", demand=data,
                            timestamps=np.asarray(timestamps), train_total=32.,
                            critical=np.arange(8), columns=np.asarray(selected))
        return {"train_year": 2012, "train_rows": count, "test_year": 2013,
                "test_rows": len(data), "selected_columns": selected,
                "training_means_kw": means[chosen].tolist(),
                "normalization": "each client divided by its 2012 mean kW",
                "critical_policy": "first 8 selected clients; simulated, not observed",
                "sampling": "15 minutes; UCI daylight-saving encoding retained",
                "raw_units": "kW, not kWh", "missing_policy": "retain documented structural zeros"}


def bike():
    with zipfile.ZipFile(RAW / "bike.zip") as z:
        name = next(n for n in z.namelist() if n.endswith("hour.csv"))
        with z.open(name) as f:
            df = pd.read_csv(f)
    train = df[df.yr == 0]
    test = df[df.yr == 1]
    scale = float(train.cnt.mean())
    times = pd.to_datetime(test.dteday) + pd.to_timedelta(test.hr, unit="h")
    gaps = int((times.diff().dropna() != pd.Timedelta(hours=1)).sum())
    values = test[["registered", "casual"]].to_numpy(float) / scale
    np.savez_compressed(PROCESSED / "bike.npz", demand=values,
                        timestamps=times.astype(str).to_numpy(dtype=str),
                        train_total=1., critical=np.array([0]),
                        columns=np.array(["registered", "casual"]))
    return {"train_year": 2011, "train_rows": len(train), "test_year": 2012,
            "test_rows": len(test), "training_total_mean": scale,
            "normalization": "both columns divided by 2011 mean total count",
            "timestamp_gaps": gaps, "missing_policy": "retain observed rows; do not impute missing hours",
            "critical_policy": "registered users; simulated policy, not safety-critical label",
            "measurement": "completed rentals, not uncensored demand"}


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, meta in SOURCES.items():
        print("Downloading/verifying", name, flush=True)
        manifest[name] = download(name, meta)
    (ROOT / "data/source_manifest.json").write_text(json.dumps(manifest, indent=2))
    processing = {"electricity": electricity(), "bike": bike()}
    processing["software"] = {"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__}
    (ROOT / "data/processing_manifest.json").write_text(json.dumps(processing, indent=2))
    print(json.dumps(processing, indent=2))
