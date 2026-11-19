#!/usr/bin/env python3
import datetime
import json
import sys
import urllib.request

MP_REPORT = "mp_report_aged.json"
PSUTIL_REPORT = "psutil_report.json"
OUTPUT_REQS = "requirements-mp.txt"
MIN_AGE_DAYS = 14.0


def fetch_pypi_meta(pkg, version):
    url = f"https://pypi.org/pypi/{pkg}/{version}/json"
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    print(f"Age verification timestamp (UTC): {now_utc.isoformat()}")

    entries = []

    # Load MediaPipe aged report
    with open(MP_REPORT, "r") as f:
        mp_data = json.load(f)
    for item in mp_data.get("install", []):
        pkg = item["metadata"]["name"]
        ver = item["metadata"]["version"]
        sha = item["download_info"]["archive_info"]["hashes"]["sha256"]
        url = item["download_info"]["url"]
        filename = url.split("/")[-1]
        entries.append({
            "name": pkg,
            "version": ver,
            "sha256": sha,
            "filename": filename,
        })

    # Load psutil report
    with open(PSUTIL_REPORT, "r") as f:
        psutil_data = json.load(f)
    for item in psutil_data.get("install", []):
        pkg = item["metadata"]["name"]
        ver = item["metadata"]["version"]
        sha = item["download_info"]["archive_info"]["hashes"]["sha256"]
        url = item["download_info"]["url"]
        filename = url.split("/")[-1]
        entries.append({
            "name": pkg,
            "version": ver,
            "sha256": sha,
            "filename": filename,
        })

    print(f"\nTotal packages to evaluate: {len(entries)}")
    print(f"{'Package':<22} | {'Version':<10} | {'Upload Date (UTC)':<20} | {'Age (days)':<10} | {'Yanked':<6} | {'SHA256 Match':<12}")
    print(f"{'-'*22}-|-{'-'*10}-|-{'-'*20}-|-{'-'*10}-|-{'-'*6}-|-{'-'*12}")

    too_young = []
    mismatched = []
    yanked = []
    req_lines = []

    for entry in entries:
        pkg = entry["name"]
        ver = entry["version"]
        sha = entry["sha256"]
        fname = entry["filename"]

        meta = fetch_pypi_meta(pkg, ver)
        file_info = next((f for f in meta.get("urls", []) if f["filename"] == fname), None)
        if not file_info:
            print(f"FATAL: {fname} not found on PyPI for {pkg}=={ver}!")
            sys.exit(1)

        upload_str = file_info.get("upload_time_iso_8601", "")
        upload_dt = datetime.datetime.fromisoformat(upload_str.replace("Z", "+00:00"))
        age_days = (now_utc - upload_dt).total_seconds() / 86400.0
        is_yanked = file_info.get("yanked", False)
        pypi_sha = file_info.get("digests", {}).get("sha256", "")
        sha_match = (sha.lower() == pypi_sha.lower())

        if age_days < MIN_AGE_DAYS:
            too_young.append((pkg, ver, age_days))
        if not sha_match:
            mismatched.append((pkg, ver))
        if is_yanked:
            yanked.append((pkg, ver))

        print(f"{pkg:<22} | {ver:<10} | {upload_str:<20} | {age_days:<10.1f} | {str(is_yanked):<6} | {str(sha_match):<12}")
        req_lines.append(f"{pkg}=={ver} --hash=sha256:{sha}\n")

    if too_young:
        print(f"\nERROR: Found {len(too_young)} package(s) younger than 14 days:")
        for p, v, a in too_young:
            print(f"  - {p}=={v}: {a:.1f} days")
        sys.exit(1)

    if mismatched:
        print(f"\nERROR: Found {len(mismatched)} SHA256 mismatches:")
        for p, v in mismatched:
            print(f"  - {p}=={v}")
        sys.exit(1)

    if yanked:
        print(f"\nERROR: Found {len(yanked)} yanked packages:")
        for p, v in yanked:
            print(f"  - {p}=={v}")
        sys.exit(1)

    with open(OUTPUT_REQS, "w") as f:
        f.writelines(req_lines)

    print(f"\nSUCCESS: All {len(entries)} packages satisfy >= 14 days age requirement.")
    print(f"Wrote {len(req_lines)} hash-pinned entries to {OUTPUT_REQS}.")


if __name__ == "__main__":
    main()
