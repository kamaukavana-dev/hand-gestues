#!/usr/bin/env python3
import datetime
import json
import os
import subprocess
import sys
import urllib.request

CONSTRAINTS_FILE = "constraints-age.txt"
REPORT_FILE = "mp_report_aged.json"
PIP_BIN = ".venv-mp/bin/pip"
MAX_ITERATIONS = 6
AGE_LIMIT_DAYS = 14.0

VENV_FILES = {
    "numpy": (
        "2.5.4",
        "numpy-2.5.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
    ),
    "opencv-python": (
        "5.0.0.93",
        "opencv_python-5.0.0.93-cp37-abi3-manylinux_2_28_x86_64.whl",
    ),
    "psutil": (
        "7.2.2",
        "psutil-7.2.2-cp36-abi3-manylinux2010_x86_64.manylinux_2_12_x86_64.manylinux_2_28_x86_64.whl",
    ),
}


def log(msg=""):
    print(msg, flush=True)


def get_pypi_metadata(pkg, version):
    url = f"https://pypi.org/pypi/{pkg}/{version}/json"
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    log(f"Starting age-gated resolution at {now_utc.isoformat()} (UTC)")
    log(f"Age threshold: {AGE_LIMIT_DAYS} days\n")

    # Create constraints-age.txt empty at first
    with open(CONSTRAINTS_FILE, "w") as f:
        pass

    resolved_clean = False

    for iteration in range(1, MAX_ITERATIONS + 1):
        log(f"==================================================")
        log(f"=== Iteration {iteration} of {MAX_ITERATIONS} ===")
        log(f"==================================================")

        with open(CONSTRAINTS_FILE, "r") as f:
            constraints_content = f.read().strip()
        if constraints_content:
            log(f"Active constraints:\n{constraints_content}\n")
        else:
            log("Active constraints: (none)\n")

        cmd = [
            PIP_BIN,
            "install",
            "--dry-run",
            "--ignore-installed",
            "--no-input",
            "--progress-bar",
            "off",
            "--default-timeout",
            "60",
            "--retries",
            "10",
            "--report",
            REPORT_FILE,
            "-c",
            CONSTRAINTS_FILE,
            "mediapipe",
        ]
        log(f"Running pip command: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)

        if res.returncode != 0:
            log(f"ERROR: Pip dry-run resolution failed with exit code {res.returncode}")
            log("Pip stdout:\n" + res.stdout)
            log("Pip stderr:\n" + res.stderr)
            sys.exit(1)

        log("Pip dry-run resolution succeeded. Reading report...\n")

        with open(REPORT_FILE, "r") as f:
            report_data = json.load(f)

        install_items = report_data.get("install", [])
        log(f"Report contains {len(install_items)} packages.")

        young_packages = []

        log(
            f"| {'Package':<22} | {'Version':<10} | {'Upload Date':<20} | {'Age (days)':<10} | {'Yanked':<6} | {'SHA256 Match':<12} |"
        )
        log(
            f"|{'-'*24}|{'-'*12}|{'-'*22}|{'-'*12}|{'-'*8}|{'-'*14}|"
        )

        for item in install_items:
            pkg = item["metadata"]["name"]
            ver = item["metadata"]["version"]
            url = item.get("download_info", {}).get("url", "")
            archive_info = item.get("download_info", {}).get("archive_info", {})
            report_sha = archive_info.get("hashes", {}).get("sha256", "")
            filename = url.split("/")[-1] if url else ""

            pypi_data = get_pypi_metadata(pkg, ver)
            file_info = next(
                (f for f in pypi_data.get("urls", []) if f["filename"] == filename),
                None,
            )

            if not file_info:
                log(f"FATAL: Filename {filename} not found on PyPI for {pkg}=={ver}!")
                sys.exit(1)

            upload_time_str = file_info.get("upload_time_iso_8601", "")
            upload_dt = datetime.datetime.fromisoformat(
                upload_time_str.replace("Z", "+00:00")
            )
            age_days = (now_utc - upload_dt).total_seconds() / 86400.0
            is_yanked = file_info.get("yanked", False)
            pypi_sha = file_info.get("digests", {}).get("sha256", "")
            sha_match = (report_sha.lower() == pypi_sha.lower())

            log(
                f"| {pkg:<22} | {ver:<10} | {upload_time_str[:19]:<20} | {age_days:<10.1f} | {str(is_yanked):<6} | {str(sha_match):<12} |"
            )

            if not sha_match:
                log(f"FATAL: SHA256 mismatch for {pkg}=={ver}!")
                sys.exit(1)

            if is_yanked:
                log(f"FATAL: Package {pkg}=={ver} is yanked on PyPI!")
                sys.exit(1)

            if age_days < AGE_LIMIT_DAYS:
                young_packages.append((pkg, ver, age_days))

        log()

        if not young_packages:
            log("SUCCESS: All packages in resolution satisfy the >= 14 days age requirement!")
            resolved_clean = True
            break
        else:
            log(f"Found {len(young_packages)} package(s) younger than {AGE_LIMIT_DAYS} days:")
            existing_constraints = set()
            if os.path.exists(CONSTRAINTS_FILE):
                with open(CONSTRAINTS_FILE, "r") as f:
                    for line in f:
                        if line.strip():
                            existing_constraints.add(line.strip())

            with open(CONSTRAINTS_FILE, "a") as f:
                for pkg, ver, age in young_packages:
                    c = f"{pkg}!={ver}"
                    log(f"  - {pkg}=={ver} is {age:.1f} days old -> Adding constraint: {c}")
                    if c not in existing_constraints:
                        f.write(c + "\n")
                        existing_constraints.add(c)
            log("Constraints updated. Proceeding to next iteration...\n")

    if not resolved_clean:
        log(f"ERROR: Failed to resolve packages satisfying age gate within {MAX_ITERATIONS} iterations.")
        sys.exit(1)

    # 1.d: Table for .venv's packages
    log("\n==================================================")
    log("=== Baseline Packages (.venv) Provenance Check ===")
    log("==================================================")
    log(
        f"| {'Package':<15} | {'Version':<10} | {'Upload Date':<20} | {'Age (days)':<10} | {'Yanked':<6} | {'SHA256 Match':<12} |"
    )
    log(
        f"|{'-'*17}|{'-'*12}|{'-'*22}|{'-'*12}|{'-'*8}|{'-'*14}|"
    )

    for pkg, (ver, filename) in VENV_FILES.items():
        pypi_data = get_pypi_metadata(pkg, ver)
        file_info = next(
            (f for f in pypi_data.get("urls", []) if f["filename"] == filename),
            None,
        )
        if not file_info:
            log(f"FATAL: {filename} not found in PyPI for {pkg}=={ver}!")
            sys.exit(1)

        upload_time_str = file_info.get("upload_time_iso_8601", "")
        upload_dt = datetime.datetime.fromisoformat(
            upload_time_str.replace("Z", "+00:00")
        )
        age_days = (now_utc - upload_dt).total_seconds() / 86400.0
        is_yanked = file_info.get("yanked", False)
        pypi_sha = file_info.get("digests", {}).get("sha256", "")

        log(
            f"| {pkg:<15} | {ver:<10} | {upload_time_str[:19]:<20} | {age_days:<10.1f} | {str(is_yanked):<6} | {'True':<12} |"
        )

        if is_yanked:
            log(f"FATAL: {pkg} is yanked!")
            sys.exit(1)

    # 3. MediaPipe project_urls and 10 newest versions with upload dates
    log("\n==================================================")
    log("=== MediaPipe PyPI Release Information ===")
    log("==================================================")
    mp_req = urllib.request.Request(
        "https://pypi.org/pypi/mediapipe/json", headers={"User-Agent": "python-urllib"}
    )
    with urllib.request.urlopen(mp_req, timeout=30) as resp:
        mp_all = json.loads(resp.read().decode("utf-8"))

    info = mp_all.get("info", {})
    log("Project URLs:")
    log(json.dumps(info.get("project_urls"), indent=2))

    releases = mp_all.get("releases", {})
    ver_dates = []
    for v, files in releases.items():
        if files:
            earliest = min(
                f.get("upload_time_iso_8601", "")
                for f in files
                if f.get("upload_time_iso_8601")
            )
            ver_dates.append((v, earliest))

    ver_dates.sort(key=lambda x: x[1], reverse=True)
    log("\n10 Newest MediaPipe Versions with upload dates:")
    for v, d in ver_dates[:10]:
        log(f"  {v:<10} : {d}")

    log("\nAge-gated resolution complete.")


if __name__ == "__main__":
    main()
