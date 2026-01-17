from __future__ import annotations

"""Attack family helpers for NSL-KDD.

NSL-KDD labels are fine-grained (e.g., 'smurf', 'neptune'). Many papers group them
into broader families: DoS, Probe, R2L, U2R, plus Normal.

We use this mapping to create realistic non-IID client splits for federated
learning experiments.
"""

from typing import Literal

AttackFamily = Literal["normal", "dos", "probe", "r2l", "u2r", "unknown"]

# Common NSL-KDD / KDD99 family groupings.
DOS = {
    "back",
    "land",
    "neptune",
    "pod",
    "smurf",
    "teardrop",
    "mailbomb",
    "apache2",
    "processtable",
    "udpstorm",
}

PROBE = {
    "ipsweep",
    "nmap",
    "portsweep",
    "satan",
    "mscan",
    "saint",
}

R2L = {
    "ftp_write",
    "guess_passwd",
    "imap",
    "multihop",
    "phf",
    "spy",
    "warezclient",
    "warezmaster",
    "sendmail",
    "named",
    "snmpgetattack",
    "snmpguess",
    "xlock",
    "xsnoop",
    "worm",
}

U2R = {
    "buffer_overflow",
    "loadmodule",
    "perl",
    "rootkit",
    "httptunnel",
    "ps",
    "sqlattack",
    "xterm",
}


def normalize_label(label: str) -> str:
    return str(label).strip().rstrip(".")


def label_to_family(label: str) -> AttackFamily:
    """Map a raw NSL-KDD label to a coarse attack family."""
    lab = normalize_label(label)
    if lab == "normal":
        return "normal"
    if lab in DOS:
        return "dos"
    if lab in PROBE:
        return "probe"
    if lab in R2L:
        return "r2l"
    if lab in U2R:
        return "u2r"
    return "unknown"
